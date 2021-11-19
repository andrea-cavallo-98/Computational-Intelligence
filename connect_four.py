import numpy as np

NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
FOUR = 4
MAX_DEPTH = 6 * 7

######
## Some settings for the run 
######

# NB: to perform complete minimax search, set depth to MAX_DEPTH
# to perform Monte Carlo Tree Search, set depth to 1
HUMAN_PLAYING = False # Set to True to play against ai, set to False to play ai vs ai
depth_ai_1 = 3 # Depth for minimax search of first ai (must be between 1 and MAX_DEPTH)
depth_ai_2 = 4 # Depth for minimax search of second ai (must be between 1 and MAX_DEPTH)
num_samples_ai_1 = 100 # Number of samples for montecarlo evaluation of first ai 
num_samples_ai_2 = 100 # Number of samples for montecarlo evaluation of second ai
ai_playing = True # Set to True if ai has the first move, False otherwise

######
## Basic functions
######

## Using the board representation

def valid_moves(board):
    """Returns columns where a disc may be played"""
    return [n for n in range(NUM_COLUMNS) if board[n, COLUMN_HEIGHT - 1] == 0]


def play(board, column, player):
    """Updates `board` as `player` drops a disc in `column`"""
    (index,) = next((i for i, v in np.ndenumerate(board[column]) if v == 0))
    board[column, index] = player


def take_back(board, column):
    """Updates `board` removing top disc from `column`"""
    (index,) = [i for i, v in np.ndenumerate(board[column]) if v != 0][-1]
    board[column, index] = 0


def four_in_a_row(board, player):
    """Checks if `player` has a 4-piece line"""
    return (
        any(
            all(board[c, r] == player)
            for c in range(NUM_COLUMNS)
            for r in (list(range(n, n + FOUR)) for n in range(COLUMN_HEIGHT - FOUR + 1))
        )
        or any(
            all(board[c, r] == player)
            for r in range(COLUMN_HEIGHT)
            for c in (list(range(n, n + FOUR)) for n in range(NUM_COLUMNS - FOUR + 1))
        )
        or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + FOUR), range(co, co + FOUR))
                for ro in range(0, NUM_COLUMNS - FOUR + 1)
                for co in range(0, COLUMN_HEIGHT - FOUR + 1)
            )
        )
        or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + FOUR), range(co + FOUR - 1, co - 1, -1))
                for ro in range(0, NUM_COLUMNS - FOUR + 1)
                for co in range(0, COLUMN_HEIGHT - FOUR + 1)
            )
        )
    )


def print_board(board):
  """Prints out the board"""
  for c in range(7):
    print(' ' + str(c), end=' ')
  print()
  print()
  for c in range(5,-1,-1):
    for r in range(7):
      if (board[r, c] == 1):
        print(' X ', end='')
      elif (board[r, c] == -1):
        print(' O ', end='')
      elif board[r,c] == 0:
        print(' - ', end='')
      else:
        print(' ? ', end='')
    print()

## Using the bitmap representation

def get_position_mask_bitmap(board, player):
    """From the board, generate the bitmap representation"""
    position, mask = '', ''
    # Start with right-most column
    for j in range(6, -1, -1):
        # Add 0-bits to sentinel 
        mask += '0'
        position += '0'
        # Start with bottom row
        for i in range(5, -1, -1):
            mask += ['0', '1'][board[j, i] != 0]
            position += ['0', '1'][board[j, i] == player]
    return int(position, 2), int(mask, 2)



def connected_four(position):
    """Check if someone has won using the bitmap configuration"""
    # Horizontal check
    m = position & (position >> 7)
    if m & (m >> 14):
        return True
    # Diagonal \
    m = position & (position >> 6)
    if m & (m >> 12):
        return True
    # Diagonal /
    m = position & (position >> 8)
    if m & (m >> 16):
        return True
    # Vertical
    m = position & (position >> 1)
    if m & (m >> 2):
        return True
    # Nothing found
    return False


def get_valid_moves(mask):
  """Get available columns to play using bitmap representation"""
  valid_moves = [c for c in range(7) if not (mask & (0b100000 << c * 7))]
  return valid_moves


def make_move(position, mask, col):
    """Add a pawn to the specified column using bitmap representation"""
    new_position = position ^ mask
    new_mask = mask | (mask + (1 << (col*7)))
    return new_position, new_mask


######
## Monte Carlo Evaluation
######

def _mc(position, mask, player):
  """Perform random simulation"""
  p = -player
  while get_valid_moves(mask):
      #print("{0:b}".format(mask) + " " +"{0:b}".format(position))
      p = -p
      c = int(np.random.choice(get_valid_moves(mask)))
      new_position, new_mask = make_move(position, mask, c)
      if connected_four(new_position ^ new_mask):
          return p
      position = new_position
      mask = new_mask
  return 0


def montecarlo(position, mask, player, num_samples):
  """Evaluate cell by running random simulations"""
  montecarlo_samples = num_samples
  cnt = 0
  for _ in range(montecarlo_samples):
    cnt += _mc(position, mask, player)
  return cnt / montecarlo_samples


######
## MinMax algorithm
######

def minimax(position, mask, depth, alpha, beta, player, num_samples):
  """Implementation of the minmax algorithm with alpha-beta pruning and 
  Monte Carlo evaluation of the nodes at the specified depth"""
  if connected_four(mask ^ position): # previous player has won
	  return (None, -player * 1)
	
  if mask == 0b1111110111111011111101111110111111011111101111110: # no more available moves, it's a draw!
	  return (None, 0)

  if depth == 0: # Evaluate node using Monte Carlo evaluation
    return (None, montecarlo(position, mask, player, num_samples)) 

	# if it's the first move, play center column (proven to be best initial move)
  if mask == 0:
    return (3, 0)

  # initialize values  
  value = -player * 2 # either +2 or -2
  column = 0 
  
  # perform valid moves and recursively analyze their effect
  for col in get_valid_moves(mask):
    new_position, new_mask = make_move(position, mask, col)
    new_score = minimax(new_position, new_mask, depth-1, alpha, beta, -player, num_samples)[1]
    if (player == 1 and new_score > value) or (player == -1 and new_score < value): # the new score has improved the previous one
      value = new_score
      column = col
    # Update alpha or beta
    if player == 1:
      alpha = max(alpha, value)
    else:
      beta = min(beta, value)
    if alpha >= beta: # alpha-beta pruning
      break			
    if (player == 1 and alpha == 1) or (player == -1 and beta == -1): # no better result can be achieved, stop exploration
      break
 
  return column, value



if __name__ == "__main__":
  
  ######
  ## Play the game (either human vs ai or ai vs ai)
  ######

  winner = 0
  board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)

  while winner == 0:

    print_board(board)
    print()
    
    if ai_playing: # ai's turn -> maximizing player
      position, mask = get_position_mask_bitmap(board, 1)
      col, _ = minimax(position, mask, depth_ai_1, -1000, 1000, 1, num_samples_ai_1)
      play(board, col, 1)
    else: # -> minimizing player
      if HUMAN_PLAYING: # human's turn
        print("Insert column where you want to play: ", end='')
        col = int(input())
        print()
      else: # second ai's turn
        position, mask = get_position_mask_bitmap(board, -1)
        col, _ = minimax(position, mask, depth_ai_2, -1000, 1000, -1, num_samples_ai_2)
      play(board, col, -1)

    # check for winners
    if four_in_a_row(board, 1):
      winner = 1
    elif four_in_a_row(board, -1):
      winner = -1

    # move on to next turn
    ai_playing = not ai_playing

    if len(valid_moves(board)) == 0: # no winners, it's a draw
      winner = -2

  # Print final configuration and winner
  print_board(board)
  print()
  if winner == 1:
    if HUMAN_PLAYING:
      print("Computer wins!")
    else:
      print("Computer 1 wins!")
  elif winner == -1:
    if HUMAN_PLAYING:
      print("Human wins!")
    else:
      print("Computer 2 wins!")
  else:
    print("It's a draw!")
