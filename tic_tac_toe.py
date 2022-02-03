import numpy as np
from itertools import permutations
from tqdm import tqdm

###
# Configurable parameters for the game
###
GAMMA = 0.9
ALPHA = 0.7
ITERATIONS = 100_000
AI_FIRST = False # Who plays first: True for AI, False for human
EPS = 0.5 # Probability of exploration vs exploitation
AGAINST_RANDOM = False # True to train the AI against a random player, False to train the AI against a reasonable player

# Global variables
TICTACTOE_MAP = np.array([[1, 6, 5], [8, 4, 0], [3, 2, 7]])
PLAYING = False 

def map_board(x, o):
    return str(sorted(x))[1:-1].replace(' ','') + "|" + str(sorted(o))[1:-1].replace(' ','')

def display(x, o):
    for r in range(3):
        for c in range(3):
            if TICTACTOE_MAP[r, c] in x:
                print("X", end=" ")
            elif TICTACTOE_MAP[r, c] in o:
                print("O", end=" ")
            else:
                print(".", end=" ")

        print(" || " + str(r * 3 + 1) + " " + str(r * 3 + 2) + " " + str(r * 3 + 3))
        print()

def won(cells):
    return any(sum(h) == 12 for h in permutations(cells, 3))

def eval_terminal(x, o):
    if won(x):
        return 1
    elif won(o):
        return -1
    else:
        return 0

def play_as_opponent(x, o, v):
    for c in range(9):
        if c not in x or c in o:
            val = 0
            try:
                val = v[map_board(x, o + [c])]
            except:
                pass
            if val == -1:
                return o + [c]
    best_move = np.random.randint(9)
    while best_move in x or best_move in o:
        best_move = np.random.randint(9)
    return o + [best_move]

def play(x, o, v, rand_explore = True):
    max_val = -2
    best_move = np.random.randint(9)
    while best_move in x or best_move in o:
        best_move = np.random.randint(9)
    if np.random.rand() < EPS or not rand_explore:
        for c in range(9):
            if c not in x and c not in o:
                val = 0
                try:
                    val = v[map_board(x + [c], o)]
                except:
                    v[map_board(x + [c], o)] = 0
                    continue
                if val > (max_val + 1e-5):
                    best_move = c
                    max_val = val
    if PLAYING:
        print(f"This move has value: {max_val}")
    return x + [best_move]

def update_value(v, x, o, x1, o1, r):

    if x1 == None and o1 == None:
        v[map_board(x,o)] = r
    else:
        if r == 0:
            if map_board(x,o) in v:
                if map_board(x1, o1) in v:
                    v[map_board(x,o)] = (1 - ALPHA) * v[map_board(x,o)] + ALPHA * (GAMMA * v[map_board(x1, o1)])
            else:
                if map_board(x1, o1) in v:
                    v[map_board(x,o)] = ALPHA * (GAMMA * v[map_board(x1, o1)])
        else:
            if map_board(x,o) in v:
                v[map_board(x,o)] = (1 - ALPHA) * v[map_board(x,o)] + ALPHA * r
            else:
                v[map_board(x,o)] = ALPHA * r

def train(n_iterations, ai_first):
    v = {}

    print("*** TRAINING ***")
    for _ in tqdm(range(n_iterations)):
        x = []
        o = []
        
        if not ai_first: # Random player makes first move
            o = o + [np.random.choice(9)]
        reward = eval_terminal(x, o)
        while reward == 0 and len(x) + len(o) < 9:
            x1 = play(x, o, v)
            reward = eval_terminal(x1, o)
            update_value(v, x, o, x1, o, reward)
            x = x1
            if reward != 0 or len(x) + len(o) >= 9:
                update_value(v, x, o, None, None, reward)
                break
            if AGAINST_RANDOM:
                o1 = np.random.choice(9)
                while o1 in o or o1 in x:
                    o1 = np.random.choice(9)
                o1 = o + [o1]
            else:
                o1 = play_as_opponent(x, o, v)
            reward = eval_terminal(x, o1)
            update_value(v, x, o, x, o1, reward)
            if reward == -1 or len(x) + len(o) >= 9:
                update_value(v, x, o1, None, None, reward)
            o = o1
        
    return v

def human_plays(x, o):
    print("Insert cell where you want to play: ", end='')
    o1 = int(input()) - 1
    while o1 < 0 or o1 >= 9:
        print("Inserted cell is not valid!")
        print("Insert cell where you want to play: ", end='')
        o1 = int(input()) - 1
    o = o + [TICTACTOE_MAP[o1 // 3, o1 % 3]]
    print()
    return o


if __name__ == "__main__":

    v = train(ITERATIONS, AI_FIRST)
    
    PLAYING = True
    x = []
    o = []    
    terminal = eval_terminal(x, o)
    while terminal == 0 and len(x) + len(o) < 9:
        display(x, o)
        if AI_FIRST:
            x1 = play(x, o, v, rand_explore=False)
            terminal = eval_terminal(x1, o)
            x = x1
            print()
            if terminal != 0 or len(x) + len(o) >= 9:
                break
            display(x, o)
            o = human_plays(x, o)
            terminal = eval_terminal(x, o)
        else:
            o = human_plays(x, o)
            terminal = eval_terminal(x, o)
            if terminal != 0 or len(x) + len(o) >= 9:
                break
            display(x, o)
            x1 = play(x, o, v, rand_explore=False)
            terminal = eval_terminal(x1, o)
            x = x1
            print()


    display(x,o)
    if terminal == 1:
        print("Computer wins!")
    elif terminal == -1:
        print("Human wins!")
    else:
        print("It's a draw!")
