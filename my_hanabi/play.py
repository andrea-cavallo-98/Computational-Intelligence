
from cgi import print_directory
from game import Game
from genetic_player import Player
import GameData

def print_state(data):
    print("Current player: " + data.currentPlayer)
    print("Player hands: ")
    for p in data.players:
        print(p.toClientString())
    print("Table cards: ")
    for pos in data.tableCards:
        print(pos + ": [ ")
        for c in data.tableCards[pos]:
            print(c.toClientString() + " ")
        print("]")
    print("Discard pile: ")
    for c in data.discardPile:
        print("\t" + c.toClientString())            
    print("Note tokens used: " + str(data.usedNoteTokens) + "/8")
    print("Storm tokens used: " + str(data.usedStormTokens) + "/3")


def evaluate_player(it, strategy):
    ### Evaluate player
    score = 0
    range_players = range(2,6)
    for NUM_PLAYERS in range_players:
        for _ in range(it):
            game = Game()
            players = []

            for id in range(NUM_PLAYERS):    
                players.append(Player(id, NUM_PLAYERS))
                game.addPlayer(str(id))

            game.start()

            ### Play game
            current_player = 0
            while not game.isGameOver():
                
                state, _ = game.satisfyRequest(GameData.ClientGetGameStateRequest(str(current_player)), str(current_player))
                #print_state(state)
                data = players[current_player].play(state, strategy)

                if type(data) == GameData.ClientHintData: # first perform action, then update other players' states
                    singleData, multipleData = game.satisfyRequest(data, str(current_player))
                    for p in range(len(players)):
                        if p != current_player:
                            players[p].update_other_players(multipleData, None)
                else: # first update other players' states, then perform action
                    for p in range(len(players)):
                        if p != current_player:
                            players[p].update_other_players(data, state)
                    singleData, multipleData = game.satisfyRequest(data, str(current_player))

                if singleData is not None:
                    if type(singleData) is GameData.ServerActionInvalid:
                        print(singleData.message)
                    if type(singleData) is GameData.ServerInvalidDataReceived:
                        print(singleData.data)
                if multipleData is not None:
                    if type(multipleData) is GameData.ServerGameOver:
                        score += multipleData.score
                

                #input()

                # Move on to next turn
                current_player = (current_player + 1) % NUM_PLAYERS
    return - score / (it * len(range_players))

if __name__ == "__main__":
    print(evaluate_player(100, [2.0, 5.0, 14.0, 1.0, 10.0, 20.0, 12.0, 0.0, 6.0, 13.0, 25.0, 11.0, 7.0, 8.0, 21.0, 23.0, 22.0, 27.0, 3.0, 17.0, 4.0, 24.0, 16.0, 15.0, 26.0, 9.0, 19.0, 18.0]))
    print(evaluate_player(100, [5.0, 10.0, 4.0, 1.0, 14.0, 11.0, 0.0, 26.0, 22.0, 8.0, 20.0, 3.0, 2.0, 18.0, 6.0, 19.0, 21.0, 27.0, 24.0, 13.0, 15.0, 9.0, 23.0, 12.0, 17.0, 25.0, 16.0, 7.0]))
    #print(evaluate_player(1, [0,10,9,19,11,21, 14,15,16, 24, 4, 25, 5, 26, 6, 27]))
    