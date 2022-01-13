
from game import Game
from genetic_player import Player
import GameData
import random

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


# Define some strategies for mixed evaluation
possible_strategies = [
    [12.0, 1.0, 15.0, 6.0, 26.0, 25.0, 10.0, 24.0, 0.0, 22.0, 13.0, 2.0, 11.0, 9.0, 21.0, 23.0, 4.0, 5.0, 8.0, 19.0, 
        20.0, 14.0, 16.0, 7.0, 18.0, 17.0, 28.0, 3.0, 27.0],
    [9.0, 4.0, 25.0, 11.0, 20.0, 21.0, 22.0, 16.0, 19.0, 17.0, 14.0, 12.0, 24.0, 13.0, 2.0, 28.0, 8.0, 6.0, 27.0, 10.0, 
        23.0, 15.0, 3.0, 5.0, 0.0, 7.0, 1.0, 18.0, 26.0],
    [3.0, 12.0, 24.0, 20.0, 2.0, 25.0, 11.0, 18.0, 5.0, 0.0, 9.0, 15.0, 22.0, 7.0, 8.0, 10.0, 27.0, 4.0, 1.0, 14.0, 
        17.0, 13.0, 23.0, 16.0, 6.0, 28.0, 19.0, 26.0, 21.0]
]


def evaluate_player(it, strategy, evaluation_type = "mirror"):
    ### Evaluate player
    score = 0
    range_players = range(2,3)
    for NUM_PLAYERS in range_players:
        for _ in range(it):
            game = Game()
            players = []
            players_strategies = []

            for id in range(NUM_PLAYERS):    
                players.append(Player(id, NUM_PLAYERS))
                game.addPlayer(str(id))
                if evaluation_type == "mixed":
                    players_strategies.append(random.choice(possible_strategies))
                elif evaluation_type == "mirror":
                    players_strategies.append(strategy)

            # Assign strategy under evaluation to a random player
            if evaluation_type == "mixed": 
                players_strategies[random.randint(len(players_strategies))] = strategy
 
            game.start()

            ### Play game
            current_player = 0
            while not game.isGameOver():
                
                state, _ = game.satisfyRequest(GameData.ClientGetGameStateRequest(str(current_player)), str(current_player))
                #print_state(state)
                data = players[current_player].play(state, players_strategies[current_player])

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
    print(evaluate_player(100, [10.0, 1.0, 6.0, 15.0, 12.0, 21.0, 5.0, 22.0, 9.0, 18.0, 13.0, 8.0, 28.0, 3.0, 16.0, 7.0, 23.0, 17.0, 2.0, 20.0, 27.0, 26.0, 4.0, 24.0, 14.0, 25.0, 19.0, 0.0, 11.0]))
    print(evaluate_player(100, [12.0, 1.0, 15.0, 6.0, 26.0, 25.0, 10.0, 24.0, 0.0, 22.0, 13.0, 2.0, 11.0, 9.0, 21.0, 23.0, 4.0, 5.0, 8.0, 19.0, 20.0, 14.0, 16.0, 7.0, 18.0, 17.0, 28.0, 3.0, 27.0]))
    #print(evaluate_player(1, [0,10,9,19,11,21, 14,15,16, 24, 4, 25, 5, 26, 6, 27]))
    