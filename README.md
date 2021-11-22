# COMPUTATIONAL-INTELLIGENCE

Labs and exercises from the course `Computational Intelligence`

## Traveling Salesman Problem
The file `tsp.py` contains a genetic algorithm for the solution of the traveling salesman problem in Python.
Different crossover types and mutation types are implemented and experimented.

## Connect-4
The file `connect_four.py` contains an algorithm playing Connect-4. 
#### Algorithm
The algorithm consists of a Minimax search with limited depth and alpha-beta pruning. Possible moves are represented by a tree, and all nodes of the tree are expanded until a depth limit is reached. Then, the value of a node is estimated through a Monte Carlo evaluation (average the results of a specified number of random simulations starting from that node), and the node with the highest value is selected. Two pruning approaches are also implemented:
* alpha-beta pruning: when the value of a node is surely worse than the value of a previously evaluated node, the expansion of that node is stopped
* when a player finds a move that will lead to a sure win (not necessarily with the lowest number of moves), the search is stopped
#### Data structure
Initially, the playing board is represented using a numpy array. An empty cell contains a 0, the maximizing player places 1s and the minimizing player places -1s in the board. However, this structure is not efficient and slows down the search. Therefore, a more optimized representation has been implemented, inspired by [this article](https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0) (which provides a more detailed description of the representation).
In particular, the board is represented as a sequence of bits that are then converted to integer numbers. Two variables are used:
* `mask`: empty cells are 0, full cells are 1
* `position`: cells occupied by the player currently playing are 1, the others are 0 (note that the position of the stones of the other player can be obtained with `mask XOR position`)
In this way, operations on the board such as checking if a player has won, adding a cell etc. can be performed using bit-wise operations which are very efficient.
#### Modalities
The program allows to play agains the algorithm or to have two instances of the algorithm playing agains each other, by setting the appropriate parameters.