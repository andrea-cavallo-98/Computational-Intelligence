# COMPUTATIONAL-INTELLIGENCE

Labs and exercises from the course `Computational Intelligence`

## Traveling Salesman Problem
The file `tsp.py` contains a genetic algorithm for the solution of the traveling salesman problem in Python.
Different crossover types and mutation types are implemented and experimented.

## Connect-4
The file `connect_four.py` contains an algorithm playing Connect-4. 
#### Algorithm
The algorithm consists of a Minimax search with limited depth and alpha-beta pruning. When the depth limit is reached, the value of a node is estimated through a Monte Carlo evaluation (average the results of a specified number of random simulations), and the node with the highest value is selected. 
#### Data structure
Initially, the playing board is represented using a numpy array. An empty cell contains a 0, the maximizing player places 1s and the minimizing player places -1s in the board. However, this structure is not efficient and slows down the search. Therefore, a more optimized representation has been implemented, inspired by [this article](https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0).
In particular, the board is represented as a sequence of bits that are then converted to integer numbers. Two numbers are used:
* `mask`: empty cells are 0, full cells are 1
* `position`: cells occupied by one of the two player are 1, the others are 0
In this way, operations on the board such as checking if a player has won, adding a cell etc. can be performed using bit-wise operations which are very efficient.
#### Modalities
The program allows to play agains the algorithm or to have two instances of the algorithm playing agains each other, by setting the appropriate parameters.