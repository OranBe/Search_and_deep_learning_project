# Decision Making by Methods of Search and Deep Learning Course Final Project
## Comparative Analysis of Machine Learning Based Heuristic using Minimax Algorithm in solveing Connect Four Game

### Overview 
In the following project we aim to develop a machine learning based heuristic using a neural network model for the Minimax algorithm when solving instances of connect four games.
Our objective is to train a neural network to evaluate the values of a connect four game states, creating a heuristic that can enhance the performance of the Minimax algorithm. 

### Description of the above file: 
1. *connect_four_game* - a grid-based board where two players alternately drop discs into columns. The implementation contains all relevant functionality such as check winning or tie conditions, insertion of disks, roles alterations and more.
2. *minimax_algorithm* - implementation of Minimax algorithm using alpha-beta pruning.
3. *heuristics* - implementation of the main heuristic based on neural network and two baseline heuritics: Random and Base.
4. *training* - file containing the heuristic's training process using Bootstrapping method.
5. *bootstrapping_connect_four_heuristic_five* - trained model for a 5x5 game board.
6. *bootstrapping_connect_four_heuristic* - trained model for a 6x7 game board.
7. *empiricalAnalysis* - experiments framework.
