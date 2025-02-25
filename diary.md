Progress

Day 1 
Set up evaluate_board using simple assignment of values to each piece

Day 2
Created a minimax function 

Day 3
Increased efficiency by including alpha-beta pruning
Starting to implement machine learning 
Decided to use machine learning to improve evaluations of the board state (which will then be used in the minimax function)

Day 4
Discovered feature engineering - started writing out different numerical things e.g. num pieces

Day 5
Decided to train model on randomforest regressor as it is easy to tune and interpret + is based on numerical data
Struggled to read the data from the csv correctly into pgn game

Day 6
Stockfish was taking too long to run, so i tried to research ways to improve efficiency
- Used chunks when reading in the csv data
- Multiprocessing (parellising stockfish operations)


The Neural Network (NN) acts as a better evaluation function for Minimax. Instead of manually assigning piece values (1 for Pawn, 9 for Queen, etc.), the NN learns to evaluate board positions based on data from real games.




---------------------------------------------------------------------------------------------------

Things i need to do 

Minimax: This is the search algorithm used to explore possible future moves in the game and choose the best move based on a given evaluation function.

Deep Learning: This is a technique for building models (e.g., neural networks) that can learn complex patterns, and it's used to evaluate board positions more effectively.

Supervised Learning: This is how you train the neural network, by using labeled data (like grandmaster games) to learn the mapping between board states and evaluations. (machine learning feature engineering - preprocessing data)

Reinforcement learning?? 

Pygame: to make the board come to life when playing it

Make into frontend website?
