Note: This is an ongoing project, and as my first AI/deep learning project, there may be areas for improvement.

---- Overview 
This repository contains a Chess AI capable of evaluating positions and playing games using various algorithms. The AI improves its 
performance over time by learning from historical game data and Stockfish evaluations. A neural network is trained to predict evaluation 
scores based on extracted game features.

---- Key Features
Position Evaluation: Uses historical data and Stockfish to generate evaluation scores for different board positions.
Feature Engineering: Extracts key features that influence a game's outcome, such as king safety and material balance.
AI Algorithms: Implements multiple techniques for move decision-making, including:
  - Minimax
  - Alpha-Beta Pruning
  - Neural Network (built using TensorFlow)
      - A sequential model with multiple dense layers, incorporating dropout layers to prevent overfitting.
      - The number of neurons decreases per layer to capture complex patterns in board positions effectively.
        
---- Current Issues
The Minimax algorithm is computationally expensive, causing significant delays in move calculation.

---- Future Goals & Improvements
- Replace Minimax with a neural network-based approach or supervised learning to improve move selection speed and efficiency.
- Experiment with different neural network architectures, activation functions, and hyperparameters (e.g., number of neurons, dropout rates)
  by comparing mean squared error (MSE) across models.

---- Usage
Train the Neural Network: Run game.py to train the model.
Play Against the AI: Run ai.py to play a game. (Note: The Minimax algorithm is currently inefficient, making the AI unplayable due to long 
computation times.)
