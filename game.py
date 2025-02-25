import chess 
import chess.pgn
import chess.engine
import pandas as pd
import helpers
import numpy as np
import multiprocessing as mp
import tensorflow
import os

from io import StringIO
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

chess_data = pd.read_csv("game_data.csv")
stockfish_path = "stockfish/stockfish.exe"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

# Convert data from csv into chess pgn format
def create_pgn_game(moves):
    pgn_game = chess.pgn.read_game(StringIO(moves))
    return pgn_game

# Collects the positions at each step of the game, the corresponding stockfish generated evaluations and game outcome
def extract_game_info(game, outcome):
    positions = []
    evaluations = []
    results = []

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        positions.append(board.fen())
        results.append(outcome)
        evaluations.append(get_stockfish_eval(board))

    return positions, evaluations, results

def get_stockfish_eval(board):
    analysis = engine.analyse(board, chess.engine.Limit(depth=10))
    return analysis["score"].relative.score(mate_score=5000)

def process_row(lines, result):
    game = create_pgn_game(lines)
    return extract_game_info(game, result)

def process_data():
    df = pd.read_csv("game_data.csv", nrows=100, usecols=["lines", "result"], dtype={"lines": "str", "result": "category"})
    print("Data loaded successfully")

    with mp.Pool(mp.cpu_count()) as pool:
        print("Starting multiprocessing")
        results = pool.starmap(process_row, zip(df["lines"].values, df["result"].values))
        print("Finished multiprocessing")

    print("Processing complete")
    all_positions, all_evaluations, all_results = map(np.concatenate, zip(*results))
    return all_positions, all_evaluations, all_results

# Each feature vector is consisted of the following:
#   - Number of pieces remaining
#   - Material balance (difference in piece value of white and black)
#   - Mobility (number of legal moves available)
#   - Board control (number of pieces which control the squares d4, e4, d5, e5)
#   - White King safety (1 if castled, 0 if not)
#   - Black King safety 
def extract_features(position):
    board = chess.Board(position)
    features = []

    # Number of pieces remaining
    features.append(len(board.piece_map().items()))

    # Material balance (difference in piece value of white and black)
    features.append(helpers.get_material_balance(board))

    # Mobility (number of legal moves available)
    features.append(len(list(move for move in board.legal_moves)))

    # Board control (number of pieces which control the squares d4, e4, d5, e5)
    features.append(helpers.get_board_control(board))

    # King safety (1 if castled, 0 if not)
    features.append(helpers.get_king_safety(board))

    return features

def train_evaluations():
    all_positions, all_evaluations, all_results = process_data()

    x = np.array([extract_features(position) for position in all_positions])
    y = np.array(all_evaluations)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Creating a sequential NN
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'), 
        Dropout(0.3),
        Dense(16, activation='relu'),  
        Dense(1) 
    ])

    # Compiling model
    model.compile(optimizer='adam', loss="mean_squared_error",
                  metrics=["accuracy"])

    # Training model
    model.fit(x_train, y_train, epochs=30)

    # Evaluating model using MSE
    mse = model.evaluate(x_test, y_test)
    print(f"Mean Squared Error: {mse}")
    model.save("chess_evaluation_model.h5")

    # Quitting stockfish
    engine.quit()

if __name__ == "__main__":
    train_evaluations()