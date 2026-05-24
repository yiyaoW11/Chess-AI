import chess 
import chess.pgn
import chess.engine
import pandas as pd
import helpers
import numpy as np
import multiprocessing as mp
import tensorflow
import os
import helpers

from io import StringIO
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

stockfish_path = "stockfish/stockfish.exe"
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    return _engine

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
    engine = get_engine()
    analysis = engine.analyse(board, chess.engine.Limit(depth=6))
    return analysis["score"].relative.score(mate_score=5000)

def process_row(lines, result):
    game = create_pgn_game(lines)
    return extract_game_info(game, result)

def process_data():
    df = pd.read_csv("game_data.csv", nrows=5000, usecols=["lines", "result"], dtype={"lines": "str", "result": "category"})
    df = df.dropna(subset=["lines"])
    print("Data loaded successfully")

    with mp.Pool(mp.cpu_count()) as pool:
        print("Starting multiprocessing")
        results = pool.starmap(process_row, zip(df["lines"].values, df["result"].values))
        print("Finished multiprocessing")

    print("Processing complete")
    all_positions, all_evaluations, all_results = map(np.concatenate, zip(*results))
    return all_positions, all_evaluations, all_results

# Creating 29 features in total for model to learn
def extract_features(position):
    board = chess.Board(position)
    features = []
    
    # Material features (one per piece type per colour) 
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                           chess.ROOK, chess.QUEEN]:
            features.append(len(board.pieces(piece_type, color)))
    
    # Piece-square table contribution (one per piece type per colour)
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                           chess.ROOK, chess.QUEEN, chess.KING]:
            total = 0
            for square in board.pieces(piece_type, color):
                if color == chess.WHITE:
                    total += helpers.PIECE_TABLES[piece_type][square]
                else:
                    total += helpers.PIECE_TABLES[piece_type][chess.square_mirror(square)]
            features.append(total)
    
    # Mobility 
    features.append(len(list(board.legal_moves)))
    
    # Mobility for the other side
    board.push(chess.Move.null())  
    features.append(len(list(board.legal_moves)))
    board.pop()
    
    # King safety
    features.append(int(board.has_castling_rights(chess.WHITE)))
    features.append(int(board.has_castling_rights(chess.BLACK)))
    features.append(int(board.is_check()))
    
    # Centre control
    features.append(helpers.get_board_control(board))
    
    # Whose turn
    features.append(1 if board.turn == chess.WHITE else 0)
    
    return features

def train_evaluations():
    all_positions, all_evaluations, all_results = process_data()

    x = np.array([extract_features(position) for position in all_positions], dtype=np.float32)
    y = np.array(all_evaluations, dtype=np.float32)

    # Normalise inputs
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0) + 1e-8  # avoid divide-by-zero
    x = (x - x_mean) / x_std
    
    # Save the normalisation params so ai.py can use them
    np.save("feature_mean.npy", x_mean)
    np.save("feature_std.npy", x_std)
    
    # Clip y to avoid the NN obsessing over rare mate scores
    y = np.clip(y, -2000, 2000)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Use MAE as a metric (interpretable in centipawns)
    model.compile(optimizer='adam', loss="mean_squared_error", metrics=["mae"])
    
    model.fit(x_train, y_train, epochs=50, batch_size=256, 
              validation_split=0.1, verbose=1)

    mse, mae = model.evaluate(x_test, y_test)
    print(f"Test MSE: {mse:.1f}")
    print(f"Test MAE: {mae:.1f} centipawns")  # ← this is what you care about

    model.save("chess_evaluation_model.keras")  # new format, no h5 warning

    if _engine is not None:
        _engine.quit()
        
if __name__ == "__main__":
    train_evaluations()