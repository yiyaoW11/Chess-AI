import chess
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import load_model
import game

# Load once
_model = load_model("chess_evaluation_model.keras")
_mean = np.load("feature_mean.npy")
_std = np.load("feature_std.npy")

@tf.function
def _fast_predict(x):
    return _model(x, training=False)

def evaluate_board_with_nn(board):
    if board.is_checkmate():
        return -9999 if board.turn == chess.WHITE else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    features = np.array(game.extract_features(board.fen()), dtype=np.float32)
    features = (features - _mean) / _std
    features = features.reshape(1, -1)
    
    score = _fast_predict(features).numpy()[0][0]

    
    return float(score)