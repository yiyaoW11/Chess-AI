# def evaluate_board_with_nn(board):
#     evaluation_model = load_model("chess_evaluation_model.h5")
#       # Extract features from the current board position
#     features = extract_features(board.fen())
#     features = np.array(features).reshape(1, -1)  # Reshape for model input

#     # Predict the evaluation score using the trained model
#     evaluation = evaluation_model.predict(features)[0][0]
#     return evaluation

#     # piece_values = {
#     #     chess.PAWN: 1,
#     #     chess.KNIGHT: 3,
#     #     chess.BISHOP: 3,
#     #     chess.ROOK: 5,
#     #     chess.QUEEN: 9,
#     #     chess.KING: 100
#     # }

#     # # White Pieces have positive values; Black Pieces have negative values
#     # white_score = sum(piece_values[piece.piece_type] for _, piece in board.piece_map().items() if piece.color == chess.WHITE)
#     # black_score = sum(piece_values[piece.piece_type] for _, piece in board.piece_map().items() if piece.color == chess.BLACK)

#     # return white_score - black_score