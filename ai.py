import game
import chess
# import tensorflow as tf
# from tensorflow.keras.models import load_model
import numpy as np
import helpers

# evaluation_model = load_model("chess_evaluation_model.h5")

# @tf.function
# def fast_eval(x):
#     return evaluation_model(x, training=False)

def evaluate_board_with_classical(board):
    return helpers.evaluate_board_classical(board)

# def evaluate_board_with_nn(board):
#     features = game.extract_features(board.fen())
#     features = np.array(features, dtype=np.float32).reshape(1, -1)
    
#     # CHANGED: use fast_eval instead of .predict()
#     evaluation = fast_eval(features).numpy()[0][0]
#     return evaluation

def quiescence(board, alpha, beta, maximisingPlayer):
    """Search only captures until the position is 'quiet'."""
    stand_pat = evaluate_board_with_classical(board)
    
    if maximisingPlayer:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
            board.push(move)
            score = quiescence(board, alpha, beta, False)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha
    else:
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat
        
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
            board.push(move)
            score = quiescence(board, alpha, beta, True)
            board.pop()
            
            if score <= alpha:
                return alpha
            if score < beta:
                beta = score
        return beta

# Returns the best score along with the best move
def minimax(board, depth, alpha, beta, maximisingPlayer):
    if board.is_game_over():
        return evaluate_board_with_classical(board), None
    if depth == 0:
        return quiescence(board, alpha, beta, maximisingPlayer), None
    
    # White is maximising player
    if maximisingPlayer:
        curr_max = float('-inf')
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            if eval > curr_max:
                curr_max = eval
                best_move = move
            
            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return curr_max, best_move

    else:
        curr_min = float('inf')
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()

            if eval < curr_min:
                curr_min = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return curr_min, best_move

def playAI():
    board = game.chess.Board()
    print("Please input move in UCI format (e.g. d2d4)")

    while not board.is_game_over():
        print(board)

        if board.turn == chess.WHITE:
            move = input("Your move: ")
            try:
                board.push_uci(move)
            except ValueError:
                print("Invalid move")
                continue
        else:
            print("AI is thinking...")
            best_ai_move = minimax(board, 2, float('-inf'), float('inf'), board.turn == chess.WHITE)[1]
            board.push(best_ai_move)
            print(f"AI played {best_ai_move}")
            
    print("Game Over!")
    print(f"Result: ", board.result())

if __name__ == "__main__":
    # evaluation_model = load_model("chess_eval_model.h5")
    playAI()