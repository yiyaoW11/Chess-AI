import game
import chess
from tensorflow.keras.models import load_model
import numpy as np

def evaluate_board_with_nn(board):
    features = game.extract_features(board.fen())
    features = np.array(features).reshape(1, -1)

    # Predict the evaluation score using the trained model
    evaluation = evaluation_model.predict(features, verbose=0)[0][0]
    return evaluation

# Returns the best score along with the best move
def minimax(board, depth, alpha, beta, maximisingPlayer):
    if depth == 0 or board.is_game_over():
        return evaluate_board_with_nn(board), None
    
    # White is maximising player
    if maximisingPlayer:
        curr_max = float('-inf')
        best_move = None

        for move in list(board.legal_moves):
            board_cpy = board.copy()
            board_cpy.push(move)

            eval, _ = minimax(board_cpy, depth - 1, alpha, beta, False)

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

        for move in list(board.legal_moves):
            board_cpy = board.copy()
            board_cpy.push(move)

            eval, _ = minimax(board_cpy, depth - 1, alpha, beta, True)

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

    while not game.terminal(board):
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
            best_ai_move = minimax(board, 2, float('-inf'), float('inf'), False)[1]
            board.push(best_ai_move)
            print(f"AI played {best_ai_move}")
            
    print("Game Over!")
    print(f"Result: ", board.result())

if __name__ == "__main__":
    evaluation_model = load_model("chess_eval_model.h5")
    playAI()