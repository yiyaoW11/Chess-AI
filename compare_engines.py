import chess
import time

import helpers
import nn_eval
import ai

def play_one_game(white_eval_fn, black_eval_fn, time_per_move=1.0):
    """Play a full game. Each side uses their own evaluation function."""
    board = chess.Board()
    while not board.is_game_over() and board.fullmove_number < 150:
        ai.transposition_table.clear()
        
        # Monkey-patch the eval function based on whose turn it is
        if board.turn == chess.WHITE:
            ai.evaluate_board_with_classical = white_eval_fn
        else:
            ai.evaluate_board_with_classical = black_eval_fn
        
        move = ai.find_best_move(board, time_limit=time_per_move)
        if move is None:
            break
        board.push(move)
    
    result = board.result()
    return result

def tournament(n_games=20, time_per_move=1.0):
    classical = helpers.evaluate_board_classical
    neural = nn_eval.evaluate_board_with_nn
    
    nn_wins = classical_wins = draws = 0
    
    for i in range(n_games):
        # Alternate colours each game for fairness
        if i % 2 == 0:
            white, black = neural, classical
            nn_is_white = True
        else:
            white, black = classical, neural
            nn_is_white = False
        
        print(f"\nGame {i+1}: NN as {'White' if nn_is_white else 'Black'}")
        start = time.time()
        result = play_one_game(white, black, time_per_move)
        print(f"  Result: {result} ({time.time()-start:.0f}s)")
        
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == nn_is_white:
            nn_wins += 1
        else:
            classical_wins += 1
    
    print(f"\n=== Final ===")
    print(f"NN: {nn_wins} wins, Classical: {classical_wins} wins, Draws: {draws}")
    nn_score = nn_wins + 0.5 * draws
    print(f"NN scored {nn_score}/{n_games} ({100*nn_score/n_games:.1f}%)")

if __name__ == "__main__":
    tournament(n_games=10, time_per_move=1.0)