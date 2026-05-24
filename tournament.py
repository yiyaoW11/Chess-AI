import chess
import time
import csv

import helpers
import nn_eval
import ai

def play_one_game(white_eval_fn, black_eval_fn, time_per_move):
    board = chess.Board()
    while not board.is_game_over() and board.fullmove_number < 150:
        ai.transposition_table.clear()
        eval_fn = white_eval_fn if board.turn == chess.WHITE else black_eval_fn
        
        legal_moves = list(board.legal_moves)
        move = ai.find_best_move(board, time_limit=time_per_move, eval_fn=eval_fn)
        
        # Hard check: is the move actually legal RIGHT NOW?
        if move is None or move not in legal_moves:
            print(f"  Engine returned bad move {move}, picking first legal move instead")
            if not legal_moves:
                break
            move = legal_moves[0]
        
        board.push(move)
    
    return board.result()

def run_match(n_games, time_per_move):
    classical = helpers.evaluate_board_classical
    neural = nn_eval.evaluate_board_with_nn
    
    nn_wins = classical_wins = draws = 0
    
    for i in range(n_games):
        if i % 2 == 0:
            white, black = neural, classical
            nn_is_white = True
        else:
            white, black = classical, neural
            nn_is_white = False
        
        result = play_one_game(white, black, time_per_move)
        
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == nn_is_white:
            nn_wins += 1
        else:
            classical_wins += 1
        
        print(f"  Game {i+1}/{n_games}: {result}  (NN: {nn_wins}W / Classical: {classical_wins}W / Draws: {draws})")
    
    return nn_wins, classical_wins, draws

def main():
    time_controls = [0.5, 1.0, 2.0]
    n_games_per_tc = 10
    
    results = []
    
    for tc in time_controls:
        print(f"\n=== Time control: {tc}s per move ===")
        start = time.time()
        nn_w, cl_w, dr = run_match(n_games_per_tc, tc)
        duration = time.time() - start
        
        nn_score = nn_w + 0.5 * dr
        nn_pct = 100 * nn_score / n_games_per_tc
        
        print(f"\nResult at {tc}s: NN {nn_w}W / Classical {cl_w}W / Draws {dr}")
        print(f"NN score: {nn_score}/{n_games_per_tc} ({nn_pct:.1f}%)")
        print(f"Duration: {duration/60:.1f} minutes")
        
        results.append({
            "time_per_move": tc,
            "n_games": n_games_per_tc,
            "nn_wins": nn_w,
            "classical_wins": cl_w,
            "draws": dr,
            "nn_score_pct": nn_pct,
            "duration_min": duration / 60,
        })
        
        # Save after each time control in case we crash partway
        with open("tournament_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print("\n=== Summary ===")
    print(f"{'Time/move':<12} {'NN%':<8} {'NN':<5} {'Cl':<5} {'Dr':<5}")
    for r in results:
        print(f"{r['time_per_move']:<12} {r['nn_score_pct']:<8.1f} {r['nn_wins']:<5} {r['classical_wins']:<5} {r['draws']:<5}")

if __name__ == "__main__":
    main()