import game
import chess
import numpy as np
import helpers
import time
import chess.polyglot

transposition_table = {}

class TimeoutError(Exception):
    pass

def evaluate_board_with_classical(board):
    return helpers.evaluate_board_classical(board)

def quiescence(board, alpha, beta, maximisingPlayer, eval_fn):
    """Search only captures until the position is 'quiet'."""
    stand_pat = eval_fn(board)
    
    if maximisingPlayer:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
            board.push(move)
            try:
                score = quiescence(board, alpha, beta, False, eval_fn)
            finally:
                board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        # for move in board.legal_moves:
        #     if not board.is_capture(move):
        #         continue
        #     board.push(move)
        #     score = quiescence(board, alpha, beta, False, eval_fn)
        #     board.pop()
            
        #     if score >= beta:
        #         return beta
        #     if score > alpha:
        #         alpha = score
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
            try:
                score = quiescence(board, alpha, beta, True, eval_fn)
            finally:
                board.pop()
            if score <= alpha:
                return alpha
            if score < beta:
                beta = score
        return beta

def minimax_with_timeout(board, depth, alpha, beta, maximisingPlayer, start_time, time_limit, eval_fn):
    # Check timer first
    if time.time() - start_time > time_limit:
        raise TimeoutError()
    
    if board.is_game_over():
        return eval_fn(board), None
    if depth == 0:
        return quiescence(board, alpha, beta, maximisingPlayer, eval_fn), None
    
    key = chess.polyglot.zobrist_hash(board)
    if key in transposition_table:
        cached_depth, cached_score, cached_move = transposition_table[key]
        if cached_depth >= depth:
            return cached_score, cached_move
    if maximisingPlayer:
        curr_max = float('-inf')
        best_move = None
        for move in order_moves(board):
            board.push(move)
            try:
                eval, _ = minimax_with_timeout(board, depth - 1, alpha, beta, False, start_time, time_limit, eval_fn)
            finally:
                board.pop()
            if eval > curr_max:
                curr_max = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        transposition_table[key] = (depth, curr_max, best_move)
        return curr_max, best_move
    else:
        curr_min = float('inf')
        best_move = None
        for move in order_moves(board):
            board.push(move)
            try:
                eval, _ = minimax_with_timeout(board, depth - 1, alpha, beta, True, start_time, time_limit, eval_fn)
            finally:
                board.pop()
            if eval < curr_min:
                curr_min = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[key] = (depth, curr_min, best_move)
        return curr_min, best_move
    

    # if maximisingPlayer:
    #     curr_max = float('-inf')
    #     best_move = None
    #     for move in order_moves(board):
    #         board.push(move)
    #         eval, _ = minimax_with_timeout(board, depth - 1, alpha, beta, False, start_time, time_limit, eval_fn)
    #         board.pop()
    #         if eval > curr_max:
    #             curr_max = eval
    #             best_move = move
    #         alpha = max(alpha, eval)
    #         if beta <= alpha:
    #             break
    #     transposition_table[key] = (depth, curr_max, best_move)
    #     return curr_max, best_move
    # else:
    #     curr_min = float('inf')
    #     best_move = None
    #     for move in order_moves(board):
    #         board.push(move)
    #         eval, _ = minimax_with_timeout(board, depth - 1, alpha, beta, True, start_time, time_limit, eval_fn)
    #         board.pop()
    #         if eval < curr_min:
    #             curr_min = eval
    #             best_move = move
    #         beta = min(beta, eval)
    #         if beta <= alpha:
    #             break
    #     transposition_table[key] = (depth, curr_min, best_move)
    #     return curr_min, best_move
    
# Iteratively deepen until time runs out and returns the best move found 
def find_best_move(board, time_limit=5.0, max_depth=20, eval_fn=None):
    if eval_fn is None:
        eval_fn = evaluate_board_with_classical

    transposition_table.clear()

    start_time = time.time()
    best_move = None

    for depth in range(1, max_depth + 1):
        try:
            score, move = minimax_with_timeout(
                board, depth, float('-inf'), float('inf'),
                board.turn == chess.WHITE,
                start_time, time_limit, eval_fn
            )

            if move is not None and move in board.legal_moves:
                best_move = move

            elapsed = time.time() - start_time
            print(f"  depth {depth}: best={move}, score={score}, time={elapsed:.2f}s")
            
            if elapsed > time_limit / 2:
                break
        except TimeoutError:
            print(f"  depth {depth}: timed out, using depth {depth-1} result")
            break
    return best_move


# Returns the best score along with the best move (legacy fixed-depth version)
def minimax(board, depth, alpha, beta, maximisingPlayer):
    if board.is_game_over():
        return evaluate_board_with_classical(board), None
    if depth == 0:
        return quiescence(board, alpha, beta, maximisingPlayer, evaluate_board_with_classical), None
    
    key = chess.polyglot.zobrist_hash(board)
    if key in transposition_table:
        cached_depth, cached_score, cached_move = transposition_table[key]
        if cached_depth >= depth:
            return cached_score, cached_move
    
    if maximisingPlayer:
        curr_max = float('-inf')
        best_move = None

        for move in order_moves(board):
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            if eval > curr_max:
                curr_max = eval
                best_move = move
            
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        
        transposition_table[key] = (depth, curr_max, best_move)
        return curr_max, best_move

    else:
        curr_min = float('inf')
        best_move = None

        for move in order_moves(board):
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()

            if eval < curr_min:
                curr_min = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

        transposition_table[key] = (depth, curr_min, best_move)
        return curr_min, best_move

# Takes board position and returns legal moves sorted by how promising each look
def order_moves(board):
    def move_score(move):
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            
            if victim is None:
                return 100
            victim_value = helpers.PIECE_VALUES[victim.piece_type]
            attacker_value = helpers.PIECE_VALUES[attacker.piece_type]

            return 10 * victim_value - attacker_value
        return 0
    return sorted(board.legal_moves, key=move_score, reverse=True)

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
    playAI()