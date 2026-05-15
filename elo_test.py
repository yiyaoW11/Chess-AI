import chess
import chess.engine
import ai  # your engine

STOCKFISH = "stockfish/stockfish.exe"

def play_game(stockfish_elo, your_engine_plays_white=True):
    """Returns 'win', 'loss', or 'draw' from your engine's perspective."""
    board = chess.Board()
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    sf.configure({"Skill Level": 0})

    while not board.is_game_over():
        is_your_turn = (board.turn == chess.WHITE) == your_engine_plays_white
        if is_your_turn:
            move = ai.minimax(board, 4, float('-inf'), float('inf'),
                              board.turn == chess.WHITE)[1]
        else:
            move = sf.play(board, chess.engine.Limit(time=0.1)).move
        board.push(move)

    sf.quit()
    result = board.result()  # "1-0", "0-1", or "1/2-1/2"
    if result == "1/2-1/2":
        return "draw"
    your_won = (result == "1-0") == your_engine_plays_white
    return "win" if your_won else "loss"

def test_at_elo(elo, n_games=6):
    wins = draws = losses = 0
    for i in range(n_games):
        # alternate colours so it's fair
        result = play_game(elo, your_engine_plays_white=(i % 2 == 0))
        if result == "win": wins += 1
        elif result == "draw": draws += 1
        else: losses += 1
        print(f"  Game {i+1}: {result}")
    score = wins + 0.5 * draws
    print(f"vs Stockfish @ {elo}: {wins}W / {draws}D / {losses}L ({score}/{n_games})")
    return score / n_games

if __name__ == "__main__":
    # for elo in [1320, 1500, 1700, 1900]:
    print(f"\n--- Testing against Stockfish Elo 0 ---")
    test_at_elo(0, n_games=6)