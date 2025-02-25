import chess.engine

engine_path = "stockfish/stockfish.exe"  # Update this!
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(depth=10))
print(info)  # If this prints, Stockfish works fine

engine.quit()