# # Using stockfish to provide evaluation of each position 
# def generate_evaluations(positions):
#     evaluations = []
    
#     for position in positions:
#         board = chess.board(position)
#         analysis = engine.analyse(board, chess.engine.Limit(depth=30))
#         evaluations.append(analysis["score"].relative.score(mate_score=5000))
#     engine.quit()
#     return evaluations

