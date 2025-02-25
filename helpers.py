import chess 

def get_board_control(board):
    num_centre_attackers = 0
    centre_squares = ["d4", "d5", "e4", "e5"]

    for square_str in centre_squares:
        square = chess.parse_square(square_str)
        attackers_white = board.attackers(chess.WHITE, square)
        attackers_black = board.attackers(chess.BLACK, square)
        num_centre_attackers += len(attackers_white) + len(attackers_black)

    return num_centre_attackers

def get_material_balance(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    white_material = sum(piece_values[piece.piece_type] for _, piece in board.piece_map().items() if piece.color == chess.WHITE)
    black_material = sum(piece_values[piece.piece_type] for _, piece in board.piece_map().items() if piece.color == chess.BLACK)

    return white_material - black_material

def get_king_safety(board):
    king_safety_white = 1 if board.has_castling_rights(chess.WHITE) else 0
    king_safety_black = 1 if board.has_castling_rights(chess.BLACK) else 0

    return king_safety_white + king_safety_black