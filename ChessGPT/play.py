import chess
import random
from .interface import Interface


def play(interface: Interface, color = "w"):
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)

    if color == "b":
        move = interface.input()
        board.push_san(move)

    while True:
        all_moves = list(board.legal_moves)
        best_move = random.choice(all_moves)
        interface.output(board.san(best_move))
        board.push(best_move)

        move = interface.input()
        board.push_san(move)
        # print(board)
