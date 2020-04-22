"""
Module for Tokens!
"""


def available_moves():
    return "Boom", "Up", "Down", "Left", "Right"


def out_of_board(xy):
    x, y = xy
    if x < 0 or y < 0 or x > 7 or y > 7:
        return True
    return False


def is_valid_move(board, xy1, xy2):
    x1, y1 = xy1
    x2, y2 = xy2

    if out_of_board(xy2):
        return False

    # Needs to move
    if x1 == x2 and y1 == y2:
        return False

    # Can't move diagonally
    if x1 != x2 and y1 != y2:
        return False

    # Can't move more than stack
    size = board.get_size(xy1)
    if abs(x1 - x2) > size or abs(y1 - y2) > size:
        return False

    if board.is_cell_empty(xy2):
        return True
    else:
        if board.get_colour(xy1) != board.get_colour(xy2):
            return False

    return True
