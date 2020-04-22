"""
Module to run the game
"""

from polkadots.search.board import Board
import polkadots.search.tokens as tokens


class Game:

    # Creates the game
    def __init__(self, data):
        self.board = Board()
        self.fill_board(data)

    # Fills the board with tokens
    def fill_board(self, data):
        for colour in ["white", "black"]:
            self.board.fill_board(colour, data[colour])

    def print_board(self):
        self.board.print_board()

    def is_valid_move(self, xy, direction, distance, colour):

        if colour != self.board.get_colour(xy):
            return False

        xy2 = dir_to_xy(xy, direction, distance)

        return tokens.is_valid_move(self.board, xy, xy2)

    def move_token(self, n, xy, direction, distance, colour):

        if colour != self.board.get_colour(xy):
            raise Exception("Cannot move opposition colour")

        xy2 = dir_to_xy(xy, direction, distance)

        self.board.move_token(n, xy, xy2)

    def boom(self, xy, colour):
        if self.board.is_cell_empty(xy) or self.board.get_colour(xy) != colour:
            raise Exception("Cell empty/cannot boom")
        else:
            self.board.boom_token(xy)

    def get_game_state(self):
        return self.board.dict_game_state()

    def end(self):
        if self.win() or self.lose():
            return True

    def win(self):
        if not self.get_game_state()["black"]:
            return True
        return False

    def lose(self):
        if not self.get_game_state()["white"]:
            return True
        return False

    def has_surrounding(self, xy):
        return self.board.has_surrounding(xy)


def dir_to_xy(xy, direction, distance):
    x, y = xy

    if direction == "Up":
        xy2 = (x, y + distance)
    elif direction == "Down":
        xy2 = (x, y - distance)
    elif direction == "Left":
        xy2 = (x - distance, y)
    elif direction == "Right":
        xy2 = (x + distance, y)
    else:
        raise Exception("Direction is incorrect")

    return xy2


def other_player(player):
    if player == "white":
        return "black"
    return "white"
