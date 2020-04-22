"""
This module processes information regarding the game state and keeps
track of the board
"""

import polkadots.search.util as util
import polkadots.search.tokens as tokens


class Board:

    # Constructor
    def __init__(self):
        self.game_state = {}

        # Create empty board
        coords = [(x, 7 - y) for y in range(8) for x in range(8)]
        for coord in coords:
            self.game_state[coord] = tuple()

    def fill_board(self, colour, token_list):
        for piece in token_list:
            size = piece[0]
            x = piece[1]
            y = piece[2]
            self.place_token(colour, size, (x, y))

    # Places stack of n tokens of colour c into x, y
    def place_token(self, colour, size, xy):
        self.game_state[xy] = (colour, size)

    def remove_token(self, n, xy):
        if n != self.get_size(xy):
            self.game_state[xy] = (self.get_colour(xy), self.get_size(xy) - n)
        else:
            self.game_state[xy] = tuple()

    # Checks if cell is empty
    def is_cell_empty(self, xy):
        if self.game_state[xy]:
            return False
        return True

    # Get tokens attributes
    def get_token(self, xy):
        return self.game_state[xy]

    def get_colour(self, xy):
        return self.get_token(xy)[0]

    def get_size(self, xy):
        return self.get_token(xy)[1]

    # Prints board
    def print_board(self):
        util.print_board(self.game_state)

    # Updates the board after move
    def move_token(self, n, xy1, xy2):
        if not tokens.is_valid_move(self, xy1, xy2):
            raise Exception("This is not a valid move.")
        else:
            new_size = n
            if not self.is_cell_empty(xy2):
                new_size = new_size + self.get_size(xy2)

            self.place_token(self.get_colour(xy1),
                             new_size,
                             xy2)
            self.remove_token(n, xy1)

    # Updates the board after boom
    def boom_token(self, xy):
        self.remove_token(self.get_size(xy), xy)

        x, y = xy
        for i in range(x - 1, x + 1 + 1):
            for j in range(y - 1, y + 1 + 1):
                ij = (i, j)

                if tokens.out_of_board(ij):
                    continue
                if not self.is_cell_empty(ij):
                    self.boom_token(ij)

    # Returns game_state in original dictionary format
    def dict_game_state(self):
        state = {"white": [], "black": []}

        # Create empty board
        coords = [(x, 7 - y) for y in range(8) for x in range(8)]
        for coord in coords:
            if self.game_state[coord]:
                state[self.get_colour(coord)].append([self.get_size(coord),
                                                      coord[0],
                                                      coord[1]])
        return state

    def has_surrounding(self, xy):
        x, y = xy

        for i in range(x-1, x+1+1):
            for j in range(y-1, y+1+1):
                ij = (i, j)

                if tokens.out_of_board(ij) or ij == xy:
                    continue
                if not self.is_cell_empty(ij):
                    return True
        return False
