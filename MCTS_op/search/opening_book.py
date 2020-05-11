# Module for opening moves

import numpy as np

import MCTS_op.search.game as game
import MCTS_op.search.tokens as tokens


class OpenBook:

    # Constructor
    def __init__(self, game_, colour):

        self.player = colour
        self.other = game.other_player(colour)

        self.early_game_turn = 0
        self.game = game_

        self.other_player_move = None
        self.is_early_game = True

        if colour == "black":
            self.book = black_opening_book()

        if colour == "white":
            self.book = white_opening_book()

        self.boom_book = get_boom_book()

    # Decides the next move
    def next_move(self):
        dict_move = self.early_game_turn + 1
        boom_diff = []
        boom_move = []

        if dict_move in self.book:

            if self.player == "white" and self.early_game_turn == 0:
                return self.book[dict_move][0]

            avail_moves = self.book[dict_move][self.other_player_move]
            boom_check_move = self.boom_book[self.player][self.other_player_move]

            #print("dict_move:", dict_move, avail_moves, self.other_player_move)
            if avail_moves is None:
                self.is_early_game = False
                return None

            for move in avail_moves:
                if self.is_comp_valid_move(move):
                    if dict_move in boom_check_move:
                        boom_move.append(move)
                        boom_diff.append(self.good_boom_check(move))
                    else:
                        return move

            if boom_diff:
                print("boom_diff", boom_diff)
                boom_diff_np = np.array(boom_diff)
                index = np.argmax(boom_diff_np)
                return boom_move[index]

        self.is_early_game = False
        return None

    # Checks for a valid move
    def is_comp_valid_move(self, move):

        action_type = move[0]

        if action_type == "BOOM":
            xy1 = move[1]
            piece = self.game.board.get_token(xy1)
            if not self.game.board.is_cell_empty(xy1):
                if piece[0] == self.player:
                    return True
        else:
            n = move[1]
            xy1 = move[2]
            xy2 = move[3]

            if not self.game.board.is_cell_empty(xy1):
                piece = self.game.board.get_token(xy1)
                if tokens.is_valid_move(self.game.board, xy1, xy2):
                    if piece[0] == self.player and piece[1] >= n:
                        return True
        return False

    def update_early_game(self, colour, action):
        if colour == self.player:
            self.early_game_turn += 1
        else:
            if self.early_game_turn == 1 and self.player == "white" or self.early_game_turn == 0 and self.player == "black":
                self.other_player_move = self.other_player_move_side(action)

    # Checks which side of the board did the player move
    @staticmethod
    def other_player_move_side(action):
        action_type = action[0]

        if action_type == "BOOM":
            xy1 = action[1]
            x, y = xy1
            if x < 2:
                return "CL"
            if x == 3:
                return "CL"
            if x == 4:
                return "CL"
            if x > 5:
                return "CR"
        else:
            xy1 = action[2]
            xy2 = action[3]
            x1, y1 = xy1
            x2, y2 = xy2
            if x1 < 2:
                return "L"
            if x1 == 3:
                if x2 <= 3:
                    return "CL"
                else:
                    return "CR"
            if x1 == 4:
                if x2 >= 4:
                    return "CR"
                else:
                    return "CL"
            if x1 > 5:
                return "R"

    # Checks if the move results in a good boom
    def good_boom_check(self, move):
        move_type = move[0]

        if move_type == "BOOM":
            xy = move[1]
        else:
            xy = move[3]

        home_boom = []
        away_boom = []
        visited = []

        self.count_boom(xy, visited, home_boom, away_boom)

        num_home_boom = len(home_boom)
        num_away_boom = len(away_boom)

        return num_away_boom - num_home_boom

    def count_boom(self, xy, visited, home_boom, away_boom):
        x, y = xy

        for i in range(x - 1, x + 1 + 1):
            for j in range(y - 1, y + 1 + 1):
                ij = (i, j)
                if ij not in visited:
                    if tokens.out_of_board(ij):
                        continue
                    if not self.game.board.is_cell_empty(ij):
                        visited.append(ij)
                        if self.game.board.get_colour(ij) == self.player:
                            home_boom.append(ij)
                            self.count_boom(ij, visited, home_boom, away_boom)
                        else:
                            away_boom.append(ij)
                            self.count_boom(ij, visited, home_boom, away_boom)

    def check_early_game(self):
        return self.is_early_game


#######################################################################################
# Dictionaries for opening moves

def get_boom_book():
    book = {
        "white":
            {
                "L": [6],
                "CL": [5],
                "CR": [],
                "R": [6]
            },
        "black":
            {
                "L": [5, 6],
                "CR": [5],
                "CL": [5],
                "R": [5, 6]
            }
    }
    return book


def white_opening_book():
    book = {
        1: [("MOVE", 1, (4, 0), (4, 1))],
        2: {
            "L": [("MOVE", 1, (0, 1), (1, 1))],
            "CL": [("MOVE", 2, (4, 1), (4, 3))],
            "CR": [("MOVE", 2, (4, 1), (2, 1))],
            "R": [("MOVE", 1, (7, 1), (6, 1))]
        },
        3: {
            "L": [("MOVE", 2, (1, 1), (3, 1))],
            "CL": [("MOVE", 2, (4, 3), (5, 3))],
            "CR": [("MOVE", 1, (3, 1), (2, 1))],
            "R": [("MOVE", 2, (6, 1), (4, 1))]
        },
        4: {
            "L": [("MOVE", 3, (3, 1), (5, 1))],
            "CL": [("MOVE", 1, (5, 3), (5, 5))],
            "CR": [("MOVE", 3, (2, 1), (2, 4)), ("MOVE", 1, (1, 1), (2, 1))],
            "R": [("MOVE", 4, (4, 1), (2, 1))]
        },
        5: {
            "L": [("MOVE", 2, (4, 1), (5, 1))],
            "CL": [("BOOM", (5, 5)), ("MOVE", 1, (5, 5), (5, 6))],
            "CR": [("MOVE", 1, (2, 4), (2, 7)), ("MOVE", 1, (2, 1), (2, 5))],
            "R": [("MOVE", 1, (3, 1), (2, 1))]
        },
        6: {
            "L": [("MOVE", 1, (5, 1), (5, 6)), ("MOVE", 1, (5, 1), (5, 5)), ("MOVE", 1, (5, 1), (5, 4))],
            "CL": [("BOOM", (5, 6)), ("MOVE", 1, (3, 1), (2, 1))],
            "CR": [("BOOM", (2, 7)), ("BOOM", (2, 5))],
            "R": [("MOVE", 1, (2, 1), (2, 6)), ("MOVE", 1, (2, 1), (2, 5)), ("MOVE", 1, (2, 1), (2, 4))]
        },
        7: {
            "L": [("BOOM", (5, 6)), ("BOOM", (5, 5)), ("BOOM", (5, 4))],
            "CL": [("MOVE", 1, (3, 1), (2, 1))],
            "CR": None,
            "R": [("BOOM", (2, 6)), ("BOOM", (2, 5)), ("BOOM", (2, 4))]
        }
    }

    return book


def black_opening_book():
    book = {
        1: {
            "L": [("MOVE", 1, (0, 6), (1, 6))],
            "CL": [("MOVE", 1, (3, 6), (4, 6))],
            "CR": [("MOVE", 1, (4, 6), (3, 6))],
            "R": [("MOVE", 1, (7, 6), (6, 6))]
        },
        2: {
            "L": [("MOVE", 2, (1, 6), (3, 6))],
            "CL": [("MOVE", 2, (4, 6), (4, 4))],
            "CR": [("MOVE", 2, (3, 6), (3, 4))],
            "R": [("MOVE", 2, (6, 6), (4, 6))]
        },
        3: {
            "L": [("MOVE", 3, (3, 6), (6, 6))],
            "CL": [("MOVE", 2, (4, 4), (5, 4))],
            "CR": [("MOVE", 2, (3, 4), (2, 4))],
            "R": [("MOVE", 3, (4, 6), (1, 6))]
        },
        4: {
            "L": [("MOVE", 1, (6, 6), (6, 2))],
            "CL": [("MOVE", 1, (5, 4), (5, 2))],
            "CR": [("MOVE", 1, (2, 4), (2, 2))],
            "R": [("MOVE", 1, (1, 6), (1, 2))]
        },
        5: {
            "L": [("MOVE", 1, (6, 2), (6, 1)), ("BOOM", (6, 2)), ("MOVE", 1, (6, 2), (5, 2))],
            "CL": [("BOOM", (5, 2)), ("MOVE", 1, (5, 2), (5, 1))],
            "CR": [("BOOM", (2, 2)), ("MOVE", 1, (2, 2), (2, 1))],
            "R": [("MOVE", 1, (1, 2), (1, 1)), ("BOOM", (1, 2)), ("MOVE", 1, (1, 2), (2, 2))]
        },
        6: {
            "L": [("BOOM", (6, 1)), ("MOVE", 1, (6, 1), (5, 1)), ("BOOM", (5, 2)), ("MOVE", 1, (5, 2), (5, 1))],
            "CL": [("BOOM", (5, 1))],
            "CR": [("BOOM", (2, 1))],
            "R": [("BOOM", (1, 1)), ("MOVE", 1, (1, 1), (2, 1)), ("BOOM", (2, 2)), ("MOVE", 1, (2, 2), (2, 1))]
        },
        7: {
            "L": [("BOOM", (5, 1))],
            "CL": None,
            "CR": None,
            "R": [("BOOM", (2, 1))]
        }
    }
    return book

'''
# Old book
def get_boom_book():
    book = {
        "white":
            {
                "L": [7],
                "CR": [5],
                "CL": [5],
                "R": [7]
            },
        "black":
            {
                "L": [7],
                "CR": [7],
                "CL": [5],
                "R": [7]
            }
    }
    return book

def white_opening_book():
    book = {
        1: [("MOVE", 1, (0, 1), (1, 1))],
        2: {
            "L": [("MOVE", 1, (1, 0), (1, 1))],
            "CL": [("MOVE", 1, (4, 1), (3, 1))],
            "CR": [("MOVE", 1, (4, 1), (3, 1))],
            "R": [("MOVE", 1, (7, 1), (6, 1))]
        },
        3: {
            "L": [("MOVE", 3, (1, 1), (4, 1))],
            "CL": [("MOVE", 2, (3, 1), (1, 1))],
            "CR": [("MOVE", 2, (3, 1), (1, 1))],
            "R": [("MOVE", 2, (6, 1), (4, 1))]
        },
        4: {
            "L": [("MOVE", 1, (4, 0), (4, 1))],
            "CL": [("MOVE", 1, (1, 1), (1, 5))],
            "CR": [("MOVE", 1, (1, 1), (1, 5))],
            "R": [("MOVE", 3, (4, 1), (2, 1))]
        },
        5: {
            "L": [("MOVE", 1, (4, 1), (4, 6)), ("MOVE", 5, (4, 1), (5, 1))],
            "CL": [("BOOM", (1, 5))],
            "CR": [("BOOM", (1, 5))],
            "R": [("MOVE", 2, (1, 1), (2, 1))]
        },
        6: {
            "L": [("MOVE", 1, (4, 6), (5, 6)), ("MOVE", 1, (5, 1), (5, 6))],
            "CL": [("MOVE", 1, (3, 0), (4, 0))],
            "CR": [("MOVE", 1, (3, 0), (4, 0))],
            "R": [("MOVE", 1, (2, 1), (2, 6)), ("MOVE", 1, (2, 1), (2, 5))]
        },
        7: {
            "L": [("BOOM", (5, 6))],
            "CL": [("MOVE", 2, (4, 0), (6, 0))],
            "CR": [("MOVE", 2, (4, 0), (6, 0))],
            "R": [("BOOM", (2, 6)), ("BOOM", (2, 5))]
        }
    }

    return book




def black_opening_book():
    book = {
        1: {
            "L": [("MOVE", 1, (0, 6), (1, 6))],
            "CR": [("MOVE", 1, (3, 6), (4, 6))],
            "CL": [("MOVE", 1, (3, 6), (4, 6))],
            "R": [("MOVE", 1, (7, 6), (6, 6))]
        },
        2: {
            "L": [("MOVE", 1, (1, 7), (1, 6))],
            "CL": [("MOVE", 1, (4, 7), (4, 6))],
            "CR": [("MOVE", 1, (4, 7), (4, 6))],
            "R": [("MOVE", 1, (6, 7), (6, 6))]
        },
        3: {
            "L": [("MOVE", 3, (1, 6), (4, 6))],
            "CL": [("MOVE", 1, (6, 7), (6, 6))],
            "CR": [("MOVE", 1, (6, 7), (6, 6))],
            "R": [("MOVE", 3, (6, 6), (3, 6))]
        },
        4: {
            "L": [("MOVE", 1, (4, 7), (4, 6))],
            "CL": [("MOVE", 3, (4, 6), (5, 6))],
            "CR": [("MOVE", 3, (4, 6), (5, 6))],
            "R": [("MOVE", 1, (3, 7), (3, 6))]
        },
        5: {
            "L": [("MOVE", 5, (4, 6), (5, 6))],
            "CL": [("MOVE", 2, (6, 6), (5, 6))],
            "CR": [("MOVE", 2, (6, 6), (5, 6))],
            "R": [("MOVE", 5, (3, 6), (2, 6))]
        },
        6: {
            "L": [("MOVE", 1, (5, 6), (5, 1))],
            "CL": [("MOVE", 1, (5, 6), (5, 1))],
            "CR": [("MOVE", 1, (5, 6), (5, 1))],
            "R": [("MOVE", 1, (2, 6), (2, 1))]
        },
        7: {
            "L": [("BOOM", (5, 1))],
            "CL": [("BOOM", (5, 1))],
            "CR": [("BOOM", (5, 1))],
            "R": [("BOOM", (2, 1))]
        }

    }
    return book
'''

