import json

import MCTS_op.search.game as game
import MCTS_op.search.tokens as tokens


class OpenBook:
    other_player_move = None
    is_early_game = True

    def __init__(self, game_, colour):

        self.player = colour
        self.other = game.other_player(colour)

        self.early_game_turn = 0
        self.game = game_

        if colour == "black":
            self.book = black_opening_book()

        if colour == "white":
            self.book = white_opening_book()

        self.boom_book = get_boom_book()

    def next_move(self):
        dict_move = self.early_game_turn + 1

        if dict_move in self.book:
            if self.player == "white" and self.early_game_turn == 0:
                return self.book[dict_move][0]

            for move in self.book[dict_move][OpenBook.other_player_move]:
                if self.is_comp_valid_move(move):
                    if dict_move == self.boom_book[self.player][OpenBook.other_player_move]:
                        # if self.good_boom_check(move):
                        return move
                    else:
                        return move

        OpenBook.is_early_game = False
        return None

    def is_comp_valid_move(self, move):

        action_type = move[0]

        if action_type == "BOOM":
            xy1 = move[1]
            piece = self.game.board.get_token(xy1)
            if piece[0] == self.player and not self.game.board.is_cell_empty(xy1):
                return True
        else:
            n = move[1]
            xy1 = move[2]
            xy2 = move[3]
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
                OpenBook.other_player_move = OpenBook.other_player_move_side(action)

    @staticmethod
    def other_player_move_side(action):
        action_type = action[0]

        if action_type == "BOOM":
            return "L"
        else:
            n = action[1]
            xy1 = action[2]

            x, y = xy1
            if x < 2:
                return "L"
            if 2 < x < 5:
                return "C"
            if x > 5:
                return "R"

    # def good_boom_check(self,move):


def check_early_game():
    return OpenBook.is_early_game


def white_opening_book():
    book = {
        1: [("MOVE", 1, (0, 1), (1, 1))],
        2: {
            "L": [("MOVE", 1, (1, 0), (1, 1))],
            "C": [("MOVE", 1, (4, 1), (3, 1))],
            "R": [("MOVE", 1, (7, 1), (6, 1))]
        },
        3: {
            "L": [("MOVE", 3, (1, 1), (4, 1))],
            "C": [("MOVE", 2, (3, 1), (1, 1))],
            "R": [("MOVE", 2, (6, 1), (4, 1))]
        },
        4: {
            "L": [("MOVE", 1, (4, 0), (4, 1))],
            "C": [("MOVE", 1, (1, 1), (1, 5))],
            "R": [("MOVE", 3, (4, 1), (2, 1))]
        },
        5: {
            "L": [("MOVE", 1, (4, 1), (4, 6)), ("MOVE", 5, (4, 1), (5, 1))],
            "C": [("BOOM", (1, 5))],
            "R": [("MOVE", 2, (1, 1), (2, 1))]
        },
        6: {
            "L": [("MOVE", 1, (4, 6), (5, 6)), ("MOVE", 1, (5, 1), (5, 6))],
            "C": [("MOVE", 1, (3, 0), (4, 0))],
            "R": [("MOVE", 1, (2, 1), (2, 6)), ("MOVE", 1, (2, 1), (2, 5))]
        },
        7: {
            "L": [("BOOM", (5, 6))],
            "C": [("MOVE", 2, (4, 0), (6, 0))],
            "R": [("BOOM", (2, 6)), ("BOOM", (2, 5))]
        }
    }

    return book


def black_opening_book():
    book = {
        1: {
            "L": [("MOVE", 1, (0, 6), (1, 6))],
            "C": [("MOVE", 1, (3, 6), (4, 6))],
            "R": [("MOVE", 1, (7, 6), (6, 6))]
        },
        2: {
            "L": [("MOVE", 1, (1, 7), (1, 6))],
            "C": [("MOVE", 1, (4, 7), (4, 6))],
            "R": [("MOVE", 1, (6, 7), (6, 6))]
        },
        3: {
            "L": [("MOVE", 3, (1, 6), (4, 6))],
            "C": [("MOVE", 1, (6, 7), (6, 6))],
            "R": [("MOVE", 3, (6, 6), (3, 6))]
        },
        4: {
            "L": [("MOVE", 1, (4, 7), (4, 6))],
            "C": [("MOVE", 3, (4, 6), (5, 6))],
            "R": [("MOVE", 1, (3, 7), (3, 6))]
        },
        5: {
            "L": [("MOVE", 5, (4, 6), (5, 6))],
            "C": [("MOVE", 2, (6, 6), (5, 6))],
            "R": [("MOVE", 5, (3, 6), (2, 6))]
        },
        6: {
            "L": [("MOVE", 1, (5, 6), (5, 1))],
            "C": [("MOVE", 1, (5, 6), (5, 1))],
            "R": [("MOVE", 1, (2, 6), (2, 1))]
        },
        7: {
            "L": [("BOOM", (5, 1))],
            "C": [("BOOM", (5, 1))],
            "R": [("BOOM", (2, 1))]
        }

    }
    return book


def get_boom_book():
    book = {
        "white":
            {
                "L": 7,
                "C": 5,
                "R": 7
            },
        "black":
            {
                "L": 7,
                "C": 7,
                "R": 7
            }
    }

    return book
