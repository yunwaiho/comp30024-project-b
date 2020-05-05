import numpy as np

import MPM.search.features as features
import MPM.search.game as game
import MPM.search.tokens as tokens


class Agent:

    def __init__(self, game_, player, past_states):

        self.game = game_
        self.player = player
        self.other = game.other_player(player)

        self.turn = 0
        self.past_states = past_states

        self.away_recently_moved = None
        self.home_recently_moved = None
        self.root = None

    '''
    Paranoid Algorithm:
    Assumes: The away player is trying to minimise home player's score
    Goal: Maximise the home player's score
    Used when the home player is in the lead
    '''

    def paranoid(self, curr_state, game_state, depth, past_states, alpha, beta, player):

        # Base Case
        if depth == 0 or game.end(game_state):
            # print(score(game_state, self.player))

            evaluation = self.score(curr_state, game_state, self.player, alg_type="paranoid")

            return None, evaluation

        best_strategy = None
        best_depth = float("-inf")

        if self.player == player:

            next_states = self.available_states(game_state, player)

            for strategy, next_state in next_states:

                if next_state[player] in past_states:
                    continue

                next_strategy, val = self.paranoid(curr_state, next_state, depth - 1, past_states, alpha,
                                                   beta, self.other)

                if val >= alpha:
                    alpha = val
                    best_strategy = strategy
                #if val == alpha and depth > best_depth:
                    #best_depth = depth
                    #best_strategy = strategy

                if alpha > beta:
                    return best_strategy, beta

            return best_strategy, alpha

        if self.player != player:

            next_states = self.available_states(game_state, player)

            for strategy, next_state in next_states:

                if next_state[player] in past_states:
                    continue

                next_strategy, val = self.paranoid(curr_state, next_state, depth - 1, past_states, alpha,
                                                   beta, self.player)
                if val <= beta:
                    beta = val
                    best_strategy = strategy

                #if val == beta and depth > best_depth:
                    #best_depth = depth
                    #best_strategy = strategy

                if beta < alpha:
                    return best_strategy, alpha

            return best_strategy, beta

    '''
    Directed Offensive Algorithm:
    Assumes:The away player is trying to maximise their own score
    Goal: Minimises the away player's score as long as does not compromise home's current score
    Used when: The home player is losing
    '''
############################### NOT FINISHED #################################################
    def directed_offensive(self, curr_state, game_state, depth, past_states, alpha, beta, player, home_threshold):

        # Base Case
        if depth == 0 or game.end(game_state):
            # print(score(game_state, self.player))

            home_eval, away_eval = self.score(curr_state, game_state, self.player, alg_type="directed")

            return None, [home_eval, away_eval]

        best_strategy = None
        best_depth = float("-inf")
        #max_other_val = float("-inf")

        if self.player == player:

            next_states = self.available_states(game_state, player)

            for strategy, next_state in next_states:

                if next_state[player] in past_states:
                    continue

                next_strategy, eval = self.directed_offensive(curr_state, next_state, depth - 1,
                                                              past_states, alpha,
                                                              beta, self.other, home_threshold)
                if eval[1] <= alpha:
                    best_eval = eval
                    alpha = eval[1]
                    best_strategy = strategy

                if alpha < beta:
                    return best_strategy, beta

            return best_strategy, alpha

        if self.player != player:

            next_states = self.available_states(game_state, player)

            for strategy, next_state in next_states:

                if next_state[player] in past_states:
                    continue

                next_strategy, eval = self.directed_offensive(curr_state, next_state, depth - 1, past_states, alpha,
                                                              beta, self.player)
                if eval[1] >= beta:
                    best_eval = eval
                    beta = eval[1]
                    best_strategy = strategy

                if beta < alpha:
                    return best_strategy, alpha

            return best_strategy, best_eval

    def score(self, curr_state, game_state, player, alg_type):

        home_eval = features.eval_function(curr_state, game_state, self.player)
        away_eval = features.eval_function(curr_state, game_state, self.other)

        if alg_type == "paranoid":
            return home_eval - away_eval
        else:
            return home_eval, away_eval

    def available_states(self, game_state, player):
        available = []
        all_available = []
        # other = game.other_player(player)

        for piece in game_state[player]:
            xy = piece[1], piece[2]
            available_moves = tokens.available_moves(player)

            for move in available_moves:
                if move == "Boom":
                    temp_game = game.Game(game_state)

                    if not temp_game.has_surrounding(xy):
                        continue

                    temp_game.boom(xy, player)
                    temp_game_state = temp_game.get_game_state()

                    all_available.append([(None, xy, move, None), temp_game.get_game_state()])

                    # If current number of home pieces <= current number of away pieces
                    if features.count_pieces(game_state[player]) < features.count_pieces(game_state[self.other]):
                        home_diff = features.count_pieces(game_state[player]) - features.count_pieces(
                            temp_game_state[player])
                        away_diff = features.count_pieces(game_state[self.other]) - features.count_pieces(
                            temp_game_state[self.other])

                        # Not worth it to trade for less
                        if home_diff >= away_diff:
                            continue

                    # If suicide for nothing
                    if features.count_pieces(game_state[self.other]) == features.count_pieces(temp_game_state[
                                                                                                  self.other]):
                        # Don't
                        continue

                    available.append([(None, xy, move, None), temp_game.get_game_state()])
                else:
                    if piece[0] == 1:
                        amounts = [1]
                    elif piece[0] == 2:
                        amounts = [1, 2]
                    else:
                        amount = min(piece[0], 8)
                        # Move whole stack or leave one or move one
                        amounts = [1, amount, amount - 1]

                    for n in amounts:
                        for distance in range(piece[0]):
                            distance = piece[0] - distance
                            temp_game = game.Game(game_state)
                            if temp_game.is_valid_move(xy, move, distance, player):
                                temp_game.move_token(n, xy, move, distance, player)
                                xy2 = game.dir_to_xy(xy, move, distance)

                                # We don't like a v pattern (inefficient move)
                                if self.creates_v(temp_game, xy2):
                                    continue

                                available.append([(n, xy, move, distance), temp_game.get_game_state()])

        if player != self.player or len(available) == 0:
            return all_available

        return available

    def count_adjacent(self, player, xy, game_=None):
        if game_ is None:
            board = self.game.board
        else:
            board = game_.board
        x, y = xy
        other = ""
        total = ""

        lr_index = [0, 1, 1, 1, 0, -1, -1, -1]
        ud_index = [1, 1, 0, -1, -1, -1, 0, 1]

        has_enemy = False

        for i in range(8):
            ij = x + lr_index[i], y + ud_index[i]

            if tokens.out_of_board(ij) or board.is_cell_empty(ij):
                other += "0"
                total += "0"
                continue

            total += "1"
            if board.get_colour(ij) == player:
                other += "0"
            else:
                other += "1"
                has_enemy = True

        if not has_enemy:
            return other, other

        return other, total

    def creates_v(self, game_, xy):
        ally_pieces, all_pieces = self.count_adjacent(self.other, xy, game_=game_)

        checked = False

        i = 1
        # Checks for a v, means two bits in a row in bit string
        while i < len(2 * ally_pieces):
            if (2 * ally_pieces)[i] == "1":
                if checked:
                    return True
                else:
                    checked = True
            else:
                checked = False

            i += 2

        return False

    def has_potential_threat(self, xy, player):

        if not xy:
            return False
        else:
            x, y = xy

        for i in range(x - 1, x + 1 + 1):
            for j in range(y - 1, y + 1 + 1):
                ij = i, j

                if tokens.out_of_board(ij) or ij == xy:
                    continue
                if not self.game.board.is_cell_empty(ij) and self.game.board.get_colour(ij) == player:
                    return True

        return False
