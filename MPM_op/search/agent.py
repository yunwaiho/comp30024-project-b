import MPM_op.search.features as features
import MPM_op.search.game as game
import MPM_op.search.tokens as tokens
import random
import pandas as pd
import numpy as np


class Agent:

    def __init__(self, game_, game_state, player):

        self.game = game_
        self.game_state = game_state
        self.player = player
        self.other = game.other_player(player)

        self.turn = 0

        self.tt = {}
        self.away_recently_moved = None
        self.home_recently_moved = None

        self.past_states = []
        self.root = None

        weights = pd.read_csv("genetic_programming/weights.csv", sep=",", header=[0])

        data = weights.sample(axis=0, random_state=random.randint(0, 1000000))

        self.weight_index = data.iloc[0, 0]
        self.weight_score = data.iloc[0, 1]
        self.weight_games = data.iloc[0, 2]
        self.weights = data.iloc[0, 3:].astype(np.float)

    class Node:

        def __init__(self, data, player, children):
            self.previous_move = data[0]
            self.game_state = data[1]
            self.player = player
            self.children = children

    def update_root(self, game_state):
        if self.root is not None:
            for node in self.root.children:
                if node.game_state == game_state:
                    self.root = node
                    return
        self.root = None

    def mp_mix(self, paranoid_threshold, max_depth):
        """
        Paranoid Algorithm: Same as regular minimax
        Assumes: The away player is trying to minimise home player's score
        Goal: Maximise the home player's score
        Used when the home player is in the lead

        Offensive Algorithm:
        Assumes: The away player is trying to maximise their own score
        Goal: Maximise the difference between home score and away score
        Used when: The home player is losing
        """

        self.game_state = self.game.get_game_state()
        self.past_states.append(self.game_state[self.player])

        home_eval, away_eval = features.eval_function(self, self.game_state, self.game_state, self.player, self.turn)
        score_diff = home_eval - away_eval

        # Protect our babies
        if score_diff > paranoid_threshold:
            decision = "p"
        # Kill their babies
        else:
            decision = "o"

        alpha = float("-inf")
        beta = float("inf")

        if self.root is None:
            self.root = self.make_node(self.game_state, self.game_state, self.player, decision)

        strategy = self.strategy(node=self.root, curr_state=self.game_state, depth=max_depth,
                                 alpha=alpha, beta=beta, player=self.player, decision=decision)
        return strategy

    def strategy(self, node, curr_state, depth, alpha, beta, player, decision):
        """
        Regular minimax function, with the difference being strategy on how the score is calculated

        Paranoid strategy wants to maximise home player's score
        Offensive strategy wants to maximise the difference between the player's score
        """

        game_state = node.game_state

        # Base Case
        if depth == 0 or game.end(game_state):
            evaluation = self.score(curr_state, game_state, decision, player, alpha, beta, extend=True)

            return None, evaluation

        if node.children is None:
            node.children = self.get_children(curr_state, game_state, player, decision)

        best_strategy = None
        next_states = node.children
        better_children = []

        if self.player == player:
            for child_node in next_states:
                strategy, next_state = child_node.previous_move, child_node.game_state

                if next_state[self.player] in self.past_states:
                    continue

                next_strategy, val = self.strategy(child_node, curr_state, depth - 1, alpha, beta, self.other, decision)

                better_children.append((val, child_node))

                if val >= alpha:
                    alpha = val
                    best_strategy = strategy
                if alpha > beta:
                    break

        if self.player != player:
            for child_node in next_states:
                strategy, next_state = child_node.previous_move, child_node.game_state

                next_strategy, val = self.strategy(child_node, curr_state, depth - 1, alpha, beta, self.player,
                                                   decision)

                better_children.append((val, child_node))

                if val <= beta:
                    beta = val
                    best_strategy = strategy
                if beta < alpha:
                    break

        node.children = self.reorder_nodes(better_children)

        if self.player == player:
            if alpha > beta:
                return best_strategy, beta
            return best_strategy, alpha
        else:
            if beta < alpha:
                return best_strategy, alpha
            return best_strategy, beta

    def make_node(self, curr_state, game_state, player, strategy):

        node_children = self.get_children(curr_state, game_state, player, strategy)
        node = self.Node((None, game_state), player, node_children)
        return node

    def get_children(self, curr_state, game_state, player, strategy):
        children = []
        next_states = self.available_states(game_state, player)

        for next_strategy, next_state in next_states:
            state_score = self.utility(curr_state, next_state, strategy, player)
            children.append((state_score, (next_strategy, next_state)))

        ordered_children = self.reorder_nodes(children)
        children = [self.Node(x, game.other_player(player), None) for x in ordered_children]
        return children

    @staticmethod
    def reorder_nodes(nodes):
        indices = [(x[0], i) for i, x in enumerate(nodes)]
        indices = sorted(indices, reverse=True)
        reordered = [nodes[i][1] for x, i in indices]
        return reordered

    def one_enemy_endgame(self, threshold, max_depth, two_enemy=False):

        game_state = self.game.get_game_state()

        # If enemy can draw or we can win
        for piece in game_state[self.player]:
            home_b = features.count_pieces(game_state[self.player])
            temp_game = game.Game(game_state)
            temp_game.boom((piece[1], piece[2]), self.player)
            home_a = features.count_pieces(temp_game.get_game_state()[self.player])
            if not temp_game.get_game_state()[self.player] or (two_enemy and home_b - home_a >= 2):
                strategy, val = self.mp_mix(threshold, max_depth)
                return strategy
            if not temp_game.get_game_state()[self.other]:
                return None, (piece[1], piece[2]), "Boom", None

        enemy = game_state[self.other][0]
        enemy_xy = enemy[1], enemy[2]

        ally = self.closest_npiece(game_state, 1, self.player, enemy_xy)
        ally_xy = ally[1], ally[2]

        # Close enough to boom
        if abs(enemy_xy[1] - ally_xy[0]) <= 1 and abs(enemy_xy[2] - ally_xy[1]) <= 1:
            return None, ally_xy, "Boom", None

        return self.go_there(1, ally, enemy_xy)

    # Doesn't take into account draws
    def two_enemy_endgame(self, threshold, max_depth):
        game_state = self.game_state
        enemy_stacks = len(game_state[self.other])

        for piece in game_state[self.player]:
            temp_game = game.Game(game_state)
            temp_game.boom((piece[1], piece[2]), self.player)
            if not temp_game.get_game_state()[self.player]:
                strategy, val = self.mp_mix(threshold, max_depth)
                return strategy
            if not temp_game.get_game_state()[self.other]:
                return None, (piece[1], piece[2]), "Boom", None

        # One stack
        if enemy_stacks == 1:
            enemy = game_state[self.other][0]
            enemy_xy = enemy[1], enemy[2]

            ally = self.closest_npiece(game_state, 2, self.player, enemy_xy)
            if ally is None:
                return self.make_stack(game_state)

            ally_xy = ally[1], ally[2]
            enemy_corner_xy = self.get_nearest_corner(ally_xy, enemy_xy)

            return self.go_there(2, ally, enemy_corner_xy)

        # Two seperate stacks
        else:
            return self.one_enemy_endgame(threshold, max_depth, two_enemy=True)

    def get_nearest_corner(self, ally_xy, enemy_xy):
        from math import sqrt, pow

        closest = None
        min_dist = float("inf")
        horizontal = [-1, 1]
        vertical = [-1, 1]

        for i in horizontal:
            for j in vertical:
                ij = enemy_xy[0] + i, enemy_xy[1] + j

                if tokens.out_of_board(ij):
                    continue
                dist = sqrt(pow(ally_xy[0] - ij[0], 2) + pow(ally_xy[1] - ij[1], 2))

                if dist < min_dist:
                    min_dist = dist
                    closest = ij
        return closest

    def make_stack(self, game_state):

        min_dist = float("inf")
        pieces = None

        for piece1 in game_state[self.player]:
            for piece2 in game_state[self.player]:
                if piece1 == piece2:
                    continue

                dist = piece1[1] - piece2[1] + piece1[2] - piece2[2]
                dist = dist / max(piece1[0], piece2[0])

                if dist < min_dist:
                    min_dist = dist
                    pieces = piece1, piece2

        if pieces[0][0] > pieces[1][0]:
            return self.go_there(pieces[0][0], pieces[0], pieces[1])
        else:
            return self.go_there(pieces[1][0], pieces[1], pieces[0])

    def go_there(self, n, piece, get_to):
        piece_xy = piece[1], piece[2]
        width = get_to[0] - piece[1]
        height = get_to[1] - piece[2]

        # Move vertically
        if abs(height) > abs(width):
            if piece[0] >= abs(height):
                if height > 0:
                    return n, piece_xy, "Up", abs(height)
                else:
                    return n, piece_xy, "Down", abs(height)
            else:
                if height > 0:
                    return piece[0], piece_xy, "Up", piece[0]
                else:
                    return piece[0], piece_xy, "Down", piece[0]
        # Move horizontally
        else:
            if piece[0] >= abs(width):
                if width > 0:
                    return n, piece_xy, "Right", abs(width)
                else:
                    return n, piece_xy, "Left", abs(width)
            else:
                if width > 0:
                    return piece[0], piece_xy, "Right", piece[0]
                else:
                    return piece[0], piece_xy, "Left", piece[0]

    def score(self, curr_state, game_state, decision, player, alpha=None, beta=None, extend=False):

        game_state_str = get_str(game_state)

        if game_state_str in self.tt:
            home_eval, away_eval = self.tt[game_state_str]
        else:
            if not game_state[player]:
                return float("-inf")
            elif not game_state[game.other_player(player)]:
                return float("inf")

            # Not a quiet node
            if extend and features.pieces_threatened(game_state, self.player) > 0:
                return self.quiesce(curr_state, game_state, alpha, beta, self.player)

            home_eval, away_eval = features.eval_function(self, curr_state, game_state, self.player, self.turn)

            self.tt[game_state_str] = (home_eval, away_eval)

        if decision == "p":
            return home_eval
        else:
            return home_eval - away_eval

    def quiesce(self, curr_state, game_state, alpha, beta, player):
        print(alpha, beta)
        home_eval, away_eval = features.eval_function(self, curr_state, game_state, self.player, self.turn)
        stand_pat = (home_eval - away_eval)

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        next_moves = self.available_states(game_state, player)
        for move, next_state in next_moves:
            score = -self.quiesce(curr_state, next_state, -beta, -alpha, game.other_player(player))
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def available_states(self, game_state, player, get_all=False):
        available = []
        all_rational_available = []
        all_available = []

        home_b, away_b = features.count_all(game_state, player)

        for piece in game_state[player]:
            xy = piece[1], piece[2]
            available_moves = tokens.available_moves(player)

            for move in available_moves:
                if move == "Boom":
                    temp_game = game.Game(game_state)

                    if not temp_game.has_surrounding(xy):
                        continue

                    temp_game.boom(xy, player)

                    all_available.append([(None, xy, move, None), temp_game.get_game_state()])

                    home_a, away_a = features.count_all(temp_game.get_game_state(), player)

                    # If suicide for nothing
                    if away_b == away_a:
                        # Don't
                        continue

                    all_rational_available.append([(None, xy, move, None), temp_game.get_game_state()])

                    if self.is_bad_boom(home_b, home_a, away_b, away_a):
                        continue

                    available.append([(None, xy, move, None), temp_game.get_game_state()])

                else:
                    # Not optimal to move in between unless you have really good strategies
                    if piece[0] == 1:
                        amounts = [1]
                    elif piece[0] == 2:
                        amounts = [1, 2]
                    else:
                        amount = min(piece[0], 8)
                        # Move whole stack or leave one or move one
                        amounts = [1, amount, amount - 1]

                    for n in range(piece[0]):
                        n = piece[0] - n

                        for distance in range(piece[0]):
                            distance = piece[0] - distance
                            temp_game = game.Game(game_state)
                            if temp_game.is_valid_move(xy, move, distance, player):
                                temp_game.move_token(n, xy, move, distance, player)
                                xy2 = game.dir_to_xy(xy, move, distance)

                                all_available.append([(n, xy, move, distance), temp_game.get_game_state()])

                                # Moving into a trap
                                if self.suicide_move(temp_game, player, xy2):
                                    continue

                                all_rational_available.append([(n, xy, move, distance), temp_game.get_game_state()])

                                if self.player == player and n not in amounts:
                                    continue

                                # We don't like a v pattern (inefficient move)
                                if self.creates_v(temp_game, xy2):
                                    continue

                                if temp_game.get_game_state()[self.player] in self.past_states:
                                    continue

                                available.append([(n, xy, move, distance), temp_game.get_game_state()])

        if player != self.player or len(available) == 0 or get_all:
            if len(all_rational_available) == 0:
                return all_available
            else:
                return all_rational_available

        return available

    # Can be replaced with another node utility function
    def utility(self, curr_state, next_state, strategy, player):

        utility = self.score(curr_state, next_state, strategy, player)

        return utility

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

    def suicide_move(self, game_, player, xy):
        curr_state = game_.get_game_state()
        temp_game = game.Game(curr_state)
        home_b, away_b = features.count_all(curr_state, player)

        # Us booming is the same as someone adj booming on their next turn
        temp_game.boom(xy, player)
        next_state = temp_game.get_game_state()
        home_a, away_a = features.count_all(next_state, player)

        if self.is_bad_boom(home_b, home_a, away_b, away_a):
            return True

        return False

    # Subject to change
    @staticmethod
    def is_bad_boom(home_b, home_a, away_b, away_a):
        diff_b = home_b - away_b
        diff_a = home_a - away_a

        # If less or equal pieces and the difference between pieces increase
        if home_b <= away_b and diff_b < diff_a:
            return True
        # If more pieces, don't accept a boom that will reduce our lead
        if home_b > away_b and diff_b <= diff_a:
            return True

        return False

    @staticmethod
    def closest_npiece(game_state, n, player, xy):
        from math import sqrt, pow

        x1, y1 = xy
        max_dist = float("inf")
        closest_ally = None

        for piece in game_state[player]:
            if piece[0] < n:
                continue

            x2, y2 = piece[1], piece[2]

            dist = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

            if dist < max_dist:
                max_dist = dist
                closest_ally = piece

        return closest_ally

    def get_board_score(self, game_state, player):

        other_scores = 0
        total_scores = 0

        for piece in game_state[player]:
            xy = piece[1], piece[2]
            position_score = 12 - piece[0]

            other, total = self.count_adjacent(player, xy)

            other_n = other.count("1")
            total_n = total.count("1")

            other_score = self.find_pattern(other, other_n) * position_score
            total_score = self.find_pattern(total, total_n) * position_score

            other_scores += other_score
            total_scores += total_score

        if total_scores == 0:
            return 0
        else:
            return other_scores / total_scores

    @staticmethod
    def find_pattern(string, string_n):
        board_config = tokens.board_configs()

        if string_n == 0:
            return 0
        if string_n == 8:
            return 8 * 8
        if string_n > 4:
            string = string.replace("1", "a")
            string = string.replace("0", "1")
            string = string.replace("a", "0")

        config = board_config[str(string_n)]

        n = 0
        value = 0

        for key, val in config.items():
            n += val
            if key in 2 * string or key in 2 * string[::-1]:
                value = val

        return string_n * string_n * (1 - value / n)

    def update_weights(self, game_state):

        # Win
        if game_state[self.player] and not game_state[self.other]:
            if self.player == "black":
                weight_score = 1
            else:
                weight_score = 1
        # Lose
        elif game_state[self.other] and not game_state[self.player]:
            weight_score = 0
        else:
            weight_score = 0.25

        total_score = self.weight_score + weight_score
        games_played = self.weight_games + 1

        lst = [total_score, games_played] + list(self.weights)

        df = pd.read_csv("genetic_programming/weights.csv", sep=",", header=[0])

        for i in range(len(lst)):
            df.iloc[self.weight_index, i + 1] = lst[i]

        df.to_csv("genetic_programming/weights.csv", index=False)


def get_str(game_state):
    import json

    return json.dumps(game_state)
