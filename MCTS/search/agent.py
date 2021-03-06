import numpy as np
import pandas as pd
import random

import MCTS.search.game as game
import MCTS.search.tokens as tokens
import MCTS.search.features as features

# MCTS taken from pseudocode on geeksforgeeks


class Agent:

    def __init__(self, game_, player, past_states, trade_prop):
        self.game = game_
        self.player = player
        self.other = game.other_player(player)

        self.trade_prop = trade_prop
        self.turn = 0
        self.past_states = past_states

        self.away_recently_moved = None
        self.home_recently_moved = None
        self.root = None

        weights = pd.read_csv("genetic_programming/weights.csv", sep=",", header=[0])

        data = weights.sample(axis=0, random_state=random.randint(0, 1000000))

        self.weight_index = data.iloc[0, 0]
        self.weight_score = data.iloc[0, 1]
        self.weight_games = data.iloc[0, 2]
        self.weights = data.iloc[0, 3:].astype(np.float)

    class Node:

        def __init__(self, agent, strategy, player, parent):
            self.move = strategy[0]
            self.data = strategy[1]
            self.val = 0
            self.n = 0

            self.agent = agent
            self.player = player
            self.parent = parent
            self.unseen = None
            self.seen = []

            self.uct = 0

        def update_stats(self, result):
            if self.n == 0:
                self.n = 1
                self.val = result
            else:
                self.val = self.val / self.n + result
                self.n += 1
                self.val = self.val / self.n

            self.update_uct()

        def get_stats(self):
            return self.val, self.n, self.data

        def update_uct(self):
            from math import sqrt, log

            if self.parent is None:
                return

            self.uct = self.val + sqrt(2) * sqrt(log(self.parent.n) / self.n)

        def create_children(self):
            self.unseen = self.agent.get_node_utility(self.data, self.player)

    def monte_carlo(self, game_state, simulations, depth):
        self.root = self.Node(self, (None, game_state), self.player, None)
        self.root.n = 1
        #    from datetime import datetime

        for i in range(simulations):
            #        startTime = datetime.now()
            path = [self.root]
            #        a = datetime.now() - startTime
            #        print(i, 1, a)
            leaf = self.selection(self.root, path)
            #        b = datetime.now() - startTime - a
            #        print(i, 2, b)
            simulation_score = self.rollout(game_state, leaf, depth)
            #        c = datetime.now() - startTime - b
            #        print(i, 3, c)
            self.backpropagate(leaf, simulation_score)
        #        d = datetime.now() - startTime - c
        #        print(i, 4, d)

        return self.best_child(self.root)

    # Function for node traversal, finds the further leaf node unvisited
    def selection(self, node, path):

        val, n, game_state = node.get_stats()

        # While not leaf node
        while n != 0 and not game.end(game_state):
            node, is_leaf = self.best_uct(node, path)

            if is_leaf:
                break

            path.append(node)
            val, n, game_state = node.get_stats()

        return node

    # Function for result of the simulation
    def rollout(self, curr_state, leaf, depth):

        node = leaf
        game_state = node.data

        from datetime import datetime

        while depth != 0 and not game.end(game_state):
            node = self.rollout_policy(node)
            game_state = node.data

            depth -= 1

        evaluation = len(game_state[self.player]) - len(game_state[self.other])

        if evaluation == 0:
            return self.evaluation(curr_state, game_state, self.player)

        return evaluation

    def rollout_policy(self, node):

        game_state = node.data
        player = node.player
        temp_game = game.Game(game_state)

        # Assume opponent is random
        if player != self.player:
            available_moves = tokens.available_moves(player)
            pieces = game_state[player]

            strategy = None
            has_valid_move = False

            while not has_valid_move:
                move = random.choice(available_moves)
                piece = random.choice(pieces)
                xy = piece[1], piece[2]

                if move == "Boom":
                    has_valid_move = True

                    temp_game.boom(xy, player)
                    strategy = [(None, xy, move, None), temp_game.get_game_state()]

                else:
                    distance = random.randint(1, piece[0])
                    amount = random.randint(1, piece[0])

                    if temp_game.is_valid_move(xy, move, distance, player):
                        has_valid_move = True

                        temp_game.move_token(amount, xy, move, distance, player)
                        strategy = [(amount, xy, move, distance), temp_game.get_game_state()]

            temp_node = self.Node(self, strategy, game.other_player(player), None)

        # Stick to the plan
        else:
            available = self.available_states(game_state, player)
            strategy = random.choice(available)
            temp_node = self.Node(self, strategy, game.other_player(player), None)

        return temp_node

    @staticmethod
    def backpropagate(node, result):
        while node is not None:
            node.update_stats(result)
            node = node.parent

    def best_child(self, root):
        from math import sqrt

        game_state = root.data
        n_sim = float("-inf")

        can_boom = False
        best_boom_diff = float("-inf")
        best_strategy = None
        best_boom = None

        home_c = features.count_pieces(game_state[self.player])
        away_c = features.count_pieces(game_state[self.other])

        potential_threat = self.has_potential_threat(self.away_recently_moved, self.player)

        if potential_threat:
            temp_game = game.Game(game_state)
            temp_game.boom(self.away_recently_moved, self.other)
            temp_game_state = temp_game.get_game_state()
            home_t = features.count_pieces(temp_game_state[self.player])
            away_t = features.count_pieces(temp_game_state[self.other])
            potential_diff = home_t - away_t
            run_aways = []
            if (home_c > away_c and potential_diff > 0) or (home_c <= away_c and potential_diff >= 0):
                potential_threat = False

        for child in root.seen:
            strategy = child.move
            next_state = child.data

            if next_state[self.player] in self.past_states:
                continue

            # If won
            if not next_state[self.other] and next_state[self.player]:
                return strategy

            if strategy[2] != "Boom":
                xy = game.dir_to_xy(xy=strategy[1], direction=strategy[2], distance=strategy[3])

                damage = sqrt(features.pieces_per_boom(next_state, self.other))
                if damage == features.count_stacks(next_state):
                    continue

                if self.has_potential_threat(xy, self.other):
                    home_b = features.count_pieces(next_state[self.player])
                    away_b = features.count_pieces(next_state[self.other])

                    temp_game = game.Game(next_state)
                    temp_game.boom(xy, self.player)
                    temp_game_state = temp_game.get_game_state()

                    home_a = features.count_pieces(temp_game_state[self.player])
                    away_a = features.count_pieces(temp_game_state[self.other])

                    # IS dead
                    if home_a == 0:
                        continue
                    # Running into a trap
                    if (home_b - home_a) > (away_b - away_a):
                        continue

                if potential_threat:
                    # Losses
                    temp_game = game.Game(next_state)
                    temp_game.boom(self.away_recently_moved, self.other)
                    temp_game_state = temp_game.get_game_state()

                    home_l = features.count_pieces(temp_game_state[self.player])
                    away_l = features.count_pieces(temp_game_state[self.other])
                    loss = home_l - away_l

                    # Gains
                    temp_game1 = game.Game(next_state)
                    temp_game1.boom(xy, self.player)
                    temp_game_state = temp_game1.get_game_state()

                    home_g1 = features.count_pieces(temp_game_state[self.player])
                    away_g1 = features.count_pieces(temp_game_state[self.other])
                    gain1 = home_g1 - away_g1

                    gain2 = float("-inf")
                    # Leave one behind and boom there
                    if not temp_game1.board.is_cell_empty(strategy[1]):
                        temp_game2 = game.Game(next_state)
                        temp_game2.boom(strategy[1], self.player)
                        temp_game_state = temp_game2.get_game_state()

                        home_g2 = features.count_pieces(temp_game_state[self.player])
                        away_g2 = features.count_pieces(temp_game_state[self.other])
                        gain2 = home_g2 - away_g2

                    gain = max(gain1, gain2)
                    if gain2 > gain1:
                        temp_game = temp_game2
                    else:
                        temp_game = temp_game1

                    # Same Cluster boomed or Moving to trade is not worth it
                    if gain <= loss:
                        continue
                    else:
                        trade_game = temp_game
                        trade_game.boom(self.away_recently_moved, self.other)
                        trade_game_state = trade_game.get_game_state()

                        home_o = features.count_pieces(trade_game_state[self.player])
                        away_o = features.count_pieces(trade_game_state[self.other])
                        outcome = home_o - away_o

                        run_aways.append((outcome, strategy))

                    # Minimise Losses
                    if loss > potential_diff:
                        run_aways.append((loss, strategy))

            # If can trade for more
            if strategy[2] == "Boom":
                home_a = features.count_pieces(next_state[self.player])
                away_a = features.count_pieces(next_state[self.other])
                diff = home_a - away_a

                if home_a == 0:
                    continue

                if (home_c > away_c and diff > 0) or (home_c <= away_c and diff >= 0):
                    if diff > best_boom_diff:
                        best_boom_diff = diff
                        best_boom = strategy
                        can_boom = True
                else:
                    continue

            if child.n > n_sim:
                n_sim = child.n
                best_strategy = strategy

        if potential_threat:
            best_move_diff = float("-inf")

            # Run away
            if len(run_aways) != 0:
                run_aways = sorted(run_aways, reverse=True)
                best_move = run_aways[0][1]
                best_move_diff = run_aways[0][0]

            # Trade first
            if best_boom_diff > potential_diff and can_boom:
                if best_move_diff > best_boom_diff:
                    return best_move
                return best_boom

        if can_boom:
            return best_boom

        # If nothing is good
        if best_strategy is None:
            print("lose")
            # We lose anyways
            moves = self.available_states(game_state, self.player)
            return random.choice(moves)[0]

        return best_strategy

    def best_uct(self, node, path):

        if node.unseen is None:
            node.create_children()

        if len(node.unseen) != 0:
            child_strategy = node.unseen.pop(0)
            child = self.Node(self, child_strategy, game.other_player(node.player), node)
            node.seen.append(child)
            return child, False

        max_uct = float("-inf")
        next_best_node = None

        for child in node.seen:
            if child in path:
                continue
            if child.uct >= max_uct:
                next_best_node = child
                max_uct = child.uct

        if next_best_node is None:
            return node, True

        return next_best_node, False

    def one_enemy_endgame(self, game_state, simulations, search_depth, two_enemy=False):
        # If enemy can draw or we can win
        for piece in game_state[self.player]:
            home_b = features.count_pieces(game_state[self.player])
            temp_game = game.Game(game_state)
            temp_game.boom((piece[1], piece[2]), self.player)
            home_a = features.count_pieces(temp_game.get_game_state()[self.player])
            if not temp_game.get_game_state()[self.player] or (two_enemy and home_b-home_a >= 2):
                strategy = self.monte_carlo(game_state, simulations, search_depth)
                return strategy
            if not temp_game.get_game_state()[self.other]:
                return None, (piece[1], piece[2]), "Boom", None

        enemy = game_state[self.other][0]
        enemy_xy = enemy[1], enemy[2]

        ally = self.closest_npiece(game_state, 1, self.player, enemy_xy)
        ally_xy = ally[1], ally[2]

        # Close enough to boom
        if abs(enemy_xy[0] - ally_xy[0]) <= 1 and abs(enemy_xy[1] - ally_xy[1]) <= 1:
            return None, ally_xy, "Boom", None

        if two_enemy:
            enemy = game_state[self.other][1]
            enemy_xy = enemy[1], enemy[2]

            ally = self.closest_npiece(game_state, 1, self.player, enemy_xy)
            ally_xy = ally[1], ally[2]

            # Close enough to boom
            if abs(enemy_xy[0] - ally_xy[0]) <= 1 and abs(enemy_xy[1] - ally_xy[1]) <= 1:
                return None, ally_xy, "Boom", None

        return self.go_there(1, ally, enemy_xy)

    # Doesn't take into account draws
    def two_enemy_endgame(self, game_state, simulations, search_depth):
        enemy_stacks = len(game_state[self.other])

        for piece in game_state[self.player]:
            temp_game = game.Game(game_state)
            temp_game.boom((piece[1], piece[2]), self.player)
            if not temp_game.get_game_state()[self.player]:
                strategy = self.monte_carlo(game_state, simulations, search_depth)
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
            return self.one_enemy_endgame(game_state, simulations, search_depth, two_enemy=True)

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
                dist = dist/max(piece1[0], piece2[0])

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

    def get_node_utility(self, game_state, player):
        unvisited_children = []

        next_moves = self.available_states(game_state, player)

        for strategy, next_state in next_moves:
            unvisited_children.append((self.utility(game_state, next_state, player), (strategy, next_state)))

        indices = [(x[0], i) for i, x in enumerate(unvisited_children)]

        next_best_child = [unvisited_children[i][1] for x, i in indices]

        return next_best_child

    # Can be replaced with another node utility function
    def utility(self, curr_state, next_state, player):
        return self.evaluation(curr_state, next_state, player)

    def evaluation(self, curr_state, game_state, player):

        home_score, away_score = features.eval_function(self, curr_state, game_state, self.player, self.turn)
        score = home_score - away_score

        if player == self.player:
            return score
        else:
            return -score

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

    # Finds the 'children' of current game state
    def available_states(self, game_state, player):
        available = []
        all_available = []
        other = game.other_player(player)

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
                    if features.count_pieces(game_state[player]) < features.count_pieces(game_state[other]):
                        home_diff = features.count_pieces(game_state[player]) - features.count_pieces(temp_game_state[player])
                        away_diff = features.count_pieces(game_state[other]) - features.count_pieces(temp_game_state[other])

                        # Not worth it to trade for less
                        if home_diff >= away_diff:
                            continue

                    # If suicide for nothing
                    if features.count_pieces(game_state[other]) == features.count_pieces(temp_game_state[other]):
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

    def suicide_move(self, game_, player, xy):
        curr_state = game_.get_game_state()
        home_b, away_b = features.count_all(curr_state, player)

        # Us booming is the same as someone adj booming on their next turn
        game_.boom(xy, player)
        next_state = game_.get_game_state()
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
