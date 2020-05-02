import numpy as np
import pandas as pd
import random

import MCTS.search.game as game
import MCTS.search.tokens as tokens

# MCTS taken from pseudocode on geeksforgeeks
class Agent:

    def __init__(self, game_, player):
        self.game = game_
        self.player = player
        self.other = game.other_player(player)

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

        for i in range(simulations):
            path = [self.root]
            leaf = self.selection(self.root, path)
            simulation_score = self.rollout(game_state, leaf, depth)
            self.backpropagate(leaf, simulation_score)

        return self.best_child(self.root)

    # Function for node traversal, finds the further leaf node unvisited
    def selection(self, node, path):

        val, n, game_state = node.get_stats()

        # While not leaf node
        while n != 0 and not game.end(game_state):
            node = self.best_uct(node, path)
            path.append(node)
            val, n, game_state = node.get_stats()

        return node

    # Function for result of the simulation
    def rollout(self, curr_state, leaf, depth):

        node = leaf
        game_state = node.data

        while depth != 0 and not game.end(game_state):
            node = self.rollout_policy(node)
            game_state = node.data

            depth -= 1

        return self.evaluation(curr_state, game_state, self.player)

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
            available = available_states(game_state, player)
            strategy = random.choice(available)
            temp_node = self.Node(self, strategy, game.other_player(player), None)

        return temp_node

    @staticmethod
    def backpropagate(node, result):
        while node is not None:
            node.update_stats(result)
            node = node.parent

    def best_child(self, root):

        uct_sim = float("-inf")
        best_strategy = None

        for child in root.seen:
            strategy = child.move
            next_state = child.data

            # If won
            if not next_state[self.other] and next_state[self.player]:
                return strategy

            if child.uct > uct_sim:
                uct_sim = child.uct
                best_strategy = strategy

        return best_strategy

    def best_uct(self, node, path):

        if node.unseen is None:
            node.create_children()

        if len(node.unseen) != 0:
            child_strategy = node.unseen.pop(0)
            child = self.Node(self, child_strategy, game.other_player(node.player), node)
            node.seen.append(child)
            return child

        max_uct = float("-inf")
        next_best_node = None

        for child in node.seen:
            if child in path:
                continue
            if child.uct >= max_uct:
                next_best_node = child
                max_uct = child.uct

        return next_best_node

    def get_node_utility(self, game_state, player):
        unvisited_children = []

        next_moves = available_states(game_state, player)

        for strategy, next_state in next_moves:
            unvisited_children.append((self.utility(game_state, next_state, player), (strategy, next_state)))

        indices = [(x[0], i) for i, x in enumerate(unvisited_children)]

        next_best_child = [unvisited_children[i][1] for x, i in indices]

        return next_best_child

    # Can be replaced with another node utility function
    def utility(self, curr_state, next_state, player):
        return self.evaluation(curr_state, next_state, player, utility=True)

    # How to define the score of the current game state
    # Open to change
    def score(self, curr_state, game_state, player, utility):

        other = game.other_player(player)

        if utility:
            if not game_state[player]:
                return float("-inf")
            if not game_state[other]:
                return float("inf")

        tokens.board_configs()

        b_home_pieces = curr_state[player]
        b_away_pieces = curr_state[other]

        a_home_pieces = game_state[player]
        a_away_pieces = game_state[other]

        home_num = count_pieces(a_home_pieces)
        away_num = count_pieces(a_away_pieces)
        total_num = home_num + away_num

        if total_num == 0:
            return 0

        home_pieces_diff = count_pieces(b_home_pieces) - home_num
        away_pieces_diff = count_pieces(b_away_pieces) - away_num

        if home_pieces_diff == 0:
            diff_ratio = 0
        else:
            diff_ratio = away_pieces_diff/home_pieces_diff

        home_stacks = count_stacks(a_home_pieces)

        home_stack_size = average_stack_size(a_home_pieces)

        home_threat = min_dist_to_boom(game_state, player)

        if away_num == 0:
            max_damage = 0
        else:
            max_damage = pieces_per_boom(game_state, player)/away_num
        if home_num == 0:
            max_losses = 0
        else:
            max_losses = pieces_per_boom(game_state, other)/home_num

        home_board_score = self.get_board_score(game_state, player)
        away_board_score = self.get_board_score(game_state, other)

        features = np.array([
            home_num,
            away_num,
            away_pieces_diff,
            home_pieces_diff,
            diff_ratio,
            home_stacks,
            home_stack_size,
            home_num/total_num*home_threat,
            max_damage,
            max_losses,
            home_board_score,
            away_board_score,
        ])

        final = np.dot(features, self.weights)

        return final

    def evaluation(self, curr_state, game_state, player, utility=False):

        home_score = self.score(curr_state, game_state, self.player, utility)
        away_score = self.score(curr_state, game_state, self.other, utility)

        score = home_score - away_score

        if player == self.player:
            return score
        else:
            return -score

    def update_weights(self, game_state):

        # Win
        if game_state[self.player] and not game_state[self.other]:
            if self.player == "black":
                weight_score = 2
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
            df.iloc[self.weight_index, i+1] = lst[i]

        df.to_csv("genetic_programming/weights.csv", index=False)

    def get_board_score(self, game_state, player):

        other_scores = 0
        total_scores = 0

        for piece in game_state[player]:
            xy = piece[1], piece[2]

            other, total = self.count_adjacent(player, xy)

            other_n = other.count("1")
            total_n = total.count("1")

            other_score = self.find_pattern(other, other_n)
            total_score = self.find_pattern(total, total_n)

            other_scores += other_score
            total_scores += total_score

        if total_scores == 0:
            return 0
        else:
            return other_scores/total_scores

    def count_adjacent(self, player, xy):
        board = self.game.board
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
            return 8*8
        if string_n > 4:
            string = string.replace("1", "a")
            string = string.replace("0", "1")
            string = string.replace("a", "0")

        config = board_config[str(string_n)]

        n = 0
        value = 0

        for key, val in config.items():
            n += val
            if key in 2*string or key in 2*string[::-1]:
                value = val

        return string_n*string_n*(1 - value/n)


# Finds the 'children' of current game state
def available_states(game_state, player):
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
                if count_pieces(game_state[player]) < count_pieces(game_state[other]):
                    home_diff = count_pieces(game_state[player]) - count_pieces(temp_game_state[player])
                    away_diff = count_pieces(game_state[other]) - count_pieces(temp_game_state[other])

                    # Not worth it to trade for less
                    if home_diff >= away_diff:
                        continue

                # If suicide for nothing
                if count_pieces(game_state[other]) == count_pieces(temp_game_state[other]):
                    # Don't
                    continue

                available.append([(None, xy, move, None), temp_game.get_game_state()])
            else:
                if piece[0] != 1:
                    amount = min(piece[0], 8)

                    # Move whole stack or leave one
                    amounts = [amount, amount-1]

                    for n in amounts:
                        if n == 1:
                            continue

                        for distance in range(piece[0]):
                            distance = piece[0] - distance

                            temp_game = game.Game(game_state)
                            if temp_game.is_valid_move(xy, move, distance, player):
                                temp_game.move_token(n, xy, move, distance, player)
                                available.append([(n, xy, move, distance), temp_game.get_game_state()])

                # Move only one
                distance = piece[0]
                amount = piece[0]
                temp_game = game.Game(game_state)
                if temp_game.is_valid_move(xy, move, distance, player):
                    temp_game.move_token(amount, xy, move, distance, player)
                    available.append([(amount, xy, move, distance), temp_game.get_game_state()])

    if len(available) == 0:
        return all_available

    return available


# Counts the number of pieces
def count_pieces(pieces):
    n = 0
    for piece in pieces:
        n += piece[0]
    return n


# Counts the number of stacks
def count_stacks(pieces):
    return len(pieces)


# Counts the average stack size
def average_stack_size(pieces):

    if count_pieces(pieces) == 0:
        return 0

    score = 0

    for piece in pieces:
        score += piece[0]^2

    return score/count_stacks(pieces)


# Returns average center of mass of all pieces player colour
def centre_of_mass(game_state, player):
    x, y, n = 0, 0, 0

    for piece in game_state[player]:
        x += piece[1]
        y += piece[2]
        n += 1

    return x / n, y / n


# Returns average distance of a colours pieces from a colour's centre of mass
def distance_from_centroid(game_state, player, own_centroid):
    from math import pow, sqrt

    if not (game_state[player] and game_state[game.other_player(player)]):
        return 0

    if own_centroid:
        x_c, y_c = centre_of_mass(game_state, player)
    else:
        x_c, y_c = centre_of_mass(game_state, game.other_player(player))
    dist, n = 0, 0

    for piece in game_state[player]:
        dist += sqrt(pow(piece[2] - y_c, 2) + pow(piece[1] - x_c, 2))
        n += 1

    if dist == 0:
        return 1

    return dist / n


# Finds the number of booms needed using single-linkage clustering
# Inspired by Krushals Algorithm
def booms_required(game_state, player):
    """

    :param game_state: current state of the game
    :param player: home player colour
    :return: n: (len(sets)) the number of booms required to kill all enemies currently
             xy: (centroids) list of size n, showing the centroids of where to boom to remove all enemies
             cluster: (weights) list of size n, the number enemy pieces in each cluster
    """
    from math import pow, sqrt

    def find_set(p, s):
        xy = (p[1], p[2])

        for key in s:
            if xy in s[key]:
                return key

    def union(p1, p2, s):
        i = find_set(p1, s)
        j = find_set(p2, s)

        s[i] = s[i].union(s[j])
        s.pop(j, None)

    sets = {}

    # Make-set
    pieces = game_state[game.other_player(player)]

    if not pieces:
        return 0, [], []

    i = 0
    for piece in pieces:
        xy = (piece[1], piece[2])

        sets[i] = {xy}
        i += 1

    weights = []

    # Get weights
    for piece1 in pieces:
        for piece2 in pieces:
            weights.append((sqrt(pow(piece1[2] - piece2[2], 2) + pow(piece1[1] - piece2[1], 2)), (piece1, piece2)))

    weights = sorted(weights)
    weight = weights.pop(0)

    while weights and weight[0] < sqrt(8):
        if weight[0] != 0:
            piece1 = weight[1][0]
            piece2 = weight[1][1]
            if find_set(piece1, sets) != find_set(piece2, sets):
                union(piece1, piece2, sets)

        weight = weights.pop(0)

    centroids = []
    weights = []
    for key in sets:
        cluster = sets[key]
        weights.append(len(cluster))
        x, y = 0, 0
        for piece in cluster:
            x += piece[0]
            y += piece[1]
        centroids.append((x / len(cluster), y / len(cluster)))

    return len(sets), centroids, weights


# Finds the minimum moves to move to a boom location
def min_dist_to_boom(game_state, player):
    from math import ceil

    if not (game_state[player] and game_state[game.other_player(player)]):
        return 0

    minimum = float("inf")

    for piece1 in game_state[player]:
        x1, y1 = piece1[1], piece1[2]

        for piece2 in game_state[game.other_player(player)]:
            x2, y2 = piece2[1], piece2[2]

            dist = y2 - y1 + x2 - x1

            minimum = min(minimum, ceil(dist/piece1[0]))

    return minimum


# Returns average distance of a colours piece to each boom centroid, weighted by the size of cluster
def avg_dist_to_boom(game_state, player, boom_centroids, boom_weights):
    from math import pow, sqrt

    if not (game_state[player] and game_state[game.other_player(player)]):
        return 0

    dist, n = 0, 0

    for piece in game_state[player]:
        for i in range(len(boom_centroids)):
            x, y = boom_centroids[i]
            weight = boom_weights[i]
            dist += weight * sqrt(pow(piece[2] - y, 2) + pow(piece[1] - x, 2))
            n += weight

    return dist / n


# Returns the number of spots that are covered by booms
def boom_area(pieces):

    locs = []

    for piece in pieces:
        x, y = piece[1], piece[2]

        for i in range(x - 1, x + 1 + 1):
            for j in range(y - 1, y + 1 + 1):
                ij = (i, j)

                if tokens.out_of_board(ij):
                    continue

                if ij not in locs:
                    locs.append(ij)

    return len(locs)


def pieces_per_boom(game_state, player):

    other = game.other_player(player)

    damages = []
    away_before = len(game_state[other])

    for piece in game_state[player]:
        temp_game = game.Game(game_state)
        xy = (piece[1], piece[2])

        temp_game.boom(xy, player)
        temp_game_state = temp_game.get_game_state()

        away_after = len(temp_game_state[other])

        damage = away_before - away_after

        damages.append(damage)

    if len(damages) == 0:
        return 0

    return max(damages)


