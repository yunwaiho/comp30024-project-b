import numpy as np

import polkadots.search.game as game
import polkadots.search.tokens as tokens


class Agent:

    def __init__(self, player):
        self.player = player
        self.other = game.other_player(player)

    def maximiser(self, curr_state, game_state, depth, past_states, alpha, beta, player):

        # Base Case
        if depth == 0 or game.end(game_state):

            #print(score(game_state, self.player))

            evaluation, features = score(curr_state, game_state, self.player)

            return None, evaluation, features

        best_strategy = None
        best_features = np.empty((1, 15))
        best_depth = float("-inf")

        if self.player == player:

            next_states = available_states(game_state, player)

            for strategy, next_state in next_states:

                if next_state[player] in past_states:
                    continue

                next_strategy, val, features = self.maximiser(curr_state, next_state, depth - 1, past_states, alpha, beta, self.other)

                if val >= alpha:
                    alpha = val
                    best_strategy = strategy
                    best_features = features
                if val == alpha and depth > best_depth:
                    best_depth = depth
                    best_strategy = strategy
                    best_features = features

                if alpha > beta:
                    return best_strategy, beta, best_features

            return best_strategy, alpha, best_features

        if self.player != player:

            next_states = available_states(game_state, player)

            for strategy, next_state in next_states:

                if next_state[player] in past_states:
                    continue

                next_strategy, val, features = self.maximiser(curr_state, next_state, depth - 1, past_states, alpha, beta, self.player)

                if val <= beta:
                    beta = val
                    best_strategy = strategy
                    best_features = features
                if val == beta and depth > best_depth:
                    best_depth = depth
                    best_strategy = strategy
                    best_features = features

                if beta < alpha:
                    return best_strategy, alpha, best_features

            return best_strategy, beta, best_features


# How to define the score of the current game state
# Open to change
def score(curr_state, game_state, player):

    other = game.other_player(player)

    # Winning Condition for player
    if not game_state[player]:
        return float("-inf"), np.empty((1, 15))
    else:
        if not game_state[other]:
            return float("inf"), np.empty((1, 15))

    b_home_pieces = curr_state[player]
    b_away_pieces = curr_state[other]

    a_home_pieces = game_state[player]
    a_away_pieces = game_state[other]

    home_num = count_pieces(a_home_pieces)
    away_num = count_pieces(a_away_pieces)
    total_num = home_num + away_num

    home_pieces_diff = count_pieces(b_home_pieces) - home_num
    away_pieces_diff = count_pieces(b_away_pieces) - away_num

    home_stacks = count_stacks(a_home_pieces)
    away_stacks = count_stacks(a_away_pieces)

    home_stack_size = average_stack_size(a_home_pieces)
    away_stack_size = average_stack_size(a_away_pieces)

    home_spread = distance_from_centroid(game_state, player, own_centroid=True)
    away_spread = distance_from_centroid(game_state, other, own_centroid=True)

    home_boom_area = boom_area(a_home_pieces)
    away_boom_area = boom_area(a_away_pieces)

    home_booms_needed, away_clusters, away_cluster_size = booms_required(game_state, player)
    away_booms_needed, home_clusters, home_cluster_size = booms_required(game_state, other)

    home_threat = min_dist_to_boom(game_state, player)
    away_threat = min_dist_to_boom(game_state, other)

    features = np.array([home_num, away_num, total_num,
                         home_pieces_diff, away_pieces_diff,
                         home_stacks, away_stacks, home_stack_size, away_stack_size,
                         home_spread, away_spread,
                         home_boom_area, away_boom_area,
                         home_threat, away_threat])

    weights = np.array([-311.74, -316.35, -629.02, 1.61, -0.14, -273.81, -264.65, -23.26, -21.37, -54.16, -52.31, -516.43, -622.78, -80.16, -75.09])

    final = np.dot(features, weights)

    return final, features


# Finds the 'children' of current game state
def available_states(game_state, player):
    available = []
    other = game.other_player(player)

    for piece in game_state[player]:
        xy = piece[1], piece[2]
        available_moves = tokens.available_moves()

        for move in available_moves:
            if move == "Boom":
                temp_game = game.Game(game_state)
                if not temp_game.has_surrounding(xy):
                    continue

                temp_game.boom(xy, player)
                temp_game_state = temp_game.get_game_state()

                # If current number of home pieces <= current number of away pieces
                if count_pieces(game_state[player]) < count_pieces(game_state[other]):
                    home_diff = count_pieces(game_state[player]) - count_pieces(temp_game_state[player])
                    away_diff = count_pieces(game_state[other]) - count_pieces(temp_game_state[other])

                    if home_diff >= away_diff:
                        continue

                # If suicide for nothing
                if count_pieces(game_state[other]) == count_pieces(temp_game_state[other]):
                    # Don't
                    continue

                available.append([(None, xy, move, None), temp_game.get_game_state()])
            else:
                for distance in range(piece[0]):
                    distance = piece[0] - distance

                    for amount in range(piece[0]):
                        amount = piece[0] - amount

                        temp_game = game.Game(game_state)
                        if temp_game.is_valid_move(xy, move, distance, player):
                            temp_game.move_token(amount, xy, move, distance, player)
                            available.append([(amount, xy, move, distance), temp_game.get_game_state()])

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
    return count_stacks(pieces)/count_pieces(pieces)


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
        return 0, None

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
    from math import pow, sqrt, ceil

    minimum = float("inf")

    for piece1 in game_state[player]:
        x1, y1 = piece1[1], piece1[2]

        for piece2 in game_state[game.other_player(player)]:
            x2, y2 = piece2[1], piece2[2]

            dist = sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2))

            minimum = min(minimum, ceil(dist/piece1[0]))

    return minimum


# Returns average distance of a colours piece to each boom centroid, weighted by the size of cluster
def avg_dist_to_boom(game_state, player, boom_centroids, boom_weights):
    from math import pow, sqrt

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