import polkadots.search.game as game
import polkadots.search.tokens as tokens


def maximiser(game_state, past_states, depth, player):

    # Base Case
    if depth == 0:
        return None, score(game_state, player), depth

    if player == player:
        max_val = float("-inf")
        best_depth = float("-inf")
        best_strategy = None

        next_states = available_states(game_state, player)

        for strategy, next_state in next_states:
            if next_state in past_states:
                continue

            if trade_pieces(game_state, next_state, player) or not next_state[game.other_player(player)]:
                return strategy, float("inf"), depth-1

            previous_states = past_states + [next_state]
            next_strategy, val, max_depth = maximiser(next_state, previous_states, depth - 1, player)

            if val > max_val:
                max_val = val
                best_strategy = strategy
            if val == max_val and max_depth > best_depth:
                max_val = val
                best_strategy = strategy
                best_depth = max_depth

        return best_strategy, max_val, best_depth

    if player == game.other_player(player):
        pass


# How to define the score of the current game state
# Open to change
# White wants to maximise this
# Black wants to minimise this
def score(game_state, player):
    # Winning Condition for white
    if not game_state[game.other_player(player)]:
        return float("inf")
    else:
        if not game_state[player]:
            return float("-inf")

    home_pieces = count_pieces(game_state[player])
    away_pieces = count_pieces(game_state[game.other_player(player)])
    dist_to_away = distance_from_centroid(game_state, player, own_centroid=False)

    home_stacks = count_stacks(game_state[player])
    away_stacks = count_stacks(game_state[game.other_player(player)])

    home_spread = distance_from_centroid(game_state, player, own_centroid=True)
    away_spread = distance_from_centroid(game_state, game.other_player(player), own_centroid=True)

    booms_n, booms_center, boom_weights = booms_required(game_state, player)

    boom_dist = dist_to_boom(game_state, player, booms_center, boom_weights)

    final = home_pieces - away_pieces - 1/(home_stacks*home_spread)

    return final


# Finds the 'children' of current game state
def available_states(game_state, player):
    available = []

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

                if count_pieces(temp_game_state[player]) < booms_required(temp_game_state, player)[0]:
                    continue

                available.append([(None, xy, move, None), temp_game.get_game_state()])
            else:
                for distance in range(piece[0]):
                    distance += 1

                    for amount in range(piece[0]):
                        amount += 1

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


def count_stacks(pieces):
    return len(pieces)


# Check to always trade one for one or more
def trade_pieces(game_state, next_state, player):
    # Boom was used
    if count_pieces(next_state[player]) < count_pieces(game_state[player]):

        # Boom killed some black pieces
        if count_pieces(next_state[game.other_player(player)]) < count_pieces(game_state[game.other_player(player)]):

            # Enough remaining pieces to cover the opposition pieces
            if count_pieces(next_state[player]) >= booms_required(next_state, player)[0]:
                return True

    return False


# Returns average center of mass of all pieces of other colour
def centre_of_mass(game_state, player):

    x, y, n = 0, 0, 0

    for piece in game_state[player]:
        x += piece[1]
        y += piece[2]
        n += 1

    return x/n, y/n


# Returns distance of a colours pieces from its centre of mass
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

    if dist==0:
        return 1

    return dist/n


# Finds the number of booms needed using single-linkage clustering
def booms_required(game_state, player):
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
        centroids.append((x/len(cluster), y/len(cluster)))

    return len(sets), centroids, weights


def dist_to_boom(game_state, player, booms_center, boom_weights):
    from math import pow, sqrt

    dist, n = 0, 0

    for piece in game_state[player]:
        for i in range(len(booms_center)):
            x, y = booms_center[i]
            weight = boom_weights[i]
            dist += weight * sqrt(pow(piece[2] - y, 2) + pow(piece[1] - x, 2))
            n += weight

    return dist/n
