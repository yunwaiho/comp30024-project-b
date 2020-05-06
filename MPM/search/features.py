# Module for the evalation function and the features used

import MPM.search.tokens as tokens
import MPM.search.board as board
import MPM.search.game as game

import numpy as np


"""def eval_function(curr_state, game_state, player, curr_state_check=False):

    # Write a get turn function

    other = game.other_player(player)

    game_pieces = curr_state[player]
    game_num_pieces = count_pieces(game_state[player])

    game_home_stacks = count_stacks(game_pieces)
    game_stack_size = average_stack_size(game_pieces)

    game_threat = min_dist_to_boom(curr_state, player)

    max_damage = pieces_per_boom(curr_state, player)
    max_losses = pieces_per_boom(curr_state, other)

    # Calcuate all the features for the game state board
    if curr_state_check:

        total_features = np.array([
            game_num_pieces,
            game_home_stacks,
            game_stack_size,
            game_threat,
            max_damage,
            max_losses
        ])

        weights = np.full(len(total_features), 1)
        final = np.dot(total_features, weights)
        return final
    else:
        total_features = np.array([
            game_num_pieces,
            game_home_stacks,
            game_stack_size,
            game_threat,
            max_damage,
            max_losses
        ])
        weights = np.full(len(total_features), 1)
        final = np.dot(total_features, weights)
        return final

"""


def eval_function(agent, curr_state, game_state, player, turn):
    other = game.other_player(player)

    tokens.board_configs()

    b_home_pieces = curr_state[player]
    b_away_pieces = curr_state[other]

    a_home_pieces = game_state[player]
    a_away_pieces = game_state[other]

    home_num = count_pieces(a_home_pieces)
    away_num = count_pieces(a_away_pieces)
    total_num = home_num + away_num

    if total_num == 0:
        return 0, 0

    home_pieces_diff = count_pieces(b_home_pieces) - home_num
    away_pieces_diff = count_pieces(b_away_pieces) - away_num

    # Higher differences have more impact on the game
    home_pieces_diff = home_pieces_diff * home_pieces_diff
    away_pieces_diff = away_pieces_diff * away_pieces_diff

    if home_pieces_diff == 0 or away_pieces_diff == 0:
        home_diff_ratio = 0
        away_diff_ratio = 0
    else:
        home_diff_ratio = home_pieces_diff / away_pieces_diff
        away_diff_ratio = away_pieces_diff / home_pieces_diff

    home_stacks = count_stacks(a_home_pieces)
    away_stacks = count_stacks(a_away_pieces)

    home_stack_size = average_stack_size(a_home_pieces)
    away_stack_size = average_stack_size(a_away_pieces)

    home_threat = min_dist_to_boom(game_state, player)
    away_threat = min_dist_to_boom(game_state, other)

    max_damage = pieces_per_boom(game_state, player)
    max_losses = pieces_per_boom(game_state, other)

    home_board_score = agent.get_board_score(game_state, player)
    away_board_score = agent.get_board_score(game_state, other)

    weights = np.array([
        100,
        22,
        -55,
        -47,
        180,
        -180,
        -11,
        -35,
        220,
        50,
        -7,
        -10,
    ])

    home_features = np.array([
        home_num,
        away_num,
        away_pieces_diff,
        home_pieces_diff,
        home_diff_ratio,
        turn * home_stacks,
        home_stack_size,
        turn * home_threat,
        max_damage,
        max_losses,
        home_board_score,
        away_board_score,
    ])

    away_features = np.array([
        away_num,
        home_num,
        home_pieces_diff,
        away_pieces_diff,
        away_diff_ratio,
        turn * away_stacks,
        away_stack_size,
        turn * away_threat,
        max_losses,
        max_damage,
        away_board_score,
        home_board_score
    ])

    home_final = np.dot(home_features, weights)
    away_final = np.dot(away_features, weights)

    return home_final, away_final

# Counts the number of pieces
def count_pieces(pieces):
    n = 0
    for piece in pieces:
        n += piece[0]
    return n


# Counts the number of stacks
def count_stacks(pieces):
    return len(pieces)


# Counts the number of pieces for both players
def count_all(game_state, player):
    home = sum([x[0] for x in game_state[player]])
    away = sum([x[0] for x in game_state[game.other_player(player)]])

    return home, away


# Counts the average stack size
def average_stack_size(pieces):
    if count_pieces(pieces) == 0:
        return 0

    score = 0

    for piece in pieces:
        score += piece[0] ^ 2

    return score / count_stacks(pieces)


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

            minimum = min(minimum, ceil(dist / piece1[0]))

    return minimum


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

    return max(damages) * max(damages)


def closest_wall(xy):
    x, y = xy

    if x < 4:
        horizontal = "Left"
    else:
        horizontal = "Right"
    if y < 4:
        vertical = "Bottom"
    else:
        vertical = "Top"

    x_dist = min(x, 7 - x)
    y_dist = min(y, 7 - y)

    if x_dist < y_dist:
        return horizontal
    # Defaults to vertical if equal or less
    else:
        return vertical


def closest_piece(game_state, player, xy):
    from math import sqrt, pow

    x1, y1 = xy
    max_dist = float("inf")
    closest_ally = None

    for piece in game_state[player]:
        x2, y2 = piece[1], piece[2]

        dist = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

        if dist < max_dist:
            max_dist = dist
            closest_ally = piece

    return closest_ally



