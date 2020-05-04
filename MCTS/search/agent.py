import numpy as np
import pandas as pd
import random

import MCTS.search.game as game
import MCTS.search.tokens as tokens

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

        home_c = count_pieces(game_state[self.player])
        away_c = count_pieces(game_state[self.other])

        potential_threat = self.has_potential_threat(self.away_recently_moved, self.player)

        if potential_threat:
            temp_game = game.Game(game_state)
            temp_game.boom(self.away_recently_moved, self.other)
            temp_game_state = temp_game.get_game_state()
            home_t = count_pieces(temp_game_state[self.player])
            away_t = count_pieces(temp_game_state[self.other])
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

                damage = sqrt(pieces_per_boom(next_state, self.other))
                if damage == count_stacks(next_state):
                    continue

                if self.has_potential_threat(xy, self.other):
                    home_b = count_pieces(next_state[self.player])
                    away_b = count_pieces(next_state[self.other])

                    temp_game = game.Game(next_state)
                    temp_game.boom(xy, self.player)
                    temp_game_state = temp_game.get_game_state()

                    home_a = count_pieces(temp_game_state[self.player])
                    away_a = count_pieces(temp_game_state[self.other])

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

                    home_l = count_pieces(temp_game_state[self.player])
                    away_l = count_pieces(temp_game_state[self.other])
                    loss = home_l - away_l

                    # Gains
                    temp_game1 = game.Game(next_state)
                    temp_game1.boom(xy, self.player)
                    temp_game_state = temp_game1.get_game_state()

                    home_g1 = count_pieces(temp_game_state[self.player])
                    away_g1 = count_pieces(temp_game_state[self.other])
                    gain1 = home_g1 - away_g1

                    gain2 = float("-inf")
                    # Leave one behind and boom there
                    if not temp_game1.board.is_cell_empty(strategy[1]):
                        temp_game2 = game.Game(next_state)
                        temp_game2.boom(strategy[1], self.player)
                        temp_game_state = temp_game2.get_game_state()

                        home_g2 = count_pieces(temp_game_state[self.player])
                        away_g2 = count_pieces(temp_game_state[self.other])
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

                        home_o = count_pieces(trade_game_state[self.player])
                        away_o = count_pieces(trade_game_state[self.other])
                        outcome = home_o - away_o

                        run_aways.append((outcome, strategy))

                    # Minimise Losses
                    if loss > potential_diff:
                        run_aways.append((loss, strategy))

            # If can trade for more
            if strategy[2] == "Boom":
                home_a = count_pieces(next_state[self.player])
                away_a = count_pieces(next_state[self.other])
                diff = home_a - away_a

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
                best_move_diff = best_move[0]

            # Trade first
            if best_boom_diff > potential_diff and can_boom:
                if best_move_diff > best_boom_diff:
                    return best_move[1]
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
        return self.evaluation(curr_state, next_state, player, utility=True)

    # How to define the score of the current game state
    # Open to change
    def score(self, curr_state, game_state, player, utility):

        other = game.other_player(player)

        if utility:
            if not game_state[player]:
                return float("-inf"), float("inf")
            if not game_state[other]:
                return float("inf"), float("-inf")

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

        if home_pieces_diff == 0 or away_pieces_diff:
            home_diff_ratio = 0
            away_diff_ratio = 0
        else:
            home_diff_ratio = home_pieces_diff/away_pieces_diff
            away_diff_ratio = away_pieces_diff/home_pieces_diff

        home_stacks = count_stacks(a_home_pieces)
        away_stacks = count_stacks(a_away_pieces)

        home_stack_size = average_stack_size(a_home_pieces)
        away_stack_size = average_stack_size(a_away_pieces)

        home_threat = min_dist_to_boom(game_state, player)
        away_threat = min_dist_to_boom(game_state, other)

        max_damage = pieces_per_boom(game_state, player)
        max_losses = pieces_per_boom(game_state, other)

        home_board_score = self.get_board_score(game_state, player)
        away_board_score = self.get_board_score(game_state, other)

        home_features = np.array([
            home_num,
            away_num,
            away_pieces_diff,
            home_pieces_diff,
            home_diff_ratio,
            self.turn*home_stacks,
            home_stack_size,
            self.turn*home_threat,
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
            self.turn*away_stacks,
            away_stack_size,
            self.turn*away_threat,
            max_losses,
            max_damage,
            away_board_score,
            home_board_score
        ])

        home_final = np.dot(home_features, self.weights)
        away_final = np.dot(away_features, self.weights)

        return home_final, away_final

    def evaluation(self, curr_state, game_state, player, utility=False):

        home_score, away_score = self.score(curr_state, game_state, self.player, utility)
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
            df.iloc[self.weight_index, i+1] = lst[i]

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
            return other_scores/total_scores

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
                    if piece[0] == 1:
                        amounts = [1]
                    elif piece[0] == 2:
                        amounts = [1, 2]
                    else:
                        amount = min(piece[0], 8)
                        # Move whole stack or leave one or move one
                        amounts = [1, amount, amount-1]

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

    def creates_v(self, game_, xy):
        ally_pieces, all_pieces = self.count_adjacent(self.other, xy, game_=game_)

        checked = False

        i = 1
        # Checks for a v, means two bits in a row in bit string
        while i < len(2*ally_pieces):
            if (2*ally_pieces)[i] == "1":
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

    def one_enemy_endgame(self):

        game_state = self.game.get_game_state()
        enemy = game_state[self.other][0]
        enemy_xy = enemy[1], enemy[2]

        ally = closest_piece(game_state, self.player, enemy_xy)
        ally_xy = ally[1], ally[2]

        width = enemy_xy[0] - ally_xy[0]
        height = enemy_xy[1] - ally_xy[1]

        if abs(width) <= 1 and abs(height) <= 1:
            return None, ally_xy, "Boom", None

        # Move vertically
        if abs(height) > abs(width):
            if ally[0] >= abs(height):
                if height > 0:
                    return 1, ally_xy, "Up", abs(height)
                else:
                    return 1, ally_xy, "Down", abs(height)
            else:
                if height > 0:
                    return ally[0], ally_xy, "Up", ally[0]
                else:
                    return ally[0], ally_xy, "Down", ally[0]
        # Move horizontally
        else:
            if ally[0] >= abs(width):
                if width > 0:
                    return 1, ally_xy, "Right", abs(width)
                else:
                    return 1, ally_xy, "Left", abs(width)
            else:
                if width > 0:
                    return ally[0], ally_xy, "Right", ally[0]
                else:
                    return ally[0], ally_xy, "Left", ally[0]
    """
    def two_enemy_endgame(self):
        game_state = self.game.get_game_state()
        enemy_stacks = len(game_state[self.other])
        
        # One stack
        if enemy_stacks == 1:
            enemy = game_state[self.other][0]
            enemy_xy = enemy[1], enemy[2]

            ally = closest_piece(game_state, self.player, enemy_xy)
            ally_xy = ally[1], ally[2]

            # Close enough to boom
            if abs(enemy_xy[1] - ally_xy[1]) <= 1 and abs(enemy_xy[2] - ally_xy[2]) <= 1:
                return None, ally_xy, "Boom", None

            # Largest piece
            piece = sorted(game_state[self.player], reverse=True)[0]
            # No stacks of two or more
            if piece[0] == 1:
                pass
            # There is ally two or more stack
            else:
                ally_xy = piece[1], piece[2]

                width = enemy_xy[1] - ally_xy[1]
                height = enemy_xy[2] - ally_xy[2]



        # Two seperate stacks
        else:
            return self.one_enemy_endgame()
    """

    def stack_up(self):

        game_state = self.game.get_game_state()

        if self.player == "white":
            row = 1
        else:
            row = 6

        if self.away_recently_moved is None:
            start = random.choice([0, 7])
            piece = [1, start, row]
        else:
            # 12 Stacks
            if self.home_recently_moved is None:
                if self.away_recently_moved[0] < 4:
                    piece = [1, 0, row]
                else:
                    piece = [1, 7, row]
            # Already Moved
            else:
                piece = sorted(game_state[self.player], reverse=True)[0]

        alternate = True
        i = 1
        while True:
            xy = piece[1] + i, piece[2]
            if tokens.out_of_board(xy) or self.game.board.is_cell_empty(xy):
                if alternate:
                    i = -i
                    alternate = False
                else:
                    i = abs(i) + 1
                    alternate = True
                continue
            else:
                break

        if i > 0:
            direction = "Right"
        else:
            direction = "Left"

        return piece[0], (piece[1], piece[2]), direction, abs(i)









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

    return max(damages)*max(damages)


