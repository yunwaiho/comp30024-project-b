import MPM.search.features as features
import MPM.search.game as game
import MPM.search.tokens as tokens


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
        self.root = None

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

        home_score, away_score = self.score(self.game_state, self.game_state)
        score_diff = home_score - away_score

        # Protect our babies
        if score_diff > paranoid_threshold:
            strategy = "p"
        # Kill their babies
        else:
            strategy = "o"

        alpha = float("-inf")
        beta = float("inf")
        next_moves = None

        for depth in range(1, max_depth + 1):
            next_moves = self.strategy(curr_state=self.game_state, node=self.root, depth=depth,
                                       alpha=alpha, beta=beta, player=self.player, strategy=strategy,
                                       next_moves=next_moves)

        return next_moves[0]

    def strategy(self, curr_state, node, depth, alpha, beta, player, strategy, previous_moves):
        """
        Regular minimax function, with the difference being strategy on how the score is calculated

        Paranoid strategy wants to maximise home player's score
        Offensive strategy wants to maximise the difference between the player's score
        """

        if not None:


        # Base Case
        if depth == 0 or game.end(game_state):

            home_eval, away_eval = self.score(curr_state, game_state)

            if strategy == "p":
                return None, home_eval
            else:
                return None, (home_eval - away_eval)

        best_strategy = None
        next_states = self.get_state_utility(curr_state, game_state, player, strategy)

        if self.player == player:
            for node in next_states:
                strategy, next_state = node.previous_move, node.game_state

                next_strategy, val = self.strategy(curr_state, next_state, depth-1, alpha, beta, self.other, strategy)
                if val >= alpha:
                    alpha = val
                    best_strategy = strategy
                if alpha > beta:

                    break

        if self.player != player:
            for node in next_states:
                strategy, next_state = node.previous_move, node.game_state

                next_strategy, val = self.strategy(curr_state, next_state, depth-1, alpha, beta, self.player, strategy)
                if val <= beta:
                    beta = val
                    best_strategy = strategy
                if beta < alpha:
                    break

        if self.player == player:
            if alpha > beta:
                return best_strategy, beta
            return best_strategy, alpha
        else:
            if beta < alpha:
                return best_strategy, alpha
            return best_strategy, beta

    """
    # Regular minimax
    def paranoid(self, curr_state, game_state, depth, past_states, alpha, beta, player):
        Paranoid Algorithm: Same as regular minimax
        Assumes: The away player is trying to minimise home player's score
        Goal: Maximise the home player's score
        Used when the home player is in the lead

        # Base Case
        if depth == 0 or game.end(game_state):
            # print(score(game_state, self.player))

            home_eval, away_eval = self.score(curr_state, game_state, self.player, alg_type="paranoid")
            
            return None, home_eval

        best_strategy = None
        best_depth = float("-inf")
        next_states = self.available_states(game_state, player)

        if self.player == player:
            for strategy, next_state in next_states:

                #if next_state[player] in past_states:
                #    continue

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
            for strategy, next_state in next_states:

                #if next_state[player] in past_states:
                #    continue

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

############################### NOT FINISHED #################################################
    def offensive(self, curr_state, game_state, depth, past_states, alpha, beta, player, home_threshold):
        Directed Offensive Algorithm:
        Assumes: The away player is trying to maximise their own score
        Goal: Maximise the difference between home score and away score
        Used when: The home player is losing
        # Base Case
        if depth == 0 or game.end(game_state):
            # print(score(game_state, self.player))

            home_eval, away_eval = self.score(curr_state, game_state, self.player, alg_type="directed")

            return None, home_eval - away_eval

        best_strategy = None
        best_depth = float("-inf")
        #max_other_val = float("-inf")

        if self.player == player:

            next_states = self.available_states(game_state, player)

            for strategy, next_state in next_states:

                if next_state[player] in past_states:
                    continue

                next_strategy, eval = self.offensive(curr_state, next_state, depth - 1,
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
    """

    def one_enemy_endgame(self):

        game_state = self.game.get_game_state()
        enemy = game_state[self.other][0]
        enemy_xy = enemy[1], enemy[2]

        ally = self.closest_npiece(game_state, 1, self.player, enemy_xy)
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

    def score(self, curr_state, game_state):

        game_state_str = get_str(game_state)

        if game_state_str in self.tt:
            home_eval, away_eval = self.tt[game_state_str]
        else:
            home_eval = features.eval_function(curr_state, game_state, self.player)
            away_eval = features.eval_function(curr_state, game_state, self.other)

            self.tt[game_state_str] = (home_eval, away_eval)

        return home_eval, away_eval

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
                    temp_game_state = temp_game.get_game_state()

                    all_available.append([(None, xy, move, None), temp_game.get_game_state()])

                    home_a, away_a = features.count_all(temp_game_state, player)

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

                                available.append([(n, xy, move, distance), temp_game.get_game_state()])

        if player != self.player or len(available) == 0 or get_all:
            if len(all_rational_available) == 0:
                return all_available
            else:
                return all_rational_available

        return available

    def get_state_utility(self, curr_state, game_state, player, strategy):

        if self.root is None or self.root.children is None:
            children = []
            next_states = self.available_states(game_state, player)

            for next_strategy, next_state in next_states:
                state_score = self.utility(curr_state, next_state, strategy)
                children.append((state_score, (next_strategy, next_state)))

            indices = [(x[0], i) for i, x in enumerate(children)]
            ordered_children = [children[i][1] for x, i in indices]
            children = [self.Node(x, game.other_player(player), None) for x in ordered_children]

            if self.root is None:
                self.root = self.Node((None, curr_state), self.player, children)
            else:
                self.root.children = children

        return self.root.children

    # Can be replaced with another node utility function
    def utility(self, curr_state, next_state, strategy):

        home_eval, away_eval = self.score(curr_state, next_state)

        if strategy == "p":
            return home_eval
        else:
            return home_eval - away_eval

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
        home_b, away_b = features.count_all(curr_state, player)

        # Us booming is the same as someone adj booming on their next turn
        game_.boom(xy, player)
        next_state = game_.get_game_state()
        home_a, away_a = features.count_all(next_state, player)

        if self.is_bad_boom(home_b, home_a, away_b, away_a):
            return True

        return False

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


def get_str(game_state):
    import json

    return json.dumps(game_state)