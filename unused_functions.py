# Counts the average stack size
def average_stack_size(pieces):
    if count_pieces(pieces) == 0:
        return 0

    score = 0

    for piece in pieces:
        score += piece[0] ^ 2

    return score / count_stacks(pieces)

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


# New
    """
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

                                potential_threat = self.has_potential_threat(xy2, self.other)

                                # Moving into a trap
                                #if potential_threat and self.suicide_move(temp_game, player, xy2):
                                #    continue

                                all_rational_available.append([(n, xy, move, distance), temp_game.get_game_state()])

                                if self.player == player and n not in amounts:
                                    continue

                                if piece[0] != 1 and n == 1 and not potential_threat:
                                    continue

                                # We don't like a v pattern (inefficient move)
                                if self.creates_v(temp_game, xy2):
                                    continue
                                if move in ["Up", "Down"] and (self.creates_v(temp_game, (xy[0] + 1, xy[1]))
                                                               or self.creates_v(temp_game, (xy[0] - 1, xy[1]))):
                                    continue
                                if move in ["Left", "Right"] and (self.creates_v(temp_game, (xy[0], xy[1] + 1))
                                                                  or self.creates_v(temp_game, (xy[0], xy[1] - 1))):
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
    """

    def suicide_move(self, game_, player, xy):
        curr_state = game_.get_game_state()
        temp_game = game.Game(curr_state)

        # Us booming is the same as someone adj booming on their next turn
        temp_game.boom(xy, player)
        next_state = temp_game.get_game_state()

        if self.is_bad_boom(curr_state, next_state, player):
            return True
        return False