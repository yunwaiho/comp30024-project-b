import random_player.search.game as game
import random_player.search.tokens as tokens


# Finds the 'children' of current game state
def available_states(game_state, player):
    available = []

    for piece in game_state[player]:
        xy = piece[1], piece[2]
        available_moves = tokens.available_moves(player)

        for move in available_moves:
            if move == "Boom":
                temp_game = game.Game(game_state)
                if not temp_game.has_surrounding(xy):
                    continue

                available.append((None, xy, move, None))
            else:
                for distance in range(piece[0]):
                    distance += 1

                    for amount in range(piece[0]):
                        amount += 1

                        temp_game = game.Game(game_state)
                        if temp_game.is_valid_move(xy, move, distance, player):
                            available.append((amount, xy, move, distance))

    return available
