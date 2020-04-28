import json
import numpy as np
import polkadots.search.game as game
import polkadots.search.agent as agent


class ExamplePlayer:
    def __init__(self, colour):
        """
        This method is called once at the beginning of the game to initialise
        your player. You should use this opportunity to set up your own internal
        representation of the game state, and any other information about the 
        game state you would like to maintain for the duration of the game.

        The parameter colour will be a string representing the player your 
        program will play as (White or Black). The value will be one of the 
        strings "white" or "black" correspondingly.
        """
        # TODO: Set up state representation.

        with open("polkadots/initial_game_state.json") as file:
            data = json.load(file)

        self.game = game.Game(data)
        self.colour = colour
        self.agent = agent.Agent(colour)

        self.turn_num = 1
        self.token_num = 12
        self.past_states = []

        self.state_score = {}

    def action(self):
        """
        This method is called at the beginning of each of your turns to request 
        a choice of action from your program.

        Based on the current state of the game, your player should select and 
        return an allowed action to play on this turn. The action must be
        represented based on the spec's instructions for representing actions.
        """
        # TODO: Decide what action to take, and return it

        import math

        # Computations
        c = 3600
        # Branching Factor
        b = 5 * self.token_num

        # Derived by computations = b^d
        search_depth = max(1, round(math.log(c) / math.log(b)))

        search_depth = 2

        alpha = float("-inf")
        beta = float("inf")

        game_state = self.game.get_game_state()

        strategy, score, features = self.agent.maximiser(game_state, game_state, search_depth, self.past_states, alpha, beta, self.colour)

        #print(features)

        features = np.append(features, score)

        self.state_score[self.turn_num] = features

        n, xy, move, distance = strategy
        if move == "Boom":

            return "BOOM", xy

        else:
            x_a, y_a = xy
            x_b, y_b = game.dir_to_xy(xy, move, distance)

            return "MOVE", n, (x_a, y_a), (x_b, y_b)

    def update(self, colour, action):
        """
        This method is called at the end of every turn (including your player’s 
        turns) to inform your player about the most recent action. You should 
        use this opportunity to maintain your internal representation of the 
        game state and any other information about the game you are storing.

        The parameter colour will be a string representing the player whose turn
        it is (White or Black). The value will be one of the strings "white" or
        "black" correspondingly.

        The parameter action is a representation of the most recent action
        conforming to the spec's instructions for representing actions.

        You may assume that action will always correspond to an allowed action 
        for the player colour (your method does not need to validate the action
        against the game rules).
        """
        # TODO: Update state representation in response to action.

        move = action[0]

        if move == "BOOM":
            xy = action[1]
            self.game.boom(xy, colour)

        else:
            n = action[1]
            xy1 = action[2]
            xy2 = action[3]

            self.game.board.move_token(n, xy1, xy2, check_valid=False)

        game_state = self.game.get_game_state()

        self.turn_num += 1
        self.token_num = agent.count_pieces(game_state["white"]) + agent.count_pieces(game_state["black"])

        import pandas as pd

        #print(self.state_score)

        df = pd.DataFrame(self.state_score)

        df = df.T

        df.to_excel("data.xlsx")
