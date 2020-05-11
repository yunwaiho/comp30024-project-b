import json
import MCTS_op.search.game as game
import MCTS_op.search.agent as agent
import MCTS_op.search.opening_book as opening_book

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

        with open("MCTS_op/initial_game_state.json") as file:
            data = json.load(file)

        self.game = game.Game(data)
        self.game_state = self.game.get_game_state()
        self.colour = colour
        self.past_states = []

        self.agent = agent.Agent(self.game, colour, self.past_states, trade_prop=0)
        # opening book change
        self.opening_book = opening_book.OpenBook(self.game, colour)
        self.trading_prop = 4

        self.home_tokens = 12
        self.away_tokens = 12
        self.turn = 0

    def action(self):
        """
        This method is called at the beginning of each of your turns to request 
        a choice of action from your program.

        Based on the current state of the game, your player should select and 
        return an allowed action to play on this turn. The action must be
        represented based on the spec's instructions for representing actions.
        """
        # TODO: Decide what action to take, and return it

        self.past_states.append(self.game_state[self.colour])

        self.home_tokens = sum([x[0] for x in self.game_state[self.colour]])
        self.away_tokens = sum([x[0] for x in self.game_state[game.other_player(self.colour)]])

        simulations = 10*self.home_tokens
        search_depth = 2

        ##################opening book change
        #action = None

        #if self.opening_book.check_early_game():
        #    action = self.opening_book.next_move
        #    if action:
        #        return action
        ########################################

        if self.away_tokens == 1 and self.home_tokens >= 1:
            strategy = self.agent.one_enemy_endgame(self.game_state, simulations, search_depth, 1)
        elif self.away_tokens == 2 and self.home_tokens >= 2:
            strategy = self.agent.two_enemy_endgame(self.game_state, simulations, search_depth, 1)
        elif self.away_tokens <= self.trading_prop and self.away_tokens < self.home_tokens:
            strategy = self.agent.trade_tokens(self.game_state, simulations, search_depth, 1)
        else:
            strategy = self.agent.monte_carlo(self.game_state, simulations, search_depth)

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

        # opening book change
        self.opening_book.update_early_game(colour, action)

        if move == "BOOM":
            xy = action[1]
            self.game.boom(xy, colour)

            if colour == self.colour:
                self.agent.home_recently_moved = None
            else:
                self.agent.away_recently_moved = None

        else:
            n = action[1]
            xy1 = action[2]
            xy2 = action[3]

            self.game.board.move_token(n, xy1, xy2, check_valid=False)

            if colour == self.colour:
                self.agent.home_recently_moved = xy2
            else:
                self.agent.away_recently_moved = xy2

        self.turn += 1
        self.agent.turn = self.turn // 2

        self.game_state = self.game.get_game_state()
        #self.agent.update_root(self.game_state)

    def end(self):

        game_state = self.game.get_game_state()
        self.agent.update_weights(game_state)

        with open("genetic_programming/score.json") as file:
            data = json.load(file)

        if game_state[self.colour]:
            if not game_state[game.other_player(self.colour)]:
                data[self.colour] += 1
            else:
                data["draw"] += 1
        elif game_state[game.other_player(self.colour)]:
            if not game_state[self.colour]:
                data[game.other_player(self.colour)] += 1
            else:
                data["draw"] += 1
        else:
            data["draw"] += 1


        with open("genetic_programming/score.json", 'w') as file:
            json.dump(data, file)
