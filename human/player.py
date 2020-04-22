
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

        print("You are {}.".format(colour))

    def action(self):
        """
        This method is called at the beginning of each of your turns to request 
        a choice of action from your program.

        Based on the current state of the game, your player should select and 
        return an allowed action to play on this turn. The action must be
        represented based on the spec's instructions for representing actions.
        """
        # TODO: Decide what action to take, and return it

        allowed_action = False
        action = None

        while not allowed_action:
            action = input("What is your move? ")

            if action in ("BOOM", "MOVE"):
                allowed_action = True
            else:
                print("Try again.")

        if action == "BOOM":
            allowed_pos = False

            while not allowed_pos:
                x = input("x position to boom: ")
                y = input("y position to boom: ")

                if x.isdigit() and y.isdigit():
                    x, y = int(x), int(y)

                    if not (x < 0 or x > 7) and not (y < 0 or y > 7):
                        return action, (x, y)
                    else:
                        print("Try again.")
                else:
                    print("Try again.")

        elif action == "MOVE":
            allowed_num, allowed_pos1, allowed_pos2 = False, False, False
            n, xy1, xy2 = None, None, None

            while not allowed_num:
                n = input("n to move: ")

                if n.isdigit():
                    n = int(n)

                    if n > 0:
                        allowed_num = True
                    else:
                        print("Try again.")
                else:
                    print("Try again.")

            while not allowed_pos1:
                x1 = input("x position to move from: ")
                y1 = input("y position to move from: ")

                if x1.isdigit() and y1.isdigit():
                    x1 = int(x1)
                    y1 = int(y1)

                    if not (x1 < 0 or x1 > 7) and not (y1 < 0 or y1 > 7):
                        xy1 = x1, y1

                        allowed_pos1 = True

                    else:
                        print("Try again.")
                else:
                    print("Try again.")

            while not allowed_pos2:
                x2 = input("x position to move to: ")
                y2 = input("y position to move to: ")

                if x2.isdigit() and y2.isdigit():
                    x2 = int(x2)
                    y2 = int(y2)

                    if not (x2 < 0 or x2 > 7) and not (y2 < 0 or y2 > 7):
                        xy2 = x2, y2

                        allowed_pos2 = True

                    else:
                        print("Try again.")
                else:
                    print("Try again.")

            return action, n, xy1, xy2
        else:
            print("This code is unreachable")

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

        return
