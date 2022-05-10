from environment import State


class ActionSpace:

    def __init__(self, state: State):

        self.state = state


class Action:

    def __init__(self, actionSpace: ActionSpace):

        self.actionSpace = actionSpace


class Agent:

    def __init__(self):
        pass

    def policy(self, state: State):

        actionSpace = ActionSpace(state)
        action = Action(actionSpace)

        return action

    def q_value(self, state: State, action: Action):

        qvl = 0.

        return qvl
