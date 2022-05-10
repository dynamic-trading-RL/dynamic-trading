
class State:

    def __init__(self):
        pass


class StateSpace:

    def __init__(self):
        pass


class ActionSpace:

    def __init__(self, state: State):

        self.state = state


class Action:

    def __init__(self, actionSpace: ActionSpace):

        self.actionSpace = actionSpace