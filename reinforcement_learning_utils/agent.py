from reinforcement_learning_utils.state_action_utils import ActionSpace, Action, State


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
