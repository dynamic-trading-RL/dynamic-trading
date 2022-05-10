import numpy as np
from reinforcement_learning_utils.state_action_utils import ActionSpace, Action, State


class Agent:

    def __init__(self):

        self.q_value_models = []

    def policy(self, state: State, eps: float = None):

        if eps is None:
            action = self._greedy_policy(state)
        else:
            action = self._eps_greedy_policy(state, eps)

        return action

    def q_value(self, state: State, action: Action):

        qvl = self._q_value_impl(state, action)

        return qvl

    def _greedy_policy(self, state: State):

        lower_bound, upper_bound = self._get_action_bounds(state)

        rescaled_trade = self._optimize_q_value(lower_bound, upper_bound)
        action = Action(rescaled_trade)

        return action

    def _eps_greedy_policy(self, state: State, eps: float):

        u = np.random.rand()

        if u < eps:
            action = self._random_action(state)
        else:
            action = self._greedy_policy(state)

        return action

    def _random_action(self, state: State):

        lower_bound, upper_bound = self._get_action_bounds(state)
        rescaled_trade = lower_bound + (upper_bound - lower_bound) * np.random.rand()
        action = Action(rescaled_trade)

        return action

    def _get_action_bounds(self, state: State):

        actionSpace = ActionSpace(state)
        lower_bound, upper_bound = actionSpace.actions_interval

        return lower_bound, upper_bound

    def _optimize_q_value(self, lower_bound: float, upper_bound: float):

        return 0.5 * (lower_bound + upper_bound)  # should be optimized

    def _q_value_impl(self, state: State, action: Action):

        qvl = 0.

        for q_value_model in self.q_value_models:

            qvl = 0.5 * (qvl + q_value_model(state, action))

        return qvl
