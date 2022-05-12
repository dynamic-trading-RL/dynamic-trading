import numpy as np
from scipy.optimize import shgo

from enums import FactorType
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import ActionSpace, Action, State


class Agent:

    def __init__(self, environment: Environment):

        self.environment = environment
        self._q_value_models = []

    def policy(self, state: State, eps: float = None):

        if eps is None:
            action = self._greedy_policy(state)
        else:
            action = self._eps_greedy_policy(state, eps)

        return action

    def q_value(self, state: State, action: Action):

        qvl = self._q_value_trading(state, action)

        return qvl

    def update_q_value_models(self, q_value_model):

        self._q_value_models.append(q_value_model)

    def _greedy_policy(self, state: State):

        action = self._greedy_policy_trading(state)

        return action

    def _eps_greedy_policy(self, state: State, eps: float):

        u = np.random.rand()

        if u < eps:
            action = self._random_action(state)
        else:
            action = self._greedy_policy(state)

        return action

    def _random_action(self, state: State):

        action = self._random_action_trading(state)

        return action

    def _greedy_policy_trading(self, state):

        lower_bound, upper_bound = self._get_action_bounds_trading(state)
        rescaled_trade = self._optimize_q_value_trading(state, lower_bound, upper_bound)
        action = Action()
        action.set_trading_attributes(rescaled_trade=rescaled_trade, shares_scale=state.shares_scale)

        return action

    def _random_action_trading(self, state):

        lower_bound, upper_bound = self._get_action_bounds_trading(state)
        rescaled_trade = lower_bound + (upper_bound - lower_bound) * np.random.rand()
        action = Action(rescaled_trade)

        return action

    def _get_action_bounds_trading(self, state: State):

        actionSpace = ActionSpace(state)
        actionSpace.set_trading_actions_interval()
        lower_bound, upper_bound = actionSpace.actions_interval

        return lower_bound, upper_bound

    def _q_value_trading(self, state: State, action: Action):

        q_value_model_input = self._extract_q_value_model_input_trading(state, action)

        qvl = 0.

        for q_value_model in self._q_value_models:

            qvl = 0.5 * (qvl + q_value_model(q_value_model_input))

        return qvl

    def _optimize_q_value_trading(self, state: State, lower_bound: float, upper_bound: float):

        def func(rescaled_trade):

            action = Action()
            action.set_trading_attributes(rescaled_trade=rescaled_trade)

            qvl = self._q_value_trading(state, action)

            return - qvl

        bounds = [(lower_bound, upper_bound)]
        res = shgo(func=func, bounds=bounds)

        return res.x[0]

    def _extract_q_value_model_input_trading(self, state, action):

        state_lst = self._extract_state_lst_trading(state)
        rescaled_trade = action.rescaled_trade
        q_value_model_input = state_lst + [rescaled_trade]

        return q_value_model_input

    def _extract_state_lst_trading(self, state):

        if self.environment.factorType == FactorType.Observable:
            current_factor = state.current_factor
            current_rescaled_shares = state.current_rescaled_shares
            state_lst = [current_factor, current_rescaled_shares]

        elif self.environment.factorType == FactorType.Latent:
            current_other_observable = state.current_other_observable
            current_rescaled_shares = state.current_rescaled_shares
            state_lst = [current_other_observable, current_rescaled_shares]

        else:
            raise NameError('Invalid factorType: ' + self.environment.factorType.value)

        return state_lst
