import numpy as np
import pandas as pd
from scipy.optimize import shgo, minimize
from joblib import dump, load

from enums import FactorType
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import ActionSpace, Action, State


class Agent:

    def __init__(self, environment: Environment):

        self.environment = environment
        self._q_value_models = []
        self._set_agent_attributes()

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

    def dump_q_value_models(self):

        for n in range(len(self._q_value_models)):

            q_value_model = self._q_value_models[n]
            dump(q_value_model, '../data/supervised_regressors/q%d.joblib' % n)

    def load_q_value_models(self, n_batches: int):

        for n in range(n_batches):

            q_value_model = load('../data/supervised_regressors/q%d.joblib' % n)
            self.update_q_value_models(q_value_model)

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

        if len(self._q_value_models) == 0:
            action = self._random_action(state)

        else:
            rescaled_trade = self._optimize_q_value_trading(state, lower_bound, upper_bound)
            action = Action()
            action.set_trading_attributes(rescaled_trade=rescaled_trade, shares_scale=state.shares_scale)

        return action

    def _random_action_trading(self, state):

        lower_bound, upper_bound = self._get_action_bounds_trading(state)
        rescaled_trade = lower_bound + (upper_bound - lower_bound) * np.random.rand()
        action = Action()
        action.set_trading_attributes(rescaled_trade=rescaled_trade, shares_scale=state.shares_scale)

        return action

    def _get_action_bounds_trading(self, state: State):

        actionSpace = ActionSpace(state)
        actionSpace.set_trading_actions_interval()
        lower_bound, upper_bound = actionSpace.actions_interval

        return lower_bound, upper_bound

    def _q_value_trading(self, state: State, action: Action):

        q_value_model_input = self.extract_q_value_model_input_trading(state, action)

        qvl = 0.

        for q_value_model in self._q_value_models:

            qvl = 0.5 * (qvl + q_value_model.predict(q_value_model_input))

        return qvl

    def _optimize_q_value_trading(self, state: State, lower_bound: float, upper_bound: float):

        def func(rescaled_trade):

            action = Action()
            action.set_trading_attributes(rescaled_trade=rescaled_trade)

            qvl = self._q_value_trading(state, action)

            return - qvl

        bounds = [(lower_bound, upper_bound)]
        # res = shgo(func=func, bounds=bounds)
        res = minimize(fun=func, bounds=bounds, x0=0.)

        return res.x[0]

    def extract_q_value_model_input_trading(self, state, action):

        state_lst = self._extract_state_lst_trading(state)
        rescaled_trade = action.rescaled_trade
        q_value_model_input = state_lst + [rescaled_trade]
        q_value_model_input = np.array(q_value_model_input, dtype=object).reshape(1, -1)

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

    def _set_agent_attributes(self):

        ticker = self.environment.market.ticker
        filename = '../data/data_source/trading_data/' + ticker + '-trading-parameters.csv'
        df_trad_params = pd.read_csv(filename, index_col=0)
        gamma = df_trad_params.loc['gamma'][0]
        kappa = df_trad_params.loc['kappa'][0]

        self.gamma = gamma
        self.kappa = kappa

        self.environment._get_trading_parameters_from_agent(self.gamma, self.kappa)
