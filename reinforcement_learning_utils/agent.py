import numpy as np
import pandas as pd
import os
from scipy.optimize import basinhopping, differential_evolution, dual_annealing, shgo, minimize
from joblib import dump, load
from scipy.stats import truncnorm

from enums import RandomActionType, StrategyType, OptimizerType, InitialQvalueEstimateType
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import ActionSpace, Action, State


class Agent:

    def __init__(self, environment: Environment,
                 optimizerType: OptimizerType = OptimizerType.shgo,
                 average_across_models: bool = True,
                 use_best_n_batch: bool = False,
                 initialQvalueEstimateType: InitialQvalueEstimateType = InitialQvalueEstimateType.zero):

        self.environment = environment

        # available optimizers: ('basinhopping', 'brute', 'differential_evolution', 'dual_annealing', 'shgo', 'local')
        self._optimizerType = optimizerType
        print(f'Using optimizer={self._optimizerType}')

        self._q_value_models = []
        self._set_agent_attributes()

        self._average_across_models = average_across_models
        self._use_best_n_batch = use_best_n_batch
        self._initialQvalueEstimateType = initialQvalueEstimateType

        self._tol = 10**-10  # numerical tolerance for bound conditions requirements

    def policy(self, state: State, eps: float = None):

        if eps is None:
            action = self._greedy_policy(state)
        else:
            action = self._eps_greedy_policy(state, eps)

        if np.abs(state.rescaled_shares + action.rescaled_trade) > 1 + self._tol:
            raise NameError(
                f'Shares went out of bound!! \n  rescaled_shares: {state.rescaled_shares:.2f} \n  rescaled_trade: {action.rescaled_trade:.2f}')

        return action

    def q_value(self, state: State, action: Action):

        qvl = self._q_value_trading(state, action)

        return qvl

    def update_q_value_models(self, q_value_model):

        self._q_value_models.append(q_value_model)

    def dump_q_value_models(self):

        for n in range(len(self._q_value_models)):
            q_value_model = self._q_value_models[n]
            dump(q_value_model,
                 os.path.dirname(os.path.dirname(__file__)) + '/data/supervised_regressors/q%d.joblib' % n)

    def load_q_value_models(self, n_batches: int):

        if self._use_best_n_batch:
            if self.best_n is None:
                n_batches = 1
            else:
                n_batches = self.best_n

        for n in range(n_batches):
            q_value_model = load(
                os.path.dirname(os.path.dirname(__file__)) + '/data/supervised_regressors/q%d.joblib' % n)
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

        if len(self._q_value_models) == 0:  # We are at batch 0: use initialization

            if self.randomActionType in (RandomActionType.RandomUniform,
                                                   RandomActionType.RandomTruncNorm):

                action = self._random_action(state)

            elif self.randomActionType == RandomActionType.GP:

                rescaled_trade = self.environment.agent_GP.policy(factor=state.factor,
                                                                  rescaled_shares=state.rescaled_shares,
                                                                  shares_scale=state.shares_scale,
                                                                  price=state.price)
                action = Action()
                action.set_trading_attributes(rescaled_trade=rescaled_trade, shares_scale=state.shares_scale)

            else:
                raise NameError('Invalid randomActionType: ' + self.randomActionType.value)

        else:
            rescaled_trade = self._optimize_q_value_trading(state)
            action = Action()
            action.set_trading_attributes(rescaled_trade=rescaled_trade, shares_scale=state.shares_scale)

        return action

    def _random_action_trading(self, state):

        lower_bound, upper_bound = self._get_action_bounds_trading(state)

        if self.randomActionType == RandomActionType.RandomUniform:

            rescaled_trade = lower_bound + (upper_bound - lower_bound) * np.random.rand()

        else:

            # TODO: We are implying a preference for randomActionType.RandomTruncNorm, in that this is chosen
            #  by default even if the initialization is set to GP. Make a better structuring of this part.

            loc = self._get_trade_loc(lower_bound, upper_bound)
            alpha = 0.12
            scale = self._get_trade_scale(lower_bound, upper_bound, alpha)

            rescaled_trade = truncnorm.rvs(a=lower_bound, b=upper_bound, loc=loc, scale=scale)

        action = Action()
        action.set_trading_attributes(rescaled_trade=rescaled_trade, shares_scale=state.shares_scale)

        return action

    def _get_trade_loc(self, lower_bound, upper_bound):

        # loc = 0.5 * (lower_bound + upper_bound)
        loc = 0.

        return loc

    def _get_trade_scale(self, lower_bound, upper_bound, alpha):

        scale = alpha * (upper_bound - lower_bound)

        return scale

    def _get_action_bounds_trading(self, state: State):

        actionSpace = ActionSpace(state, self.strategyType)
        actionSpace.set_trading_actions_interval()
        lower_bound, upper_bound = actionSpace.actions_interval

        return lower_bound, upper_bound

    def _q_value_trading(self, state: State, action: Action):

        if len(self._q_value_models) == 0:

            if self._initialQvalueEstimateType == InitialQvalueEstimateType.random:
                qvl = state.average_past_pnl*state.shares * np.random.randn()

            elif self._initialQvalueEstimateType == InitialQvalueEstimateType.zero:
                qvl = 0.

        else:

            q_value_model_input = self.extract_q_value_model_input_trading(state, action)

            qvl = 0.

            if self._average_across_models:
                for q_value_model in self._q_value_models:
                    qvl = 0.5 * (qvl + q_value_model.predict(q_value_model_input))
            else:
                q_value_model = self._q_value_models[-1]
                qvl = q_value_model.predict(q_value_model_input)

        return qvl

    def _optimize_q_value_trading(self, state: State):

        lower_bound, upper_bound = self._get_action_bounds_trading(state)

        def func(rescaled_trade):

            action = Action()
            action.set_trading_attributes(rescaled_trade=rescaled_trade)

            qvl = self._q_value_trading(state, action)

            return - qvl

        bounds = [(lower_bound, upper_bound)]
        x0 = self._get_trade_loc(lower_bound, upper_bound)

        if self._optimizerType == OptimizerType.basinhopping:
            res = basinhopping(func=func, x0=x0)
            return res.x[0]

        elif self._optimizerType == OptimizerType.brute:

            Ns = 4 * int(upper_bound * state.shares_scale - lower_bound * state.shares_scale)
            xx = np.linspace(lower_bound, upper_bound, Ns)
            ff = np.array([func(x) for x in xx])
            x = xx[np.argmin(ff)]
            return x

        elif self._optimizerType == OptimizerType.differential_evolution:
            res = differential_evolution(func=func, bounds=bounds)
            return res.x[0]

        elif self._optimizerType == OptimizerType.dual_annealing:
            res = dual_annealing(func=func, bounds=bounds)
            return res.x[0]

        elif self._optimizerType == OptimizerType.shgo:
            res = shgo(func=func, bounds=bounds)
            return res.x[0]

        elif self._optimizerType == OptimizerType.local:
            res = minimize(fun=func, bounds=bounds, x0=np.array(x0))
            return res.x[0]

        else:
            raise NameError(f'Invalid optimizerType: {self._optimizerType}')

    def extract_q_value_model_input_trading(self, state, action):

        state_lst = self._extract_state_lst_trading(state)
        rescaled_trade = action.rescaled_trade
        q_value_model_input = state_lst + [rescaled_trade]
        q_value_model_input = np.array(q_value_model_input, dtype=object).reshape(1, -1)

        return q_value_model_input

    def _extract_state_lst_trading(self, state):

        state_shape = state.environment.state_shape
        state_lst = [None] * len(state_shape)

        # get info
        rescaled_shares = state.rescaled_shares
        factor = state.factor
        ttm = state.ttm
        price = state.price
        pnl = state.pnl
        average_past_pnl = state.average_past_pnl
        try:
            action_GP = state.action_GP
        except:
            action_GP = None

        # fill list
        state_lst[0] = rescaled_shares

        if self.environment.factor_in_state:
            state_lst[state_shape['factor']] = factor
        if self.environment.ttm_in_state:
            state_lst[state_shape['ttm']] = ttm
        if self.environment.price_in_state:
            state_lst[state_shape['price']] = price
        if self.environment.pnl_in_state:
            state_lst[state_shape['pnl']] = pnl
        if self.environment.average_past_pnl_in_state:
            state_lst[state_shape['average_past_pnl']] = average_past_pnl
        if self.environment.GP_action_in_state:
            state_lst[state_shape['action_GP']] = action_GP

        return state_lst

    def _set_agent_attributes(self):

        filename = os.path.dirname(os.path.dirname(__file__)) + '/data/data_source/settings/settings.csv'
        df_trad_params = pd.read_csv(filename, index_col=0)

        filename = os.path.dirname(os.path.dirname(__file__)) +\
                   '/data/data_source/market_data/commodities-summary-statistics.xlsx'
        df_lam_kappa = pd.read_excel(filename, index_col=0, sheet_name='Simplified contract multiplier')
        df_lam_kappa = df_lam_kappa.loc[self.environment.market.ticker]  # TODO: should it be self.environment.ticker?

        gamma = float(df_trad_params.loc['gamma'][0])
        kappa = float(df_lam_kappa.loc['kappa'])

        randomActionType = \
            RandomActionType(str(df_trad_params.loc['randomActionType'][0]))
        strategyType = StrategyType(df_trad_params.loc['strategyType'][0])

        self.gamma = gamma
        self.kappa = kappa
        self.randomActionType = randomActionType
        self.strategyType = strategyType

        self.environment.get_trading_parameters_from_agent(self.gamma, self.kappa)

        if self.randomActionType == RandomActionType.GP:
            self.environment.observe_GP = True
            self.environment.instantiate_market_benchmark_and_agent_GP()

        try:
            self.best_n = int(load(os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/best_n.joblib'))
        except:
            self.best_n = None
            print('Notice: agent is not yet trained')