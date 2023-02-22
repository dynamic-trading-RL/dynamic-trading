import numpy as np
import pandas as pd
import os
from scipy.optimize import basinhopping, differential_evolution, dual_annealing, shgo, minimize
from joblib import dump, load
from scipy.stats import truncnorm

from dynamic_trading.enums.enums import (RandomActionType, StrategyType, OptimizerType, InitialQvalueEstimateType,
                                         SupervisedRegressorType)
from dynamic_trading.gen_utils.utils import instantiate_polynomialFeatures, find_polynomial_minimum
from dynamic_trading.reinforcement_learning_utils.environment import Environment
from dynamic_trading.reinforcement_learning_utils.state_action_utils import ActionSpace, Action, State


class Agent:
    """
    Class defining a reinforcement learning agent.

    """

    def __init__(self, environment: Environment,
                 optimizerType: OptimizerType = OptimizerType.shgo,
                 average_across_models: bool = True,
                 use_best_n_batch: bool = False,
                 initialQvalueEstimateType: InitialQvalueEstimateType = InitialQvalueEstimateType.zero,
                 supervisedRegressorType: SupervisedRegressorType = SupervisedRegressorType.ann,
                 polynomial_regression_degree: int = None,
                 alpha_ewma: float = 0.5):
        """
        Class constructor.

        Parameters
        ----------
        environment : Environment
            Environment in which the agent is operating. See :class:`~dynamic_trading.market_utils.financial_time_series.FinancialTimeSeries` for more details.
        optimizerType ::class:`~dynamic_trading.enums.enums.OptimizerType`
            Determines which global optimizer to use in the greedy policy optimization. Refer to
            :class:`~dynamic_trading.enums.enums.OptimizerType` for more details.
        average_across_models : bool
            Boolean determining whether the SARSA algorithm performs model averaging across batches.
        use_best_n_batch : bool
            Boolean determining whether the last or the best available (average of) model should be used.
        initialQvalueEstimateType : :class:`~dynamic_trading.enums.enums.InitialQvalueEstimateType`
            Setting for the initialization of the state-action value function. Refer to
            :class:`~dynamic_trading.enums.enums.InitialQvalueEstimateType` for more details.
        supervisedRegressorType : :class:`~dynamic_trading.enums.enums.SupervisedRegressorType`
            Determines what kind of supervised regressor should be used to fit the state-action value function. Refer to
            :class:`~dynamic_trading.enums.enums.SupervisedRegressorType` for more details.
        polynomial_regression_degree : int
            Integer determining the (maximum) polynomial degree considered in case of polynomial regression.
        alpha_ewma : float
            Parameter for exponentially weighted moving average used for defining model averages.

        """

        self._environment = environment

        # available optimizers: ('basinhopping', 'brute', 'differential_evolution', 'dual_annealing', 'shgo', 'local')
        self._optimizerType = optimizerType
        print(f'Using optimizer={self._optimizerType}')

        self._q_value_models = []
        self._set_agent_attributes()

        self._average_across_models = average_across_models
        self._use_best_n_batch = use_best_n_batch
        self._initialQvalueEstimateType = initialQvalueEstimateType
        self._supervisedRegressorType = supervisedRegressorType
        self._polynomial_regression_degree = polynomial_regression_degree

        if alpha_ewma > 1 or alpha_ewma < 0:
            raise NameError(f'Invalid alpha_ewma = {alpha_ewma}: must be 0 <= alpha_ewma <= 1')
        if alpha_ewma == 1:
            self._average_across_models = True
        self._alpha_ewma = alpha_ewma

        self._tol = 10 ** -10  # numerical tolerance for bound conditions requirements

        if self._supervisedRegressorType == SupervisedRegressorType.polynomial_regression:
            self._total_polynomial_optimizations = 0
            self._missing_polynomial_optima = 0

        self._alpha_truncnorm = 0.01

    def policy(self, state: State, eps: float = None):
        """
        Implementation of the agent's policy.

        Parameters
        ----------
        state : State
            State variable.
        eps : float
            If not `None`, parameter for epsilon-greedy policy

        Returns
        -------
        action: Action
            Action performed by the agent given the :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State`.

        """

        if eps is None:
            action = self._greedy_policy(state)
        else:
            action = self._eps_greedy_policy(state, eps)

        if np.abs(state.rescaled_shares + action.rescaled_trade) > 1 + self._tol:
            raise NameError(
                f'Shares went out of bound!! \n  rescaled_shares: {state.rescaled_shares:.2f} \n  rescaled_trade: '
                f'{action.rescaled_trade:.2f}')

        return action

    def q_value(self, state: State, action: Action):
        """
        Implementation of the state-action value function.

        Parameters
        ----------
        state : State
            State variable.
        action : Action
            Action variable.

        Returns
        -------
        qvl : float
            Value of q(s, a).

        """

        qvl = self._q_value_trading(state, action)

        return qvl

    def update_q_value_models(self, q_value_model):
        """
        Appends a new model for the state-action value function to those considered on this agent.

        Parameters
        ----------
        q_value_model : scikit-learn :obj:`RegressorMixin`
            The new model to append.

        """

        self._q_value_models.append(q_value_model)

    def dump_q_value_models(self):
        """
        Service function that dumps all models for the state-action value function.

        """

        for n in range(len(self._q_value_models)):
            q_value_model = self._q_value_models[n]
            dump(q_value_model,
                 os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                 + '/resources/data/supervised_regressors/q%d.joblib' % n)

    def load_q_value_models(self, n_batches: int):
        """
        Service function that loads all models for the state-action value function.

        """

        if self._use_best_n_batch and self._best_n is not None:
            n_batches = self._best_n
        else:
            print(f'Want to use best batch but best_n not yet computed. Possible cause: on-the-fly testing.')

        for n in range(n_batches):
            try:
                q_value_model = load(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    + '/resources/data/supervised_regressors/q%d.joblib' % n)
                self.update_q_value_models(q_value_model)
            except:
                print(f'Trying to load q{n} but it is not fitted. Possible cause: on-the-fly testing.')

    def qvl_from_ravel_input(self, q_value_model_input):
        """
        Service function that evaluates a model for the state-action value function given an input expressed in proper
        array_like format.

        Parameters
        ----------
        q_value_model_input : array_like
            The input to the state-action value function model.

        Returns
        -------
        qvl : float
            Value of q(s, a).

        """
        if self._supervisedRegressorType == SupervisedRegressorType.polynomial_regression:
            poly = instantiate_polynomialFeatures(degree=self._polynomial_regression_degree)

            q_value_model_input = poly.fit_transform(q_value_model_input)
        if self._average_across_models:

            q_value_model = self._q_value_models[0]
            qvl = q_value_model.predict(q_value_model_input)
            alpha_ewma = self._alpha_ewma
            for q_value_model in self._q_value_models[1:]:
                qvl_new = q_value_model.predict(q_value_model_input)
                qvl = alpha_ewma * qvl_new + (1 - alpha_ewma) * qvl

        else:
            q_value_model = self._q_value_models[-1]
            qvl = q_value_model.predict(q_value_model_input)
        return qvl

    def print_proportion_missing_polynomial_optima(self):
        """
        Service function printing the proportion of undefined polynomial optima in case of polynomial regression for the
        state-action value function model.

        """

        if self._total_polynomial_optimizations > 0:
            proportion_missing = self._missing_polynomial_optima / self._total_polynomial_optimizations
        else:
            proportion_missing = -1
        print(f'Total number of polynomial optimizations: {self._total_polynomial_optimizations}')
        print(f'Number of missing polynomial optima: {self._missing_polynomial_optima}')
        print(f'Proportion of missing polynomial optima: {proportion_missing * 100: .2f} %')

    def extract_q_value_model_input_trading(self, state, action):
        """
        Service function that, for a given state and action (as expressed in terms of objects :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State` and
        :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.Action`), extracts the array_like input for a state-action value function model.

        Parameters
        ----------
        state : State
            State variable.
        action : Action
            Action variable.

        Returns
        -------
        q_value_model_input : array_like
            array_like input for a state-action value function model

        """

        state_lst = self._extract_state_lst_trading(state)
        rescaled_trade = action.rescaled_trade
        q_value_model_input = state_lst + [rescaled_trade]
        q_value_model_input = np.array(q_value_model_input, dtype=object).reshape(1, -1)

        return q_value_model_input

    def set_polynomial_regression_degree(self, degree):
        self._polynomial_regression_degree = degree

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

            if self._randomActionType in (RandomActionType.RandomUniform,
                                          RandomActionType.RandomTruncNorm):

                action = self._random_action(state)

            elif self._randomActionType == RandomActionType.GP:

                rescaled_trade = self._environment.agent_GP.policy(factor=state.factor,
                                                                   rescaled_shares=state.rescaled_shares,
                                                                   shares_scale=state.shares_scale,
                                                                   price=state.price)
                action = Action()
                action.set_trading_attributes(rescaled_trade=rescaled_trade, shares_scale=state.shares_scale)

            else:
                raise NameError(f'Invalid randomActionType: {self._randomActionType.value}')

        else:
            rescaled_trade = self._optimize_q_value_trading(state)
            action = Action()
            action.set_trading_attributes(rescaled_trade=rescaled_trade, shares_scale=state.shares_scale)

        return action

    def _random_action_trading(self, state):

        lower_bound, upper_bound = self._get_action_bounds_trading(state)

        if self._randomActionType == RandomActionType.RandomUniform:

            rescaled_trade = lower_bound + (upper_bound - lower_bound) * np.random.rand()

        else:

            # TODO: We are implying a preference for randomActionType.RandomTruncNorm, in that this is chosen
            #  by default even if the initialization is set to GP. Make a better structuring of this part.

            loc = self._get_trade_loc(lower_bound, upper_bound)
            scale = self._get_trade_scale(lower_bound, upper_bound, self._alpha_truncnorm)

            rescaled_trade = truncnorm.rvs(a=lower_bound, b=upper_bound, loc=loc, scale=scale)

        action = Action()
        action.set_trading_attributes(rescaled_trade=rescaled_trade, shares_scale=state.shares_scale)

        return action

    @staticmethod
    def _get_trade_loc(lower_bound, upper_bound, return_mean=False):

        if return_mean:
            loc = 0.5 * (lower_bound + upper_bound)
        else:
            loc = 0.

        return loc

    @staticmethod
    def _get_trade_scale(lower_bound, upper_bound, alpha):

        scale = alpha * (upper_bound - lower_bound)

        return scale

    def _get_action_bounds_trading(self, state: State):

        actionSpace = ActionSpace(state, self._strategyType)
        actionSpace.set_trading_actions_interval()
        lower_bound, upper_bound = actionSpace.actions_interval

        return lower_bound, upper_bound

    def _q_value_trading(self, state: State, action: Action):

        qvl = None

        if len(self._q_value_models) == 0:
            if self._initialQvalueEstimateType == InitialQvalueEstimateType.random:
                qvl = np.random.randn()
            elif self._initialQvalueEstimateType == InitialQvalueEstimateType.zero:
                qvl = 0.
        else:
            q_value_model_input = self.extract_q_value_model_input_trading(state, action)
            qvl = self.qvl_from_ravel_input(q_value_model_input)

            if np.ndim(qvl) > 0:
                qvl = qvl.item()

        return qvl

    def _optimize_q_value_trading(self, state: State):

        lower_bound, upper_bound = self._get_action_bounds_trading(state)

        if self._supervisedRegressorType == SupervisedRegressorType.polynomial_regression:

            x_optimal = self._optimize_polynomial_q_value_trading(state, lower_bound, upper_bound)

        else:

            x_optimal = self._optimize_general_q_value_trading(state, lower_bound, upper_bound)

        return x_optimal

    def _optimize_polynomial_q_value_trading(self, state, lower_bound, upper_bound):

        # todo: this can be heavily optimized

        # compute complete list of polynomial coefficients
        q_value_model = self._q_value_models[0]
        coef = q_value_model.coef_
        for q_value_model in self._q_value_models[1:]:
            coef = 0.5 * (coef + q_value_model.coef_)

        # compute complete list of polynomial variables names
        poly = instantiate_polynomialFeatures(self._polynomial_regression_degree)
        fake_action = Action()
        fake_action.set_trading_attributes(rescaled_trade=1)
        q_value_model_input = self.extract_q_value_model_input_trading(state, fake_action)
        action_name = f'x{q_value_model_input.shape[1] - 1}'
        poly.fit(q_value_model_input)
        q_value_model_input = poly.fit_transform(q_value_model_input)
        variables_names = poly.get_feature_names_out()
        for i in range(len(variables_names)):
            if action_name not in variables_names[i]:
                variables_names[i] += f' {action_name}^0'
            if action_name in variables_names[i] and f'{action_name}^' not in variables_names[i]:
                variables_names[i] = variables_names[i].replace(action_name, f'{action_name}^1')

        # aggregate all terms multiplying action^d
        aggregate_coef = []
        for d in range(self._polynomial_regression_degree + 1):
            # get positions of variables names that are multiplying action^d
            positions_d = [i for i in range(len(variables_names)) if f'{action_name}^{d}' in variables_names[i]]

            # get coefficient of action^d
            coef_d = np.sum(coef[positions_d] * q_value_model_input[0, positions_d])

            aggregate_coef.append(coef_d)

        aggregate_coef = - np.array(aggregate_coef)  # we want to find the maximum

        x_optimal, flag_error = find_polynomial_minimum(coef=aggregate_coef, bounds=(lower_bound, upper_bound))

        self._total_polynomial_optimizations += 1
        if flag_error:
            self._missing_polynomial_optima += 1

        return x_optimal

    def _optimize_general_q_value_trading(self, state, lower_bound, upper_bound):

        x0 = self._get_trade_loc(lower_bound, upper_bound)
        bounds = [(lower_bound, upper_bound)]

        x_optim_when_error = truncnorm.rvs(a=lower_bound, b=upper_bound, loc=x0,
                                           scale=self._alpha_truncnorm * (upper_bound - lower_bound))

        def func(rescaled_trade):

            action = Action()
            action.set_trading_attributes(rescaled_trade=rescaled_trade)

            qvl = self._q_value_trading(state, action)

            return - qvl

        if self._optimizerType == OptimizerType.basinhopping:
            res = basinhopping(func=func, x0=x0)
            x_optimal = res.x[0]

        elif self._optimizerType == OptimizerType.brute:

            Ns = 4 * int(upper_bound * state.shares_scale - lower_bound * state.shares_scale)
            xx = np.linspace(lower_bound, upper_bound, Ns)
            ff = np.array([func(x) for x in xx])
            x_optimal = xx[np.argmin(ff)]

        elif self._optimizerType == OptimizerType.differential_evolution:
            res = differential_evolution(func=func, bounds=bounds)
            x_optimal = res.x[0]

        elif self._optimizerType == OptimizerType.dual_annealing:
            res = dual_annealing(func=func, bounds=bounds)
            x_optimal = res.x[0]

        elif self._optimizerType == OptimizerType.shgo:
            res = shgo(func=func, bounds=bounds)
            x_optimal = res.x[0]

        elif self._optimizerType == OptimizerType.local:
            res = minimize(fun=func, bounds=bounds, x0=np.array(x0))
            x_optimal = res.x[0]

        else:
            raise NameError(f'Invalid optimizerType: {self._optimizerType}')

        if x_optimal == lower_bound or x_optimal == upper_bound:
            x_optimal = x_optim_when_error

        return x_optimal

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
            print(f'action_GP is not set')

        # fill list
        state_lst[0] = rescaled_shares

        if self._environment.state_factor:
            state_lst[state_shape['factor']] = factor
        if self._environment.state_ttm:
            state_lst[state_shape['ttm']] = ttm
        if self._environment.state_price:
            state_lst[state_shape['price']] = price
        if self._environment.state_pnl:
            state_lst[state_shape['pnl']] = pnl
        if self._environment.state_average_past_pnl:
            state_lst[state_shape['average_past_pnl']] = average_past_pnl
        if self._environment.state_GP_action:
            state_lst[state_shape['action_GP']] = action_GP

        return state_lst

    def _set_agent_attributes(self):

        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += '/resources/data/data_source/settings.csv'
        df_trad_params = pd.read_csv(filename, index_col=0)

        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += '/resources/data/data_source/market_data/commodities-summary-statistics.xlsx'
        df_lam_kappa = pd.read_excel(filename, index_col=0, sheet_name='Simplified contract multiplier')
        df_lam_kappa = df_lam_kappa.loc[self._environment.market.ticker]  # TODO: should it be self.environment.ticker?

        gamma = float(df_trad_params.loc['gamma'][0])
        kappa = float(df_lam_kappa.loc['kappa'])

        randomActionType =\
            RandomActionType(str(df_trad_params.loc['randomActionType'][0]))
        strategyType = StrategyType(df_trad_params.loc['strategyType'][0])

        self._gamma = gamma
        self._kappa = kappa
        self._randomActionType = randomActionType
        self._strategyType = strategyType

        self._push_trading_parameters_to_environment()

        if self._randomActionType == RandomActionType.GP:
            self._environment._observe_GP = True
            self._environment.instantiate_market_benchmark_and_agent_GP()

        self._best_n = None
        try:
            self._best_n = int(load(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                                    + '/resources/data/data_tmp/best_n.joblib'))
        except:
            print(f'Notice: agent is not yet trained, therefore it is impossible to set best_n')

    def _push_trading_parameters_to_environment(self):
        self._environment.set_trading_parameters(self._gamma, self._kappa)

    @property
    def supervisedRegressorType(self):
        """
        :class:`~dynamic_trading.enums.enums.SupervisedRegressorType` considered by the agent.

        """
        return self._supervisedRegressorType

    @property
    def kappa(self):
        """
        Risk-aversion parameter.

        """
        return self._kappa

    @property
    def gamma(self):
        """
        Cumulative future rewards discount factor.

        """
        return self._gamma
