import numpy as np
import pandas as pd
import os

from dynamic_trading.benchmark_agents.agents import AgentGP
from dynamic_trading.enums.enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType
from dynamic_trading.reinforcement_learning_utils.state_action_utils import Action, State
from dynamic_trading.market_utils.market import Market, instantiate_market


class Environment:
    """
    Class defining a reinforcement learning environment.

    """

    def __init__(self, market: Market, random_initial_state: bool = True):
        """
        Class constructor.

        Parameters
        ----------
        market : :class:`~dynamic_trading.market_utils.market.Market`
            The market on which the agent is operating. Refer to the documentation of
            :class:`~dynamic_trading.market_utils.market.Market` for more details.
        random_initial_state : bool
            Boolean determining whether the initial state :math:`s_0` is selected randomly.

        """

        self._kappa = None
        self._gamma = None
        self._agent_GP = None
        self._market_benchmark = None
        self._market = market
        self._random_initial_state = random_initial_state
        self._set_attributes()

    def instantiate_initial_state_trading(self, n: int, j: int, shares_scale: float = 1):
        """
        Instantiates the state at time :math:`t=0` for the j-th episode on the n-th batch.

        Parameters
        ----------
        n : int
            Batch iteration.
        j : int
            Episode identifier.
        shares_scale : float
            Factor for rescaling the shares :math:`M`.

        Returns
        -------
        state : :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State`
            Initial state. Refer to :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State` for
            more details.

        """

        state = State(environment=self)

        if self._random_initial_state:
            rescaled_shares = -1 + 2*np.random.rand()
            pnl, factor, price, average_past_pnl = self._get_market_simulations_training(n=n, j=j, t=-1)
        else:
            rescaled_shares = 0.
            pnl, factor, price, average_past_pnl = self._get_market_simulations_training(n=n, j=j, t=0)

        ttm = self._t_

        if self._observe_GP:
            rescaled_trade_GP = self._agent_GP.policy(factor=factor,
                                                      rescaled_shares=rescaled_shares,
                                                      shares_scale=shares_scale,
                                                      price=price)
            action_GP = Action()
            action_GP.set_trading_attributes(rescaled_trade=rescaled_trade_GP,
                                             shares_scale=shares_scale)
        else:
            action_GP = None

        state.set_trading_attributes(factor=factor,
                                     rescaled_shares=rescaled_shares,
                                     shares_scale=shares_scale,
                                     price=price,
                                     pnl=pnl,
                                     average_past_pnl=average_past_pnl,
                                     action_GP=action_GP,
                                     ttm=ttm)

        return state

    def instantiate_market_benchmark_and_agent_GP(self):
        """
        A GP agent operating in the environment. See :class:`~dynamic_trading.benchmark_agents.agents.AgentGP` for more
        details.

        """

        self._market_benchmark = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear,
                                                    factorDynamicsType=FactorDynamicsType.AR,
                                                    ticker=self._ticker,
                                                    riskDriverType=RiskDriverType.PnL)
        self._agent_GP = AgentGP(market=self._market_benchmark)

    def compute_reward_and_next_state(self, state: State, action: Action, n: int, j: int, t: int,
                                      predict_pnl_for_reward: bool):
        """
        Computes reward and next state for a given state and action.

        Parameters
        ----------
        state : :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State`
            State variable currently observed. Refer
            to :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State` for more details.
        action : :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.Action`
            Action variable currently selected. Refer
            to :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.Action` for more details.
        n : int
            Batch iteration.
        j : int
            Episode iteration
        t : int
            Time iteration.
        predict_pnl_for_reward : bool
            Boolean determining whether the PnL is predicted in terms of the factor in the reward definition. If
            ``False``, then the reward is computed as
            :math:`R_{t+1} = \gamma(n^{'}_t x_{t+1} - 0.5\kappa n^{'}_t\Sigma n_t)-c(\Delta n_t)`; if ``True``, it is
            computed as
            :math:`R_{t+1} = \gamma(n^{'}_t g(f_t) - 0.5\kappa n^{'}_t\Sigma n_t)-c(\Delta n_t)` where
            :math:`g(f_t)=E[x_{t+1}|f_t]`.

        Returns
        -------
        next_state : :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State`
            Next state. Refer to :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State` for
            more details.
        reward : float
            Reward for taking :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.Action` when
            observing :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State`.

        """

        next_state = self._compute_next_state(state=state, action=action, n=n, j=j, t=t)
        reward = self._compute_reward(state=state, action=action, next_state=next_state,
                                      predict_pnl_for_reward=predict_pnl_for_reward)

        return next_state, reward

    def compute_trading_cost(self, action, sig2):
        """
        Computes the trading cost for a given action.

        Parameters
        ----------
        action : :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.Action`
            Action implemented. Refer
            to :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.Action` for more details.
        sig2 : float
            P\&L variance.

        Returns
        -------
        trading_cost : float
            The trading cost :math:`0.5 \lambda \Delta n^{'}_{t}\Sigma \Delta n_{t}`.

        """

        trade = action.trade
        trading_cost = 0.5 * trade * self._lam * sig2 * trade

        return trading_cost

    def compute_trading_risk(self, state, sig2):
        """
        Computes the trading risk for a given state.

        Parameters
        ----------
        state : :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State`
            State observed. Refer to :class:`~dynamic_trading.reinforcement_learning_utils.state_action_utils.State` for
            more details.
        sig2 :
            P\&L variance.

        Returns
        -------
        trading_risk : float
            The trading risk :math:`0.5 \kappa n^{'}_t\Sigma n_t`.

        """

        shares = state.shares
        trading_risk = 0.5 * shares * self._kappa * sig2 * shares

        return trading_risk

    def set_trading_parameters(self, gamma, kappa):
        """
        Service function that overwrites the environment's gamma and kappa parameter as being pushed from outside.

        Parameters
        ----------
        gamma : float
            The factor used to discount the cumulative future rewards target :math:`\sum_{k\ge 0} \gamma^k R_{t+k+1}` in
            the dynamic programming problem defining the optimal allocation strategy.
        kappa : float
            The risk aversion parameter :math:`\kappa` appearing in the risk definition
            :math:`0.5 \kappa n^{'}_t\Sigma n_t`.

        """
        self._gamma = gamma
        self._kappa = kappa

    def _compute_reward(self, state: State, action: Action, next_state: State, predict_pnl_for_reward: bool):

        reward = self._compute_trading_reward(state, action, next_state, predict_pnl_for_reward)

        return reward

    def _compute_next_state(self, state: State, action: Action, n: int, j: int, t: int):

        next_state = self._compute_trading_next_state(state, action, n, j, t)

        return next_state

    def _compute_trading_reward(self, state, action, next_state, predict_pnl_for_reward):

        p_t = state.price
        f_t = state.factor

        if predict_pnl_for_reward:
            pnl_tp1 = self._market.next_step_pnl(factor=f_t, price=p_t)
        else:
            p_tp1 = next_state.price
            pnl_tp1 = p_tp1 - p_t

        a_t = action
        n_t = next_state.shares
        sig2 = self._market.next_step_sig2(factor=f_t, price=p_t)
        cost = self.compute_trading_cost(a_t, sig2)

        reward = self._gamma * (n_t * pnl_tp1 - 0.5 * self._kappa * n_t * sig2 * n_t) - cost

        return reward

    def _compute_trading_next_state(self, state: State, action: Action, n: int, j: int, t: int):

        rescaled_shares = state.rescaled_shares
        shares_scale = state.shares_scale

        next_rescaled_shares = rescaled_shares + action.rescaled_trade

        pnl, factor, price, average_past_pnl = self._get_market_simulations_training(n=n, j=j, t=t)
        next_factor = factor
        next_price = price
        next_pnl = pnl
        next_average_past_pnl = average_past_pnl
        next_ttm = self._t_ - t

        if self._observe_GP:
            next_rescaled_shares_GP = self._agent_GP.policy(factor=factor,
                                                            rescaled_shares=rescaled_shares,
                                                            shares_scale=shares_scale,
                                                            price=price)
            next_action_GP = Action()
            next_action_GP.set_trading_attributes(rescaled_trade=next_rescaled_shares_GP,
                                                  shares_scale=shares_scale)
        else:
            next_action_GP = None

        next_state = State(environment=self)
        next_state.set_trading_attributes(factor=next_factor,
                                          rescaled_shares=next_rescaled_shares,
                                          shares_scale=shares_scale,
                                          price=next_price,
                                          pnl=next_pnl,
                                          average_past_pnl=next_average_past_pnl,
                                          action_GP=next_action_GP,
                                          ttm=next_ttm)

        return next_state

    def _get_market_simulations_training(self, n: int, j: int, t: int):

        return (self._market.simulations_training[n]['pnl'][j, t],
                self._market.simulations_training[n]['factor'][j, t],
                self._market.simulations_training[n]['price'][j, t],
                self._market.simulations_training[n]['average_past_pnl'][j, t])

    def _set_attributes(self):

        self._set_trading_attributes()

    def _set_trading_attributes(self):

        # input file
        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += '/resources/data/data_source/settings.csv'
        df_trad_params = pd.read_csv(filename, index_col=0)

        # ticker
        self._ticker = self._market.ticker

        # t_
        self._t_ = int(df_trad_params.loc['t_'][0])

        # lam
        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += '/resources/data/data_source/market_data/commodities-summary-statistics.xlsx'
        df_lam_kappa = pd.read_excel(filename, index_col=0, sheet_name='Simplified contract multiplier')
        df_lam_kappa = df_lam_kappa.loc[self._ticker]
        lam = float(df_lam_kappa.loc['lam'])
        self._lam = lam

        # state_factor
        if str(df_trad_params.loc['state_factor'][0]) == 'Yes':
            state_factor = True
        elif str(df_trad_params.loc['state_factor'][0]) == 'No':
            state_factor = False
        else:
            raise NameError('state_factor in settings file must be either Yes or No')
        self._state_factor = state_factor

        # state_GP_action and observe_GP
        if str(df_trad_params.loc['state_GP_action'][0]) == 'Yes':
            state_GP_action = True
            observe_GP = True
        elif str(df_trad_params.loc['state_GP_action'][0]) == 'No':
            state_GP_action = False
            observe_GP = False
        else:
            raise NameError('state_GP_action in settings file must be either Yes or No')
        self._state_GP_action = state_GP_action
        self._observe_GP = observe_GP
        if self._observe_GP:
            self.instantiate_market_benchmark_and_agent_GP()

        # state_ttm
        if str(df_trad_params.loc['state_ttm'][0]) == 'Yes':
            state_ttm = True
            self._add_absorbing_state = True
        elif str(df_trad_params.loc['state_ttm'][0]) == 'No':
            state_ttm = False
            self._add_absorbing_state = False
        else:
            raise NameError('state_ttm in settings file must be either Yes or No')
        self._state_ttm = state_ttm

        # state_price
        if str(df_trad_params.loc['state_price'][0]) == 'Yes':
            state_price = True
        elif str(df_trad_params.loc['state_price'][0]) == 'No':
            state_price = False
        else:
            raise NameError('state_price in settings file must be either Yes or No')
        self._state_price = state_price

        # state_pnl
        if str(df_trad_params.loc['state_pnl'][0]) == 'Yes':
            state_pnl = True
        elif str(df_trad_params.loc['state_pnl'][0]) == 'No':
            state_pnl = False
        else:
            raise NameError('state_pnl in settings file must be either Yes or No')
        self._state_pnl = state_pnl

        # state_average_past_pnl
        if str(df_trad_params.loc['state_average_past_pnl'][0]) == 'Yes':
            state_average_past_pnl = True
        elif str(df_trad_params.loc['state_average_past_pnl'][0]) == 'No':
            state_average_past_pnl = False
        else:
            raise NameError('state_average_past_pnl in settings file must be either Yes or No')
        self._state_average_past_pnl = state_average_past_pnl

        # state_shape
        # Define the structure of the state variable depending on the values assigned to
        # self.state_factor, self.state_ttm, self.state_pnl, self.state_GP_action
        # The most complete state is given by
        # (rescaled_shares, factor, ttm, price, pnl, average_past_pnl, action_GP)

        bool_values = [self._state_factor,
                       self._state_ttm,
                       self._state_price,
                       self._state_pnl,
                       self._state_average_past_pnl,
                       self._state_GP_action]
        str_values = ['factor',
                      'ttm',
                      'price',
                      'pnl',
                      'average_past_pnl',
                      'action_GP']
        binary_values = [int(bool_value) for bool_value in bool_values]

        self._state_shape = {'rescaled_shares': 0}

        count_prev = 0
        for i in range(len(bool_values)):
            count_curr = count_prev + binary_values[i]
            if count_curr != count_prev:
                self._state_shape[str_values[i]] = count_curr
                count_prev = count_curr

    @property
    def add_absorbing_state(self):
        """
        Boolean determining whether an absorbing state should be added at the end of each episode. Particularly used
        when the time-to-maturity is in the state variable. In this case, at the end of each episode an instance of the
        ultimate state where time-to-maturity is zero and a zero reward is assigned.

        """
        return self._add_absorbing_state

    @property
    def agent_GP(self):
        """
        A GP agent operating in the environment. Particularly used in case the RL agent is benchmarking a GP agent.

        """
        return self._agent_GP

    @property
    def state_factor(self):
        """
        Determines whether the factor :math:`f_t` is included in the state variable.

        """
        return self._state_factor

    @property
    def state_ttm(self):
        """
        Determines whether the time-to-maturity :math:`T-t` is included in the state variable.

        """
        return self._state_ttm

    @property
    def state_price(self):
        """
        Determines whether the security's price :math:`p_t` is included in the state variable.

        """
        return self._state_price

    @property
    def state_pnl(self):
        """
        Determines whether the security's P\&L :math:`p_t-p_{t-1}` is included in the state variable.

        """
        return self._state_pnl

    @property
    def state_average_past_pnl(self):
        """
        Determines whether an average of the security's past P\&Ls is included in the state variable.

        """
        return self._state_average_past_pnl

    @property
    def state_GP_action(self):
        """
        Determines whether GP agent action is included in the state variable.

        """
        return self._state_GP_action

    @property
    def market(self):
        """
        The market in which the agent is operating. Refer to :class:`~dynamic_trading.market_utils.market.Market` for
        more details.

        """
        return self._market

    @property
    def gamma(self):
        """
        The factor used to discount the cumulative future rewards target :math:`\sum_{k\ge 0} \gamma^k R_{t+k+1}` in the
        dynamic programming problem defining the optimal allocation strategy.

        """
        return self._gamma

    @property
    def observe_GP(self):
        """
        Boolean determining whether a GP agent is observed in the environment.

        """
        return self._observe_GP

    @property
    def state_shape(self):
        """
        Useful dict that helps to determine the state variable content when unravelling it as input to state-action
        value function model. This dict allows to easily generalize the state variable to include user specified
        variables. See more in the README.

        """
        return self._state_shape

    @property
    def t_(self):
        """
        Length :math:`T` of the episodes.

        """
        return self._t_
