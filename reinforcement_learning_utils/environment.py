import numpy as np
import pandas as pd
import os

from benchmark_agents.agents import AgentGP
from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType
from reinforcement_learning_utils.state_action_utils import Action, State
from market_utils.market import Market, instantiate_market


class Environment:

    def __init__(self, market: Market, random_initial_state: bool = True):

        self.kappa = None
        self.gamma = None
        self.agent_GP = None
        self.market_benchmark = None
        self.market = market
        self._random_initial_state = random_initial_state
        self._set_attributes()

    def compute_reward_and_next_state(self, state: State, action: Action, n: int, j: int, t: int, predict_pnl_for_reward: bool):

        next_state = self._compute_next_state(state=state, action=action, n=n, j=j, t=t)
        reward = self._compute_reward(state=state, action=action, next_state=next_state,
                                      predict_pnl_for_reward=predict_pnl_for_reward)

        return next_state, reward

    def instantiate_initial_state_trading(self, n: int, j: int, shares_scale: float = 1):

        state = State(environment=self)

        if self._random_initial_state:
            rescaled_shares = -1 + 2*np.random.rand()
            other_observable = 0.
            pnl, factor, price, average_past_pnl = self._get_market_simulation_trading(n=n, j=j, t=-1)
        else:
            rescaled_shares = 0.
            other_observable = 0.
            pnl, factor, price, average_past_pnl = self._get_market_simulation_trading(n=n, j=j, t=0)

        ttm = self.t_

        if self.observe_GP:
            rescaled_trade_GP = self.agent_GP.policy(factor=factor,
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
                                     other_observable=other_observable,
                                     shares_scale=shares_scale,
                                     price=price,
                                     pnl=pnl,
                                     average_past_pnl=average_past_pnl,
                                     action_GP=action_GP,
                                     ttm=ttm)

        return state

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
            pnl_tp1 = self.market.next_step_pnl(factor=f_t, price=p_t)
        else:
            p_tp1 = next_state.price
            pnl_tp1 = p_tp1 - p_t

        a_t = action
        n_t = next_state.shares
        sig2 = self.market.next_step_sig2(factor=f_t, price=p_t)
        cost = self.compute_trading_cost(a_t, sig2)

        reward = self.gamma * (n_t * pnl_tp1 - 0.5 * self.kappa * n_t * sig2 * n_t) - cost

        return reward

    def _compute_trading_next_state(self, state: State, action: Action, n: int, j: int, t: int):

        rescaled_shares = state.rescaled_shares
        shares_scale = state.shares_scale

        next_rescaled_shares = rescaled_shares + action.rescaled_trade

        pnl, factor, price, average_past_pnl = self._get_market_simulation_trading(n=n, j=j, t=t)
        next_factor = factor
        next_other_observable = 0.
        next_price = price
        next_pnl = pnl
        next_average_past_pnl = average_past_pnl
        next_ttm = self.t_ - t

        if self.observe_GP:
            next_rescaled_shares_GP = self.agent_GP.policy(factor=factor,
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
                                          other_observable=next_other_observable,
                                          shares_scale=shares_scale,
                                          price=next_price,
                                          pnl=next_pnl,
                                          average_past_pnl=next_average_past_pnl,
                                          action_GP=next_action_GP,
                                          ttm=next_ttm)

        return next_state

    def compute_trading_cost(self, action, sig2):

        trade = action.trade

        return 0.5 * trade * self.lam * sig2 * trade

    def compute_trading_risk(self, state, sig2):

        shares = state.shares

        return 0.5 * shares * self.kappa * sig2 * shares

    def _get_market_simulation_trading(self, n: int, j: int, t: int):

        return (self.market.simulations_trading[n]['pnl'][j, t],
                self.market.simulations_trading[n]['factor'][j, t],
                self.market.simulations_trading[n]['price'][j, t],
                self.market.simulations_trading[n]['average_past_pnl'][j, t])

    def _set_attributes(self):

        self._set_trading_attributes()

    def _set_trading_attributes(self):

        # input file
        filename = os.path.dirname(os.path.dirname(__file__)) + '/data/data_source/settings/settings.csv'
        df_trad_params = pd.read_csv(filename, index_col=0)

        # ticker
        self.ticker = self.market.ticker

        # t_
        self.t_ = int(df_trad_params.loc['t_'][0])

        # lam
        filename = os.path.dirname(os.path.dirname(__file__)) +\
                   '/data/data_source/market_data/commodities-summary-statistics.xlsx'
        df_lam_kappa = pd.read_excel(filename, index_col=0, sheet_name='Simplified contract multiplier')
        df_lam_kappa = df_lam_kappa.loc[self.ticker]
        lam = float(df_lam_kappa.loc['lam'])
        self.lam = lam

        # state_factor
        if str(df_trad_params.loc['state_factor'][0]) == 'Yes':
            state_factor = True
        elif str(df_trad_params.loc['state_factor'][0]) == 'No':
            state_factor = False
        else:
            raise NameError('state_factor in settings file must be either Yes or No')
        self.state_factor = state_factor

        # state_GP_action and observe_GP
        if str(df_trad_params.loc['state_GP_action'][0]) == 'Yes':
            state_GP_action = True
            observe_GP = True
        elif str(df_trad_params.loc['state_GP_action'][0]) == 'No':
            state_GP_action = False
            observe_GP = False
        else:
            raise NameError('state_GP_action in settings file must be either Yes or No')
        self.state_GP_action = state_GP_action
        self.observe_GP = observe_GP
        if self.observe_GP:
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
        self.state_ttm = state_ttm

        # state_price
        if str(df_trad_params.loc['state_price'][0]) == 'Yes':
            state_price = True
        elif str(df_trad_params.loc['state_price'][0]) == 'No':
            state_price = False
        else:
            raise NameError('state_price in settings file must be either Yes or No')
        self.state_price = state_price

        # state_pnl
        if str(df_trad_params.loc['state_pnl'][0]) == 'Yes':
            state_pnl = True
        elif str(df_trad_params.loc['state_pnl'][0]) == 'No':
            state_pnl = False
        else:
            raise NameError('state_pnl in settings file must be either Yes or No')
        self.state_pnl = state_pnl

        # state_average_past_pnl
        if str(df_trad_params.loc['state_average_past_pnl'][0]) == 'Yes':
            state_average_past_pnl = True
        elif str(df_trad_params.loc['state_average_past_pnl'][0]) == 'No':
            state_average_past_pnl = False
        else:
            raise NameError('state_average_past_pnl in settings file must be either Yes or No')
        self.state_average_past_pnl = state_average_past_pnl

        # state_shape
        # Define the structure of the state variable depending on the values assigned to
        # self.state_factor, self.state_ttm, self.state_pnl, self.state_GP_action
        # The most complete state is given by
        # (rescaled_shares, factor, ttm, price, pnl, average_past_pnl, action_GP)

        bool_values = [self.state_factor,
                       self.state_ttm,
                       self.state_price,
                       self.state_pnl,
                       self.state_average_past_pnl,
                       self.state_GP_action]
        str_values = ['factor',
                      'ttm',
                      'price',
                      'pnl',
                      'average_past_pnl',
                      'action_GP']
        binary_values = [int(bool_value) for bool_value in bool_values]

        self.state_shape = {'rescaled_shares': 0}

        count_prev = 0
        for i in range(len(bool_values)):
            count_curr = count_prev + binary_values[i]
            if count_curr != count_prev:
                self.state_shape[str_values[i]] = count_curr
                count_prev = count_curr

    def instantiate_market_benchmark_and_agent_GP(self):
        self.market_benchmark = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear,
                                                   factorDynamicsType=FactorDynamicsType.AR,
                                                   ticker=self.ticker,
                                                   riskDriverType=RiskDriverType.PnL)
        self.agent_GP = AgentGP(market=self.market_benchmark)

    def get_trading_parameters_from_agent(self, gamma: float, kappa: float):

        self.gamma = gamma
        self.kappa = kappa

    @property
    def add_absorbing_state(self):
        return self._add_absorbing_state
