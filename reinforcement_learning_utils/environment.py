import pandas as pd
import os

from benchmark_agents.agents import AgentGP
from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType
from reinforcement_learning_utils.state_action_utils import Action, State
from market_utils.market import Market, instantiate_market


class Environment:

    def __init__(self, market: Market):

        self.kappa = None
        self.gamma = None
        self.agent_GP = None
        self.market_benchmark = None
        self.market = market
        self._set_attributes()

    def compute_reward_and_next_state(self, state: State, action: Action, n: int, j: int, t: int, predict_pnl_for_reward: bool):

        next_state = self._compute_next_state(state=state, action=action, n=n, j=j, t=t)
        reward = self._compute_reward(state=state, action=action, next_state=next_state,
                                      predict_pnl_for_reward=predict_pnl_for_reward)

        return next_state, reward

    def instantiate_initial_state_trading(self, n: int, j: int, shares_scale: float = 1):

        state = State(environment=self)

        rescaled_shares = 0.
        other_observable = 0.

        pnl, factor, price = self._get_market_simulation_trading(n=n, j=j, t=0)
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

        _, factor, price = self._get_market_simulation_trading(n=n, j=j, t=t)
        next_factor = factor
        next_other_observable = 0.
        next_price = price
        next_pnl = next_price - state.price
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
                self.market.simulations_trading[n]['price'][j, t])

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

        # factor_in_state
        if str(df_trad_params.loc['factor_in_state'][0]) == 'Yes':
            factor_in_state = True
        elif str(df_trad_params.loc['factor_in_state'][0]) == 'No':
            factor_in_state = False
        else:
            raise NameError('factor_in_state in settings file must be either Yes or No')
        self.factor_in_state = factor_in_state

        # GP_action_in_state and observe_GP
        if str(df_trad_params.loc['GP_action_in_state'][0]) == 'Yes':
            GP_action_in_state = True
            observe_GP = True
        elif str(df_trad_params.loc['GP_action_in_state'][0]) == 'No':
            GP_action_in_state = False
            observe_GP = False
        else:
            raise NameError('GP_action_in_state in settings file must be either Yes or No')
        self.GP_action_in_state = GP_action_in_state
        self.observe_GP = observe_GP
        if self.observe_GP:
            self.instantiate_market_benchmark_and_agent_GP()

        # ttm_in_state
        if str(df_trad_params.loc['ttm_in_state'][0]) == 'Yes':
            ttm_in_state = True
        elif str(df_trad_params.loc['GP_action_in_state'][0]) == 'No':
            ttm_in_state = False
        else:
            raise NameError('ttm_in_state in settings file must be either Yes or No')
        self.ttm_in_state = ttm_in_state

        # pnl_in_state
        if str(df_trad_params.loc['price_in_state'][0]) == 'Yes':
            price_in_state = True
        elif str(df_trad_params.loc['price_in_state'][0]) == 'No':
            price_in_state = False
        else:
            raise NameError('price_in_state in settings file must be either Yes or No')
        self.price_in_state = price_in_state

        # pnl_in_state
        if str(df_trad_params.loc['pnl_in_state'][0]) == 'Yes':
            pnl_in_state = True
        elif str(df_trad_params.loc['pnl_in_state'][0]) == 'No':
            pnl_in_state = False
        else:
            raise NameError('pnl_in_state in settings file must be either Yes or No')
        self.pnl_in_state = pnl_in_state

        # state_shape
        # Define the structure of the state variable depending on the values assigned to
        # self.factor_in_state, self.ttm_in_state, self.pnl_in_state, self.GP_action_in_state
        # The most complete state is given by
        # (rescaled_shares, factor, ttm, price, action_GP)

        bool_values = [self.factor_in_state,
                       self.ttm_in_state,
                       self.price_in_state,
                       self.pnl_in_state,
                       self.GP_action_in_state]
        str_values = ['factor',
                      'ttm',
                      'price',
                      'pnl',
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
