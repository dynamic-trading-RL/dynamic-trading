import pandas as pd
import os

from reinforcement_learning_utils.state_action_utils import Action, State
from market_utils.market import Market


class Environment:

    def __init__(self, market: Market):

        self.market = market
        self._set_attributes()

    def compute_reward_and_next_state(self, state: State, action: Action, n: int, j: int, t: int):

        reward_RL = self._compute_reward(state=state, action=action)
        next_state = self._compute_next_state(state=state, action=action, n=n, j=j, t=t)

        return reward_RL, next_state

    def instantiate_initial_state_trading(self, n: int, j: int, shares_scale: float = 1):

        current_rescaled_shares = 0.
        current_other_observable = 0.

        state = State()
        pnl, factor, price = self._get_market_simulation_trading(n=n, j=j, t=0)
        state.set_trading_attributes(current_factor=factor,
                                     current_rescaled_shares=current_rescaled_shares,
                                     current_other_observable=current_other_observable,
                                     shares_scale=shares_scale,
                                     current_price=price)

        return state

    def _compute_reward(self, state: State, action: Action):

        reward = self._compute_trading_reward(state, action)

        return reward

    def _compute_next_state(self, state: State, action: Action, n: int, j: int, t: int):

        next_state = self._compute_trading_next_state(state, action, n, j, t)

        return next_state

    def _compute_trading_reward(self, state, action):

        current_shares = state.current_shares
        current_factor = state.current_factor
        current_price = state.current_price

        pnl = self.market.next_step_pnl(factor=current_factor, price=current_price)
        sig2 = self.market.next_step_sig2(factor=current_factor, price=current_price)

        cost = self.compute_trading_cost(action, sig2)

        reward = self.gamma * (current_shares * pnl - 0.5*self.kappa*current_shares*sig2*current_shares) - cost

        return reward

    def _compute_trading_next_state(self, state: State, action: Action, n: int, j: int, t: int):

        current_rescaled_shares = state.current_rescaled_shares
        shares_scale = state.shares_scale

        next_rescaled_shares = current_rescaled_shares + action.rescaled_trade

        _, factor, price = self._get_market_simulation_trading(n=n, j=j, t=t)
        next_factor = factor
        next_other_observable = 0.
        next_price = price

        next_state = State()
        next_state.set_trading_attributes(current_factor=next_factor,
                                          current_rescaled_shares=next_rescaled_shares,
                                          current_other_observable=next_other_observable,
                                          shares_scale=shares_scale,
                                          current_price=next_price)

        return next_state

    def compute_trading_cost(self, action, sig2):

        trade = action.trade

        return 0.5 * trade * self.lam * sig2 * trade

    def compute_trading_risk(self, state, sig2):

        current_shares = state.current_shares

        return 0.5 * current_shares * self.kappa * sig2 * current_shares

    def _get_market_simulation_trading(self, n: int, j: int, t: int):

        return (self.market.simulations_trading[n]['pnl'][j, t],
                self.market.simulations_trading[n]['factor'][j, t],
                self.market.simulations_trading[n]['price'][j, t])

    def _set_attributes(self):

        self._set_trading_attributes()

    def _set_trading_attributes(self):

        ticker = self.market.ticker
        filename = os.path.dirname(os.path.dirname(__file__)) +\
                   '/data/data_source/settings/settings.csv'
        df_trad_params = pd.read_csv(filename, index_col=0)
        lam = float(df_trad_params.loc['lam'][0])

        self.lam = lam

        self.factorType = self.market.factorType

    def _get_trading_parameters_from_agent(self, gamma: float, kappa: float):

        self.gamma = gamma
        self.kappa = kappa
