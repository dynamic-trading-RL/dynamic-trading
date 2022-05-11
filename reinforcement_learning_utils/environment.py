import pandas as pd

from reinforcement_learning_utils.state_action_utils import Action, State
from market_utils.market import Market


class Environment:

    def __init__(self, market: Market):

        self.market = market
        self._set_attributes()

    def compute_reward_and_next_state(self, state: State, action: Action):

        reward = self._compute_reward(state=state, action=action)
        next_state = self._compute_next_state(state=state, action=action)

        return reward, next_state

    def _compute_reward(self, state: State, action: Action):

        reward = self._compute_trading_reward(state, action)

        return reward

    def _compute_next_state(self, state: State, action: Action):

        next_state = self._compute_trading_next_state(state, action)

        return next_state

    def _compute_trading_reward(self, state, action):

        current_shares = state.current_shares
        current_factor = state.current_factor
        current_price = state.current_price

        pnl = self.market.next_step_pnl(factor=current_factor, price=current_price)
        sig2 = self.market.next_step_pnl_sig2(factor=current_factor, price=current_price)

        cost = self._compute_trading_cost(action, sig2)

        reward = self.gamma * (current_shares * pnl - 0.5*self.kappa*current_shares*sig2*current_shares) - cost

        return reward

    def _compute_trading_next_state(self, state, action):

        current_rescaled_shares = state.current_rescaled_shares
        shares_scale = state.shares_scale

        next_factor = state.next_factor
        next_rescaled_shares = current_rescaled_shares + action.rescaled_trade
        next_other_observables = state.next_other_observable
        next_price = state.next_price

        next_state = State()
        next_state.set_trading_attributes(current_factor=next_factor,
                                         current_rescaled_shares=next_rescaled_shares,
                                         current_other_observable=next_other_observables,
                                         shares_scale=shares_scale,
                                         current_price=next_price)

        return next_state

    def _compute_trading_cost(self, action, sig2):

        trade = action.trade

        return 0.5 * trade * self.lam * sig2 * trade

    def _set_attributes(self):

        self._set_trading_attributes()

    def _set_trading_attributes(self):

        ticker = self.market.ticker
        filename = '../data/data_source/trading_data/' + ticker + '-trading-parameters.csv'
        df_trad_params = pd.read_csv(filename, index_col=0)
        gamma = df_trad_params.loc['gamma'][0]
        kappa = df_trad_params.loc['kappa'][0]
        lam = df_trad_params.loc['lam'][0]

        self.gamma = gamma
        self.kappa = kappa
        self.lam = lam
