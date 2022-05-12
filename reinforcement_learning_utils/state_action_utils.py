

class State:

    def __init__(self):

        pass

    def set_trading_attributes(self, current_factor, current_rescaled_shares, current_other_observable, shares_scale,
                               current_price):

        self.shares_scale = shares_scale

        self.current_factor = current_factor
        self.current_rescaled_shares = current_rescaled_shares
        self.current_other_observable = current_other_observable  # e.g. average of last few PnLs, in case the factor is hidden
        self.current_shares = self.current_rescaled_shares * self.shares_scale
        self.current_price = current_price

    def set_extra_trading_attributes(self, next_factor: float, next_price: float, next_other_observable: float):

        self.next_factor = next_factor
        self.next_price = next_price
        self.next_other_observable = next_other_observable


class ActionSpace:

    def __init__(self, state: State):

        self.state = state

    def set_trading_actions_interval(self):

        current_rescaled_shares = self.state.current_rescaled_shares
        self.actions_interval = (-1 - current_rescaled_shares, 1 - current_rescaled_shares)


class Action:

    def __init__(self):

        pass

    def set_trading_attributes(self, rescaled_trade: float, shares_scale: float = 1):

        self.rescaled_trade = rescaled_trade
        self.shares_scale = shares_scale
        self.trade = self.rescaled_trade * self.shares_scale
