

class State:

    def __init__(self, current_factor: float, current_rescaled_shares: float, shares_scale: float = 1,
                 other_observable: float = None):

        self.current_factor = current_factor
        self.current_rescaled_shares = current_rescaled_shares

        self.other_observable = other_observable  # e.g. average of last few PnLs, in case the factor is hidden

        self.shares_scale = shares_scale
        self.current_shares = self.current_rescaled_shares * self.shares_scale


class ActionSpace:

    def __init__(self, state: State):

        self.state = state
        self._set_actions_interval()

    def _set_actions_interval(self):

        current_rescaled_shares = self.state.current_rescaled_shares

        self.actions_interval = (-1 - current_rescaled_shares, 1 - current_rescaled_shares)


class Action:

    def __init__(self, rescaled_trade: float, shares_scale: float = 1):

        self.rescaled_trade = rescaled_trade

        self.shares_scale = shares_scale
        self.trade = self.rescaled_trade * self.shares_scale
