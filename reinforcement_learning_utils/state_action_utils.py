from enums import StrategyType


class Action:

    def __init__(self):
        self.trade = None
        self.shares_scale = None
        self.rescaled_trade = None

    def set_trading_attributes(self, rescaled_trade: float, shares_scale: float = 1):

        self.rescaled_trade = rescaled_trade
        self.shares_scale = shares_scale
        self.trade = self.rescaled_trade * self.shares_scale


class State:

    def __init__(self, environment):

        self.environment = environment

        self.action_GP = None
        self.price = None
        self.pnl = None
        self.average_past_pnl = None
        self.shares = None
        self.other_observable = None
        self.rescaled_shares = None
        self.factor = None
        self.shares_scale = None
        self.ttm = None

    def set_trading_attributes(self, factor, rescaled_shares, other_observable, shares_scale,
                               price, pnl, average_past_pnl, action_GP, ttm):

        self.shares_scale = shares_scale

        self.factor = factor
        self.rescaled_shares = rescaled_shares
        self.other_observable = other_observable  # e.g. average of last  PnLs when factor is hidden
        self.shares = self.rescaled_shares * self.shares_scale
        self.price = price
        self.pnl = pnl
        self.average_past_pnl = average_past_pnl
        self.action_GP = action_GP
        self.ttm = ttm


class ActionSpace:

    def __init__(self, state: State, strategyType: StrategyType = StrategyType.Unconstrained):

        self.state = state
        self.strategyType = strategyType
        self.actions_interval = None

    def set_trading_actions_interval(self):

        rescaled_shares = self.state.rescaled_shares

        if self.strategyType == StrategyType.Unconstrained:
            self.actions_interval = (-1 - rescaled_shares, 1 - rescaled_shares)
        elif self.strategyType == StrategyType.LongOnly:
            self.actions_interval = (- rescaled_shares, 1 - rescaled_shares)
        else:
            raise NameError(f'Invalid strategyType = {self.strategyType.value}')
