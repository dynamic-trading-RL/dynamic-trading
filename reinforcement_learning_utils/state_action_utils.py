from enums import StrategyType


class State:

    def __init__(self):
        self.next_other_observable = None
        self.next_price = None
        self.next_factor = None
        self.current_action_GP = None
        self.current_price = None
        self.current_shares = None
        self.current_other_observable = None
        self.current_rescaled_shares = None
        self.current_factor = None
        self.shares_scale = None

    def set_trading_attributes(self, current_factor, current_rescaled_shares, current_other_observable, shares_scale,
                               current_price, action_GP):

        self.shares_scale = shares_scale

        self.current_factor = current_factor
        self.current_rescaled_shares = current_rescaled_shares
        self.current_other_observable = current_other_observable  # e.g. average of last  PnLs when factor is hidden
        self.current_shares = self.current_rescaled_shares * self.shares_scale
        self.current_price = current_price
        self.current_action_GP = action_GP

    def set_extra_trading_attributes(self, next_factor: float, next_price: float, next_other_observable: float):

        self.next_factor = next_factor
        self.next_price = next_price
        self.next_other_observable = next_other_observable


class ActionSpace:

    def __init__(self, state: State, strategyType: StrategyType = StrategyType.Unconstrained):
    
        self.state = state
        self.strategyType = strategyType
        self.actions_interval = None

    def set_trading_actions_interval(self):

        current_rescaled_shares = self.state.current_rescaled_shares

        if self.strategyType == StrategyType.Unconstrained:
            self.actions_interval = (-1 - current_rescaled_shares, 1 - current_rescaled_shares)
        elif self.strategyType == StrategyType.LongOnly:
            self.actions_interval = (- current_rescaled_shares, 1 - current_rescaled_shares)
        else:
            raise NameError(f'Invalid strategyType = {self.strategyType.value}')


class Action:

    def __init__(self):
        self.trade = None
        self.shares_scale = None
        self.rescaled_trade = None

    def set_trading_attributes(self, rescaled_trade: float, shares_scale: float = 1):

        self.rescaled_trade = rescaled_trade
        self.shares_scale = shares_scale
        self.trade = self.rescaled_trade * self.shares_scale
