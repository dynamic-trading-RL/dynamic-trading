from dynamic_trading.enums.enums import StrategyType


class Action:
    """
    Class defining an action.

    Attributes
    ----------
    rescaled_trade : float
        A trade, rescaled by the factor :obj:`shares_scale`.
    shares_scale : float
        Factor for rescaling the shares :math:`M`.
    trade: float
        A trade.

    """

    def __init__(self):
        """
        Class constructor.

        """
        self.trade = None
        self.shares_scale = None
        self.rescaled_trade = None

    def set_trading_attributes(self, rescaled_trade: float, shares_scale: float = 1):
        """
        Service method for setting the class attributes.

        Parameters
        ----------
        rescaled_trade : float
            A trade, rescaled by the factor :obj:`shares_scale`.
        shares_scale : float
            Factor for rescaling the shares :math:`M`.

        """

        self.rescaled_trade = rescaled_trade
        self.shares_scale = shares_scale
        self.trade = self.rescaled_trade * self.shares_scale


class State:
    """
    Class defining a state.

    Attributes
    ----------
    action_GP : Action
        Action performed by a GP agent.
    average_past_pnl : float
        Average past security's P\&Ls.
    environment : Environment
        The environment generating the state.
    factor : float
        Observation of the factor :math:`f_{t}`.
    pnl : float
        Current security's P\&L observation.
    price : float
        Price :math:`p_{t}` of the security.
    rescaled_shares : float
        Current rescaled shares :math:`n_{t} / M`.
    shares : float
        Current shares.
    shares_scale : float
        Factor for rescaling the shares :math:`M`.
    ttm : float
        Current time-to-maturity.

    """

    def __init__(self, environment):
        """
        Class constructor.

        Parameters
        ----------
        environment : Environment
            The environment generating the state.

        """

        self.environment = environment

        self.action_GP = None
        self.price = None
        self.pnl = None
        self.average_past_pnl = None
        self.shares = None
        self.rescaled_shares = None
        self.factor = None
        self.shares_scale = None
        self.ttm = None

    def set_trading_attributes(self, factor, rescaled_shares, shares_scale,
                               price, pnl, average_past_pnl, action_GP, ttm):
        """
        Service method to set the class attributes.

        Parameters
        ----------
        factor : float
            Observation of the factor :math:`f_{t}`.
        rescaled_shares : float
            Current rescaled shares :math:`n_{t} / M`.
        shares_scale : float
            Factor for rescaling the shares :math:`M`.
        price : float
            Price :math:`p_{t}` of the security.
        pnl : float
            Current security's P\&L observation.
        average_past_pnl : float
            Average past security's P\&Ls.
        action_GP : Action
            Action performed by a GP agent.
        ttm : float
            Current time-to-maturity.

        """

        self.shares_scale = shares_scale

        self.factor = factor
        self.rescaled_shares = rescaled_shares
        self.shares = self.rescaled_shares * self.shares_scale
        self.price = price
        self.pnl = pnl
        self.average_past_pnl = average_past_pnl
        self.action_GP = action_GP
        self.ttm = ttm


class ActionSpace:
    """
    Class defining the action space.

    Attributes
    ----------
    actions_interval : tuple
        Two-dimensional tuple defining the bounds on which the action is defined.
    state: State
        The action space depends on the current state.
    strategyType : :class:`~dynamic_trading.enums.enums.StrategyType`
        The strategy being performed by the agent. Refer to :class:`~dynamic_trading.enums.enums.StrategyType` for more
        details.

    """

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
