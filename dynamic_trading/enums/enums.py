from enum import Enum


class RiskDriverDynamicsType(Enum):
    """
    Enum determining the type of risk-driver dynamics.

    """

    Linear = 'Linear'  #: A linear model.
    NonLinear = 'NonLinear'  #: A non-linear threshold model.
    gBm = 'gBm'


class FactorDynamicsType(Enum):
    """
    Enum determining the type of factor dynamics.

    """

    AR = 'AR'  #: AR(1) model.
    SETAR = 'SETAR'  #: SETAR(1) model.
    GARCH = 'GARCH'  #: GARCH(1, 1) model.
    TARCH = 'TARCH'  #: TARCH(1, 1, 1) model.
    AR_TARCH = 'AR_TARCH'  #: AR(1) model with TARCH(1, 1, 1) residuals.


class FactorComputationType(Enum):
    """
    Enum determining the type of filtering on the factor time series transformed as in
    :class:`~dynamic_trading.enums.enums.FactorTransformationType`.

    """

    MovingAverage = 'MovingAverage'  #: Uses moving average.
    StdMovingAverage = 'StdMovingAverage'  #: Uses a moving average divided by the moving standard deviation.


class FactorTransformationType(Enum):
    """
    Enum determining the type of transformation to apply to the financial entity used as factor. For example, we can
    take as factor the VIX as is, or its increments, or its log-increments.

    """

    NoTransformation = 'NoTransformation'  #: No transformation is applied to the factor time series.
    Diff = 'Diff'  #: Consider the differences.
    LogDiff = 'LogDiff'  #: Consider the log-differences.


class RiskDriverType(Enum):
    """
    Enum determining the type of risk-driver :math:`x_t` to consider in the factor model :math:`x_t=g(f_t)`.

    """

    PnL = 'PnL'  #: The variable :math:`x_t` coincides with the P\&L of the security.
    Return = 'Return'  #: The variable :math:`x_t` coincides with the return of the security.


class FactorSourceType(Enum):
    """
    Enum determining whether the factor is provided exogenously or constructed starting from the traded security.

    """

    Constructed = 'Constructed'  #: The factor is constructed starting from the traded security.
    Exogenous = 'Exogenous'  #: The factor is provided exogenously.


class ModeType(Enum):
    """
    Enum determining whether a :class:`~dynamic_trading.market_utils.financial_time_series.FinancialTimeSeries` object should be considered for in-sample purposes
    or out-of-sample purposes. In the first case, the time series is being used to calibrate the models used in the MDP
    training the RL agent. In the second case, the time series is being used to backtest a trained RL agent on real
    data. Refer to :class:`~dynamic_trading.market_utils.financial_time_series.FinancialTimeSeries` for more details.

    """

    InSample = 'InSample'  #: In-sample mode.
    OutOfSample = 'OutOfSample'  #: Out-of-sample mode.


class RandomActionType(Enum):
    """
    Enum determining the type of random action to be performed in a :math:`\epsilon`-greedy exploration. Used also in
    case :class:`~dynamic_trading.enums.enums.InitialQvalueEstimateType` is :obj:`random`.

    """

    RandomUniform = 'RandomUniform'  #: Random uniform on action space :math:`A`.
    RandomTruncNorm = 'RandomTruncNorm'  #: Truncated normal on action space :math:`A`.
    GP = 'GP'  #: Rather than a random action, the RL performs the GP policy (optimal for the linear case).


class StrategyType(Enum):
    """
    Enum determining what type of trading strategy an agent is following.

    """

    Unconstrained = 'Unconstrained'  #: Unconstrained strategy: the agent tries to optimize the trade without constraints.
    LongOnly = 'LongOnly'  #: Long-only strategy: the agent tries to optimize the trade with a long-only constraint on its portfolio.


class OptimizerType(Enum):
    """
    Enum determining what type of optimizer to use in the greedy policy computation. Refer to the :obj:`scipy.optimize`
    documentation for more details.

    """

    basinhopping = 'basinhopping'  #: Basin-hopping algorithm.
    brute = 'brute'  #: Brute-force algorithm.
    differential_evolution = 'differential_evolution'  #: Differential evolution algorithm.
    dual_annealing = 'dual_annealing'  #: Dual-annealing algorithm.
    shgo = 'shgo'  #: SHGO algorithm.
    local = 'local'  #: Local optimization.


class SupervisedRegressorType(Enum):
    """
    Enum determining what type of supervised regressor is fitted to interpolate the state-action value function
    :math:`q(s,a)`. For more details, refer to the :obj:`sklearn`.

    """

    ann = 'ann'  #: Artificial neural network.
    gradient_boosting = 'gradient_boosting'  #: Gradient boosting regressor.
    polynomial_regression = 'polynomial_regression'  #: Polynomial regressor.


class InitialQvalueEstimateType(Enum):
    """
    Enum determining what type of initialization should be used for the state-action value function :math:`q(s,a)`
    as starting point of the learning procedure.

    """

    zero = 'zero'  #: State-action value function is initialized to 0.
    random = 'random'  #: State-action value function is not initialized; random actions are used. See also :class:`~dynamic_trading.enums.enums.RandomActionType`.
