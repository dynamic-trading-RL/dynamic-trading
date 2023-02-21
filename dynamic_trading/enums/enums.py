from enum import Enum


class RiskDriverDynamicsType(Enum):
    """
    Enum determining the type of risk-driver dynamics.

    """

    Linear = 'Linear'  #: A linear model.
    NonLinear = 'NonLinear'  #: A non-linear threshold model.


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
    Enum determining the type of filtering on the factor time series transformed as in FactorTransformationType.

    """

    MovingAverage = 'MovingAverage'  #: Uses moving average.
    StdMovingAverage = 'StdMovingAverage'  #: Uses a moving average standardized by the moving standard deviation.


class FactorTransformationType(Enum):
    """
    Enum determining the type of factor transformation to consider.

    """

    NoTransformation = 'NoTransformation'  #: No transformation is applied to the factor time series.
    Diff = 'Diff'  #: Consider the differences.
    LogDiff = 'LogDiff'  # Consider the log-differences.


class RiskDriverType(Enum):
    """
    Enum determining the type of risk driver.

    """

    PnL = 'PnL'  #: The variable :math:`x_t` coincides with the P\&L of the security.
    Return = 'Return'  #: The variable :math:`x_t` coincides with the return of the security.


class FactorSourceType(Enum):
    """
    Enum determining whether the factor is provided exogenously or constructed.

    """

    Constructed = 'Constructed'  #: The factor is constructed starting from the security time series.
    Exogenous = 'Exogenous'  # The factor is provided exogenously.


class ModeType(Enum):
    """
    Enum determining whether a FinancialTimeSeries object should be considered for in-sample purposes (e.g. estimation)
    or out-of-sample purposes (e.g. evaluation).

    """

    InSample = 'InSample'  #: In-sample mode.
    OutOfSample = 'OutOfSample'  #: Out-of-sample mode.


class RandomActionType(Enum):
    """
    Enum determining the type of random action to be performed for state-action value initialization or for the
    epsilon-greedy exploration.

    """

    RandomUniform = 'RandomUniform'  #: Uniform on action space.
    RandomTruncNorm = 'RandomTruncNorm'  #: Truncated normal on action space
    GP = 'GP'  #: GP action.


class StrategyType(Enum):
    """
    Enum determining what type of strategy an agent is following.

    """

    Unconstrained = 'Unconstrained'  #: Unconstrained strategy
    LongOnly = 'LongOnly'  #: Long-only strategy: position on the security can never go negative.


class OptimizerType(Enum):
    """
    Enum determining what type of optimizer to use in the greedy policy computation. See scipy.optimize for more
    details.

    """

    basinhopping = 'basinhopping'  #: Basin-hopping algorithm
    brute = 'brute'  #: Brute-force algorithm
    differential_evolution = 'differential_evolution'  #: Differential evolution algorithm
    dual_annealing = 'dual_annealing'  #: Dual-annealing algorithm
    shgo = 'shgo'  #: SHGO algorithm.
    local = 'local'  #: Local optimization.


class SupervisedRegressorType(Enum):
    """
    Enum determining what type of supervised regressor to use for interpolating the state-action value function.

    """

    ann = 'ann'  #: Artificial neural network
    gradient_boosting = 'gradient_boosting'  #: Gradient boosting regressor
    polynomial_regression = 'polynomial_regression'  #: Polynomial regressor


class InitialQvalueEstimateType(Enum):
    """
    Enum determining what type of initialization should be used for the state-action value function.

    """

    zero = 'zero'  #: state-action value function is initialized to 0.
    random = 'random'  #: state-action value function is not initialized; random actions are used. See also RandomActionType
