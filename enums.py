from enum import Enum


class RiskDriverDynamicsType(Enum):

    Linear = 'Linear'
    NonLinear = 'NonLinear'


class FactorDynamicsType(Enum):

    AR = 'AR'
    SETAR = 'SETAR'
    GARCH = 'GARCH'
    TARCH = 'TARCH'
    AR_TARCH = 'AR_TARCH'


class FactorComputationType(Enum):

    MovingAverage = 'MovingAverage'
    StdMovingAverage = 'StdMovingAverage'


class FactorTransformationType(Enum):

    Diff = 'Diff'
    LogDiff = 'LogDiff'


class RiskDriverType(Enum):

    PnL = 'PnL'
    Return = 'Return'


class FactorSourceType(Enum):

    Constructed = 'Constructed'
    Exogenous = 'Exogenous'


class ModeType(Enum):

    InSample = 'InSample'
    OutOfSample = 'OutOfSample'


class EstimateInitializationType(Enum):

    RandomUniform = 'RandomUniform'
    RandomTruncNorm = 'RandomTruncNorm'
    GP = 'GP'


class StrategyType(Enum):

    Unconstrained = 'Unconstrained'
    LongOnly = 'LongOnly'


class OptimizerType(Enum):

    basinhopping = 'basinhopping'
    brute = 'brute'
    differential_evolution = 'differential_evolution'
    dual_annealing = 'dual_annealing'
    shgo = 'shgo'
    local = 'local'