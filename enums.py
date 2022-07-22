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


class FactorType(Enum):

    Observable = 'Observable'
    Latent = 'Latent'


class FactorDefinitionType(Enum):

    MovingAverage = 'MovingAverage'
    StdMovingAverage = 'StdMovingAverage'


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
