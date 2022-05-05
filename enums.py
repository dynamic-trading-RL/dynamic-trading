from enum import Enum


class AssetDynamicsType(Enum):

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
