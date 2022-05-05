import pandas as pd
from enums import AssetDynamicsType, FactorDynamicsType


class Dynamics:

    def __init__(self):

        self._parameters = {}

    def _set_linear_parameters(self, param_dict):

        self._parameters['mu'] = param_dict['mu']
        self._parameters['B'] = param_dict['B']
        self._parameters['sig2'] = param_dict['sig2']

    def _set_threshold_parameters(self, param_dict):

        self._parameters['c'] = param_dict['c']
        self._parameters['mu_0'] = param_dict['mu_0']
        self._parameters['B_0'] = param_dict['B_0']
        self._parameters['sig2_0'] = param_dict['sig2_0']
        self._parameters['mu_1'] = param_dict['mu_1']
        self._parameters['B_1'] = param_dict['B_1']
        self._parameters['sig2_1'] = param_dict['sig2_1']


class AssetDynamics(Dynamics):

    def __init__(self, assetDynamicsType: AssetDynamicsType):

        super().__init__()
        self._assetDynamicsType = assetDynamicsType

    def set_parameters(self, param_dict):

        if self._assetDynamicsType == AssetDynamicsType.Linear:

            self._set_linear_parameters(param_dict)

        elif self._assetDynamicsType == AssetDynamicsType.NonLinear:

            self._set_threshold_parameters(param_dict)

        else:
            raise NameError('Invalid return dynamics')

    def read_asset_parameters(self):

        if self._assetDynamicsType == AssetDynamicsType.Linear:

            params = pd.read_excel('data_tmp/return_calibrations.xlsx',
                                   sheet_name='linear',
                                   index_col=0)

        elif self._assetDynamicsType == AssetDynamicsType.NonLinear:

            params = pd.read_excel('data_tmp/return_calibrations.xlsx',
                                   sheet_name='non-linear',
                                   index_col=0)

        else:
            raise NameError('Invalid assetDynamicsType')

        return params['param'].to_dict()


class FactorDynamics(Dynamics):

    def __init__(self, factorDynamicsType: FactorDynamicsType):

        super().__init__()
        self._factorDynamicsType = factorDynamicsType

    def set_parameters(self, param_dict):

        if self._factorDynamicsType == FactorDynamicsType.AR:

            self._set_linear_parameters(param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.SETAR:

            self._set_threshold_parameters(param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.GARCH:

            self._set_garch_parameters(param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.TARCH:

            self._set_tarch_parameters(param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.AR_TARCH:

            self._set_artarch_parameters(param_dict)

        else:
            raise NameError('Invalid factor dynamics')

    def read_factor_parameters(self):

        if self._factorDynamicsType == FactorDynamicsType.AR:

            params = pd.read_excel('data_tmp/factor_calibrations.xlsx',
                                   sheet_name='AR',
                                   index_col=0)

        elif self._factorDynamicsType == FactorDynamicsType.SETAR:

            params = pd.read_excel('data_tmp/factor_calibrations.xlsx',
                                   sheet_name='SETAR',
                                   index_col=0)

        elif self._factorDynamicsType == FactorDynamicsType.GARCH:

            params = pd.read_excel('data_tmp/factor_calibrations.xlsx',
                                   sheet_name='GARCH',
                                   index_col=0)

        elif self._factorDynamicsType == FactorDynamicsType.TARCH:

            params = pd.read_excel('data_tmp/factor_calibrations.xlsx',
                                   sheet_name='TARCH',
                                   index_col=0)

        elif self._factorDynamicsType == FactorDynamicsType.AR_TARCH:

            params = pd.read_excel('data_tmp/factor_calibrations.xlsx',
                                   sheet_name='AR-TARCH',
                                   index_col=0)

        else:
            raise NameError('Invalid factorDynamicsType')

        return params['param'].to_dict()

    def _set_artarch_parameters(self, param_dict):

        self._set_tarch_parameters(param_dict)
        self._parameters['B'] = param_dict['B']

    def _set_tarch_parameters(self, param_dict):

        self._set_garch_parameters(param_dict)
        self._parameters['gamma'] = param_dict['gamma']
        self._parameters['c'] = param_dict['c']

    def _set_garch_parameters(self, param_dict):

        self._parameters['mu'] = param_dict['mu']
        self._parameters['omega'] = param_dict['omega']
        self._parameters['alpha'] = param_dict['alpha']
        self._parameters['beta'] = param_dict['beta']


class MarketDynamics:

    def __init__(self,
                 assetDynamics: AssetDynamics,
                 factorDynamics: FactorDynamics):

        self._assetDynamics = assetDynamics
        self._factorDynamics = factorDynamics


class AllMarkets:

    def __init__(self):

        self._allMarkets_dict = {}

    def fill_allMarkets_dict(self, d):

        for key, item in d.items():

            self._allMarkets_dict[key] = item
