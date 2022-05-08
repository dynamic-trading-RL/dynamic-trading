import pandas as pd

from calibrator import DynamicsCalibrator, build_filename_calibrations
from enums import AssetDynamicsType, FactorDynamicsType


class Dynamics:

    def __init__(self):

        self.parameters = {}
        self.factor_predicts_pnl = None

    def _read_parameters_from_file(self, ticker):

        filename = self._get_filename(ticker)

        if type(self) == AssetDynamics:
            sheet_name = self.assetDynamicsType.value

        elif type(self) == FactorDynamics:
            sheet_name = self.factorDynamicsType.value

        else:
            raise NameError('dynamicsType not properly set')

        params = pd.read_excel(filename,
                               sheet_name=sheet_name,
                               index_col=0)

        return params['param'].to_dict()

    def _check_factor_predicts_pnl_in_file(self, ticker):

        filename = self._get_filename(ticker)

        sheet_name = 'factor_predicts_pnl'

        factor_predicts_pnl_df = pd.read_excel(filename, sheet_name=sheet_name)
        factor_predicts_pnl_str = str(factor_predicts_pnl_df['factor_predicts_pnl'].iloc[0])
        factor_predicts_pnl_in_file = factor_predicts_pnl_str == 'True'

        if self.factor_predicts_pnl != factor_predicts_pnl_in_file:
            raise NameError('factor_predicts_pnl in ' + filename + ' is not as it should be')

    def _set_asset_start_price_from_file(self, ticker):

        if type(self) == AssetDynamics:

            filename = self._get_filename(ticker)

            sheet_name = 'start_price'

            start_price_df = pd.read_excel(filename, sheet_name=sheet_name)
            start_price = start_price_df['start_price'].iloc[0]

            self.start_price = start_price

    def _get_filename(self, ticker):

        if type(self) == AssetDynamics:
            var_type = 'asset'
        elif type(self) == FactorDynamics:
            var_type = 'factor'
        else:
            raise NameError('dynamicsType not properly set')

        factor_predicts_pnl = self.factor_predicts_pnl

        filename = build_filename_calibrations(factor_predicts_pnl, ticker, var_type)

        return filename

    def _read_asset_start_price_from_calibrator(self, dynamicsCalibrator):

        start_price = dynamicsCalibrator.financialTimeSeries.info['start_price']

        return start_price

    def _set_parameters_from_calibrator(self, dynamicsCalibrator):

        if type(self) == AssetDynamics:
            var_type = 'asset'
            dynamicsType = self.assetDynamicsType
            self.start_price = self._read_asset_start_price_from_calibrator(dynamicsCalibrator)
        elif type(self) == FactorDynamics:
            var_type = 'factor'
            dynamicsType = self.factorDynamicsType
        else:
            raise NameError('Invalid dynamics')

        self.factor_predicts_pnl = dynamicsCalibrator.factor_predicts_pnl
        param_dict = dynamicsCalibrator.get_param_dict(var_type, dynamicsType)
        self._set_parameters_from_dict_impl(param_dict)

    def _set_parameters_from_file(self, ticker, factor_predicts_pnl):

        self.factor_predicts_pnl = factor_predicts_pnl
        self._check_factor_predicts_pnl_in_file(ticker)
        self._set_asset_start_price_from_file(ticker)
        param_dict = self._read_parameters_from_file(ticker)
        self._set_parameters_from_dict_impl(param_dict)

    def _set_parameters_from_dict_impl(self, param_dict):

        if type(self) == AssetDynamics:

            self._set_asset_parameters_from_dict(param_dict)

        elif type(self) == FactorDynamics:

            self._set_factor_parameters_from_dict(param_dict)

        else:
            raise NameError('Invalid dynamics')

    def _set_asset_parameters_from_dict(self, param_dict):

        if self.assetDynamicsType == AssetDynamicsType.Linear:

            self._set_linear_parameters(param_dict)

        elif self.assetDynamicsType == AssetDynamicsType.NonLinear:

            self._set_threshold_parameters(param_dict)

        else:
            raise NameError('Invalid asset dynamics')

    def _set_factor_parameters_from_dict(self, param_dict):

        if self.factorDynamicsType == FactorDynamicsType.AR:

            self._set_linear_parameters(param_dict)

        elif self.factorDynamicsType == FactorDynamicsType.SETAR:

            self._set_threshold_parameters(param_dict)

        elif self.factorDynamicsType == FactorDynamicsType.GARCH:

            self._set_garch_parameters(param_dict)

        elif self.factorDynamicsType == FactorDynamicsType.TARCH:

            self._set_tarch_parameters(param_dict)

        elif self.factorDynamicsType == FactorDynamicsType.AR_TARCH:

            self._set_ar_tarch_parameters(param_dict)

        else:
            raise NameError('Invalid factor dynamics')

    def _set_linear_parameters(self, param_dict: dict):

        self.parameters['mu'] = param_dict['mu']
        self.parameters['B'] = param_dict['B']
        self.parameters['sig2'] = param_dict['sig2']

    def _set_threshold_parameters(self, param_dict: dict):

        self.parameters['c'] = param_dict['c']
        self.parameters['mu_0'] = param_dict['mu_0']
        self.parameters['B_0'] = param_dict['B_0']
        self.parameters['sig2_0'] = param_dict['sig2_0']
        self.parameters['mu_1'] = param_dict['mu_1']
        self.parameters['B_1'] = param_dict['B_1']
        self.parameters['sig2_1'] = param_dict['sig2_1']
        self.parameters['p'] = param_dict['p']

    def _set_garch_parameters(self, param_dict: dict):

        self.parameters['mu'] = param_dict['mu']
        self.parameters['omega'] = param_dict['omega']
        self.parameters['alpha'] = param_dict['alpha']
        self.parameters['beta'] = param_dict['beta']

    def _set_tarch_parameters(self, param_dict: dict):

        self._set_garch_parameters(param_dict)
        self.parameters['gamma'] = param_dict['gamma']
        self.parameters['c'] = param_dict['c']

    def _set_ar_tarch_parameters(self, param_dict: dict):

        self._set_tarch_parameters(param_dict)
        self.parameters['B'] = param_dict['B']


class AssetDynamics(Dynamics):

    def __init__(self, assetDynamicsType: AssetDynamicsType):

        super().__init__()
        self.assetDynamicsType = assetDynamicsType
    
    def set_parameters_from_calibrator(self, dynamicsCalibrator: DynamicsCalibrator):

        super()._set_parameters_from_calibrator(dynamicsCalibrator)

    def set_parameters_from_file(self, ticker, factor_predicts_pnl):

        super()._set_parameters_from_file(ticker, factor_predicts_pnl)

class FactorDynamics(Dynamics):

    def __init__(self, factorDynamicsType: FactorDynamicsType):

        super().__init__()
        self.factorDynamicsType = factorDynamicsType

    def set_parameters_from_calibrator(self, dynamicsCalibrator: DynamicsCalibrator):

        super()._set_parameters_from_calibrator(dynamicsCalibrator)

    def set_parameters_from_file(self, ticker, factor_predicts_pnl):

        super()._set_parameters_from_file(ticker, factor_predicts_pnl)


class MarketDynamics:

    def __init__(self, assetDynamics: AssetDynamics, factorDynamics: FactorDynamics):

        self.assetDynamics = assetDynamics
        self.factorDynamics = factorDynamics
        self._set_factor_predicts_pnl()
        self._set_start_price()

    def get_assetDynamicsType_and_parameters(self):

        return self.assetDynamics.assetDynamicsType, self.assetDynamics.parameters

    def _set_factor_predicts_pnl(self):

        if self.assetDynamics.factor_predicts_pnl == self.factorDynamics.factor_predicts_pnl:

            self.factor_predicts_pnl = self.assetDynamics.factor_predicts_pnl

        else:
            raise NameError('factor_predicts_pnl different for asset and factor dynamics')

    def _set_start_price(self):

        self.start_price = self.assetDynamics.start_price


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':

    assetDynamics = AssetDynamics(AssetDynamicsType.NonLinear)
    factorDynamics = FactorDynamics(FactorDynamicsType.AR_TARCH)

    assetDynamics.set_parameters_from_file(ticker='WTI', factor_predicts_pnl=True)
    factorDynamics.set_parameters_from_file(ticker='WTI', factor_predicts_pnl=True)

    marketDynamics = MarketDynamics(assetDynamics=assetDynamics,
                                    factorDynamics=factorDynamics)
