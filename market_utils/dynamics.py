import pandas as pd

from calibrator import DynamicsCalibrator, build_filename_calibrations
from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType


class Dynamics:

    def __init__(self):

        self.parameters = {}
        self.riskDriverType = None

    def _read_parameters_from_file(self, ticker):

        filename = self._get_filename(ticker)

        if type(self) == RiskDriverDynamics:
            sheet_name = self.riskDriverDynamicsType.value

        elif type(self) == FactorDynamics:
            sheet_name = self.factorDynamicsType.value

        else:
            raise NameError('dynamicsType not properly set')

        params = pd.read_excel(filename,
                               sheet_name=sheet_name,
                               index_col=0)

        return params['param'].to_dict()

    def _check_riskDriverType_in_file(self, ticker):

        filename = self._get_filename(ticker)

        sheet_name = 'riskDriverType'

        riskDriverType_df = pd.read_excel(filename, sheet_name=sheet_name)
        riskDriverType_in_file = RiskDriverType(riskDriverType_df['riskDriverType'].iloc[0])

        if self.riskDriverType != riskDriverType_in_file:
            raise NameError('riskDriverType in ' + filename + ' is not as it should be')

    def _set_risk_driver_start_price_from_file(self, ticker):

        if type(self) == RiskDriverDynamics:

            filename = self._get_filename(ticker)

            sheet_name = 'start_price'

            start_price_df = pd.read_excel(filename, sheet_name=sheet_name)
            start_price = start_price_df['start_price'].iloc[0]

            self.start_price = start_price

    def _get_filename(self, ticker):

        if type(self) == RiskDriverDynamics:
            var_type = 'risk-driver'
        elif type(self) == FactorDynamics:
            var_type = 'factor'
        else:
            raise NameError('dynamicsType not properly set')

        riskDriverType = self.riskDriverType

        filename = build_filename_calibrations(riskDriverType, ticker, var_type)

        return filename

    def _read_risk_driver_start_price_from_calibrator(self, dynamicsCalibrator):

        start_price = dynamicsCalibrator.financialTimeSeries.info['start_price']

        return start_price

    def _set_parameters_from_calibrator(self, dynamicsCalibrator):

        if type(self) == RiskDriverDynamics:
            var_type = 'risk-driver'
            dynamicsType = self.riskDriverDynamicsType
            self.start_price = self._read_risk_driver_start_price_from_calibrator(dynamicsCalibrator)
        elif type(self) == FactorDynamics:
            var_type = 'factor'
            dynamicsType = self.factorDynamicsType
        else:
            raise NameError('Invalid dynamics')

        self.riskDriverType = dynamicsCalibrator.riskDriverType
        param_dict = dynamicsCalibrator.get_param_dict(var_type, dynamicsType)
        self._set_parameters_from_dict_impl(param_dict)

    def _set_parameters_from_file(self, ticker, riskDriverType):

        self.riskDriverType = riskDriverType
        self._check_riskDriverType_in_file(ticker)
        self._set_risk_driver_start_price_from_file(ticker)
        param_dict = self._read_parameters_from_file(ticker)
        self._set_parameters_from_dict_impl(param_dict)

    def _set_parameters_from_dict_impl(self, param_dict):

        if type(self) == RiskDriverDynamics:

            self._set_risk_driver_parameters_from_dict(param_dict)

        elif type(self) == FactorDynamics:

            self._set_factor_parameters_from_dict(param_dict)

        else:
            raise NameError('Invalid dynamics')

    def _set_risk_driver_parameters_from_dict(self, param_dict):

        if self.riskDriverDynamicsType == RiskDriverDynamicsType.Linear:

            self._set_linear_parameters(param_dict)

        elif self.riskDriverDynamicsType == RiskDriverDynamicsType.NonLinear:

            self._set_threshold_parameters(param_dict)

        else:
            raise NameError('Invalid riskDriverDynamicsType: ' + self.riskDriverDynamicsType.value)

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


class RiskDriverDynamics(Dynamics):

    def __init__(self, riskDriverDynamicsType: RiskDriverDynamicsType):

        super().__init__()
        self.riskDriverDynamicsType = riskDriverDynamicsType

    def set_parameters_from_calibrator(self, dynamicsCalibrator: DynamicsCalibrator):

        super()._set_parameters_from_calibrator(dynamicsCalibrator)

    def set_parameters_from_file(self, ticker, riskDriverType):

        super()._set_parameters_from_file(ticker, riskDriverType)

class FactorDynamics(Dynamics):

    def __init__(self, factorDynamicsType: FactorDynamicsType):

        super().__init__()
        self.factorDynamicsType = factorDynamicsType

    def set_parameters_from_calibrator(self, dynamicsCalibrator: DynamicsCalibrator):

        super()._set_parameters_from_calibrator(dynamicsCalibrator)

    def set_parameters_from_file(self, ticker, riskDriverType):

        super()._set_parameters_from_file(ticker, riskDriverType)


class MarketDynamics:

    def __init__(self, riskDriverDynamics: RiskDriverDynamics, factorDynamics: FactorDynamics):

        self.riskDriverDynamics = riskDriverDynamics
        self.factorDynamics = factorDynamics
        self._set_riskDriverType()
        self._set_start_price()

    def get_riskDriverDynamicsType_and_parameters(self):

        return self.riskDriverDynamics.riskDriverDynamicsType, self.riskDriverDynamics.parameters

    def _set_riskDriverType(self):

        if self.riskDriverDynamics.riskDriverType == self.factorDynamics.riskDriverType:

            self.riskDriverType = self.riskDriverDynamics.riskDriverType

        else:
            raise NameError('riskDriverType different for risk-driver and factor dynamics')

    def _set_start_price(self):

        self.start_price = self.riskDriverDynamics.start_price


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':

    riskDriverDynamics = RiskDriverDynamics(RiskDriverDynamicsType.NonLinear)
    factorDynamics = FactorDynamics(FactorDynamicsType.AR_TARCH)

    riskDriverDynamics.set_parameters_from_file(ticker='WTI', riskDriverType=RiskDriverType.PnL)
    factorDynamics.set_parameters_from_file(ticker='WTI', riskDriverType=RiskDriverType.PnL)

    marketDynamics = MarketDynamics(riskDriverDynamics=riskDriverDynamics,
                                    factorDynamics=factorDynamics)
