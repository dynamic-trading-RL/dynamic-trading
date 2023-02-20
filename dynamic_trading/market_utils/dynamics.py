from typing import Tuple

import pandas as pd

from dynamic_trading.market_utils.calibrator import DynamicsCalibrator, _build_filename_calibrations
from dynamic_trading.enums.enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType


class Dynamics:
    """
    Base class for expressing the dynamics of a risk-driver or factor.

    :ivar FactorDynamicsType factorDynamicsType: Dynamics assigned to the factor.
    :ivar dict parameters: Dictionary containing the dynamics parameter.
    :ivar RiskDriverDynamicsType riskDriverDynamicsType: Dynamics assigned to the risk-driver.
    :ivar RiskDriverType riskDriverType: Type of risk-driver.
  """

    def __init__(self):
        """
        Class constructor.
        """

        self.factorDynamicsType = None
        self.riskDriverDynamicsType = None
        self.parameters = {}
        self.riskDriverType = None

    def _read_parameters_from_file(self, ticker: str):

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

    def _check_riskDriverType_in_file(self, ticker: str):

        filename = self._get_filename(ticker)

        sheet_name = 'riskDriverType'

        riskDriverType_df = pd.read_excel(filename, sheet_name=sheet_name)
        riskDriverType_in_file = RiskDriverType(riskDriverType_df['riskDriverType'].iloc[0])

        if self.riskDriverType != riskDriverType_in_file:
            raise NameError('riskDriverType in ' + filename + ' is not as it should be')

    def _set_risk_driver_start_price_from_file(self, ticker: str):

        if type(self) == RiskDriverDynamics:

            filename = self._get_filename(ticker)

            sheet_name = 'start_price'

            start_price_df = pd.read_excel(filename, sheet_name=sheet_name)
            start_price = start_price_df['start_price'].iloc[0]

            self.start_price = start_price

    def _get_filename(self, ticker: str):

        if type(self) == RiskDriverDynamics:
            var_type = 'risk-driver'
        elif type(self) == FactorDynamics:
            var_type = 'factor'
        else:
            raise NameError('dynamicsType not properly set')

        riskDriverType = self.riskDriverType

        filename = _build_filename_calibrations(riskDriverType, ticker, var_type)

        return filename

    @staticmethod
    def _read_risk_driver_start_price_from_calibrator(dynamicsCalibrator: DynamicsCalibrator):

        start_price = dynamicsCalibrator.financialTimeSeries.info['start_price']

        return start_price

    def _set_parameters_from_calibrator(self, dynamicsCalibrator: DynamicsCalibrator):

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
        param_dict = dynamicsCalibrator.get_params_dict(var_type, dynamicsType)
        self._set_parameters_from_dict_impl(param_dict)

    def _set_parameters_from_file(self, ticker: str, riskDriverType: RiskDriverType):

        self.riskDriverType = riskDriverType
        self._check_riskDriverType_in_file(ticker)
        self._set_risk_driver_start_price_from_file(ticker)
        param_dict = self._read_parameters_from_file(ticker)
        self._set_parameters_from_dict_impl(param_dict)

    def _set_parameters_from_dict_impl(self, param_dict: dict):

        if type(self) == RiskDriverDynamics:

            self._set_risk_driver_parameters_from_dict(param_dict)

        elif type(self) == FactorDynamics:

            self._set_factor_parameters_from_dict(param_dict)

        else:
            raise NameError('Invalid dynamics')

    def _set_risk_driver_parameters_from_dict(self, param_dict: dict):

        if self.riskDriverDynamicsType == RiskDriverDynamicsType.Linear:

            self._set_linear_parameters(param_dict)

        elif self.riskDriverDynamicsType == RiskDriverDynamicsType.NonLinear:

            self._set_threshold_parameters(param_dict)

        else:
            raise NameError('Invalid riskDriverDynamicsType: ' + self.riskDriverDynamicsType.value)

    def _set_factor_parameters_from_dict(self, param_dict: dict):

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
    """
    General class for expressing the dynamics of a risk-driver.

    :ivar RiskDriverDynamicsType riskDriverDynamicsType: Dynamics assigned to the risk-driver.
  """

    def __init__(self, riskDriverDynamicsType: RiskDriverDynamicsType):
        """
        Class constructur.

        :param RiskDriverDynamicsType riskDriverDynamicsType: Dynamics assigned to the risk-driver.
        """

        super().__init__()
        self.riskDriverDynamicsType = riskDriverDynamicsType

    def set_parameters_from_calibrator(self, dynamicsCalibrator: DynamicsCalibrator):
        """
        Set class attributes starting from a DynamicsCalibrator.

        :param DynamicsCalibrator dynamicsCalibrator: DynamicsCalibrator used to calibrate the model.
        """

        super()._set_parameters_from_calibrator(dynamicsCalibrator)

    def set_parameters_from_file(self, ticker: str, riskDriverType: RiskDriverType):
        """
        Set class attributes starting from a file.

        :param str ticker: Ticker of the security.
        :param RiskDriverType riskDriverType: Type of risk-driver.
        """

        super()._set_parameters_from_file(ticker, riskDriverType)


class FactorDynamics(Dynamics):
    """
    General class for expressing the dynamics of a factor.

    :ivar FactorDriverDynamicsType factorDriverDynamicsType: Dynamics assigned to the factor.
    """

    def __init__(self, factorDynamicsType: FactorDynamicsType):
        """
        Class constructur.

        :param FactorDynamicsType factorDynamicsType: Dynamics assigned to the factor.
        """

        super().__init__()
        self.factorDynamicsType = factorDynamicsType

    def set_parameters_from_calibrator(self, dynamicsCalibrator: DynamicsCalibrator):
        """
        Set class attributes starting from a DynamicsCalibrator.

        :param DynamicsCalibrator dynamicsCalibrator: DynamicsCalibrator used to calibrate the model.
        """

        super()._set_parameters_from_calibrator(dynamicsCalibrator)

    def set_parameters_from_file(self, ticker: str, riskDriverType: RiskDriverType):
        """
        Set class attributes starting from a file.

        :param str ticker: Ticker of the security.
        :param RiskDriverType riskDriverType: Type of risk-driver.
        """

        super()._set_parameters_from_file(ticker, riskDriverType)


class MarketDynamics:
    """
    General class for expressing the market dynamics combining risk-driver and factor dynamics'.

    :ivar FactorDynamics factorDynamics: Instance of FactorDynamics.
    :ivar RiskDriverDynamics riskDriverDynamics: Instance of RiskDriverDynamics.
    :ivar RiskDriverType riskDriverType: Type of risk-driver.
    :ivar float start_price: Price of the security at time :math:`t=0`.
    """

    def __init__(self, riskDriverDynamics: RiskDriverDynamics, factorDynamics: FactorDynamics):
        """
        Class constructur.

        :param RiskDriverDynamics riskDriverDynamics: Dynamics assigned to the risk-driver.
        :param FactorDynamics factorDynamics: Dynamics assigned to the factor.
        """

        self.riskDriverDynamics = riskDriverDynamics
        self.factorDynamics = factorDynamics
        self._set_riskDriverType()
        self._set_start_price()

    def get_riskDriverDynamicsType_and_parameters(self) -> Tuple[RiskDriverDynamicsType, dict]:
        """
        Service function that returns the riskDriverDynamicsType with its parameters.

        :return: riskDriverDynamicsType and parameters.
        """

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
