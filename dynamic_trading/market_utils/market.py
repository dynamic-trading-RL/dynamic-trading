import copy

import numpy as np
import pandas as pd
import os

from dynamic_trading.market_utils.dynamics import MarketDynamics, RiskDriverDynamics, FactorDynamics
from dynamic_trading.enums.enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType, ModeType
from dynamic_trading.market_utils.financial_time_series import FinancialTimeSeries


class Market:
    """
    General class for describing a financial time series.

    Attributes
    ----------
    financialTimeSeries : FinancialTimeSeries
        Financial time series acting on the market. See :class:`~dynamic_trading.market_utils.financial_time_series.FinancialTimeSeries` for more details.
    marketDynamics : MarketDynamics
        Market dynamics of risk-driver and factor in the :class:`~dynamic_trading.market_utils.financial_time_series.FinancialTimeSeries`.
    market_id : str
        Identifier of this :class:`~dynamic_trading.market_utils.market.Market` object.
    riskDriverType : :class:`~dynamic_trading.enums.enums.RiskDriverType`
        Risk-driver type assigned to the :class:`~dynamic_trading.market_utils.financial_time_series.FinancialTimeSeries`. See :class:`~dynamic_trading.enums.enums.RiskDriverType`. for more details.
    simulations: dict
        Dictionary indexes by market variable names 'risk-driver', 'factor', 'pnl', possibly 'sig' in case of ARCH
        dynamics. Each value in the dictionary is a pandas.Dataframe containing the simulations of the specific market
        variable.
    simulations_training : pandas.DataFrame
        Dictionary containing simulations used for training. As opposed to :obj:`simulations`, the primary key of this
        dictionary is an integer `n` indicating the batch on which the training is operating.
    start_price : float
        Price of the security at time :math:`t=0`.
    ticker : str
        An ID to identify the traded security.

    """

    def __init__(self, financialTimeSeries: FinancialTimeSeries, marketDynamics: MarketDynamics):
        """
        Class constructor.

        Parameters
        ----------
        financialTimeSeries : FinancialTimeSeries
            Financial time series operating on the market.
        marketDynamics : MarketDynamics
            Market dynamics assigned to the financial time series.

        """

        self.financialTimeSeries = financialTimeSeries
        self.marketDynamics = marketDynamics
        self._set_market_attributes()
        self.simulations_training = None

    def _next_step_risk_driver(self, factor: float):

        riskDriverDynamicsType, parameters = self.marketDynamics.get_riskDriverDynamicsType_and_parameters()

        if riskDriverDynamicsType == RiskDriverDynamicsType.Linear:

            return parameters['mu'] + parameters['B']*factor

        elif riskDriverDynamicsType == RiskDriverDynamicsType.NonLinear:

            if factor < parameters['c']:

                return parameters['mu_0'] + parameters['B_0']*factor

            else:

                return parameters['mu_1'] + parameters['B_1']*factor

    def next_step_pnl(self, factor: float, price: float = None):
        """
        Computes the next step P\&L of the security given the current price and factor.

        Parameters
        ----------
        factor : float
            Observation of the factor :math:`f_{t}`.
        price : float
            Price :math:`p_{t}` of the security.

        Returns
        -------
        pnl : float
            Next step P\&L.

        """

        if self.riskDriverType == RiskDriverType.PnL:

            pnl = self._next_step_risk_driver(factor)

        elif self.riskDriverType == RiskDriverType.Return:

            ret = self._next_step_risk_driver(factor)
            pnl = price * ret

        else:
            raise NameError('Invalid riskDriverType: ' + self.riskDriverType.value)

        return pnl

    def next_step_sig2(self, factor: float, price: float = None):
        """
        Computes the next-step P\&L variance for risk-cost evaluation purposes

        Parameters
        ----------
        factor : float
            Observation of the factor :math:`f_{t}`, used to predict the price change variance :math:`\Sigma=\Sigma_t`
            in case of non-constant variance, e.g. if the factor model is done on the security's return rather than
            P\&L.
        price : float
            Price :math:`p_{t}` of the security, used to predict the price change variance :math:`\Sigma=\Sigma_t`
            in case of non-constant variance, e.g. if the factor model is done on the security's return rather than
            P\&L.

        Returns
        -------
        sig2 : float
            Next step P\&L variance.

        """

        if self.riskDriverType == RiskDriverType.PnL:

            sig2 = self._get_sig2(factor)

        elif self.riskDriverType == RiskDriverType.Return:

            sig2_ret = self._get_sig2(factor)
            sig2 = price**2 * sig2_ret

        else:
            raise NameError('Invalid riskDriverType: ' + self.riskDriverType.value)

        return sig2

    def simulate(self, j_: int, t_: int, delta_stationary: int = 50):
        """
        Simulate all variables contained in the market.

        Parameters
        ----------
        j_: int
            Number of paths.
        t_ : int
            Length :math:`T` of each path.
        delta_stationary : int
            In order to get stationary simulations of the factor, it simulates paths of length
            :obj:`t_ + delta_stationary`, then it drops the first :obj:`delta_stationary`.

        """

        self._simulate_factor(j_, t_, delta_stationary)
        self._simulate_risk_driver_and_pnl_and_price()

    def simulate_market_trading(self, n: int, j_episodes: int, t_: int):
        """
        Simulate all variables contained in the market for training purposes.

        Parameters
        ----------
        n : int
            Batch on which the training is being done.
        j_episodes: int
            Number of paths.
        t_ : int
            Length :math:`T` of each path.

        """

        if self.simulations_training is None:
            self.simulations_training = {}

        self.simulate(j_=j_episodes, t_=t_)
        self.simulations_training[n] = copy.deepcopy(self.simulations)

    def _get_sig2(self, factor: float = None):

        riskDriverDynamicsType, parameters = self.marketDynamics.get_riskDriverDynamicsType_and_parameters()

        if riskDriverDynamicsType == RiskDriverDynamicsType.Linear:

            return parameters['sig2']

        elif riskDriverDynamicsType == RiskDriverDynamicsType.NonLinear:

            if factor is None:

                p = parameters['p']

                return p * parameters['sig2_0'] + (1 - p) * parameters['sig2_0']

            else:

                if factor < parameters['c']:
                    return parameters['sig2_0']
                else:
                    return parameters['sig2_1']

        else:
            raise NameError(f'Invalid riskDriverDynamicsType: {riskDriverDynamicsType.value}')

    def _simulate_factor(self, j_: int, t_: int, delta_stationary: int):

        factorDynamicsType = self.marketDynamics.factorDynamics.factorDynamicsType
        parameters = self.marketDynamics.factorDynamics.parameters

        factor, norm, t_stationary = self._initialize_factor_simulations(delta_stationary, j_, t_)

        if factorDynamicsType == FactorDynamicsType.AR:

            self._simulate_factor_ar(factor, norm, parameters, t_stationary)

        elif factorDynamicsType == FactorDynamicsType.SETAR:

            self._simulate_factor_setar(factor, norm, parameters, t_stationary)

        elif factorDynamicsType == FactorDynamicsType.GARCH:

            self._simulate_factor_garch(factor, j_, norm, parameters, t_stationary)

        elif factorDynamicsType == FactorDynamicsType.TARCH:

            self._simulate_factor_tarch(factor, j_, norm, parameters, t_stationary)

        elif factorDynamicsType == FactorDynamicsType.AR_TARCH:

            self._simulate_factor_ar_tarch(factor, j_, norm, parameters, t_stationary)

        else:
            raise NameError('Invalid factorDynamicsType')

        self.simulations['factor'] = factor[:, -t_:]

    def _simulate_factor_ar(self, factor, norm, parameters, t_stationary):
        B, mu, sig2 = self._get_linear_params(parameters)
        ind = range(factor.shape[0])
        for t in range(1, t_stationary):
            self._next_step_factor_linear(B, factor, ind, mu, norm, sig2, t)

    def _simulate_factor_setar(self, factor, norm, parameters, t_stationary):
        B_0, B_1, c, mu_0, mu_1, sig2_0, sig2_1 = self._get_threshold_params(parameters)
        for t in range(1, t_stationary):
            ind_0, ind_1 = self._get_threshold_indexes(c, factor, t)

            self._next_step_factor_linear(B_0, factor, ind_0, mu_0, norm, sig2_0, t)
            self._next_step_factor_linear(B_1, factor, ind_1, mu_1, norm, sig2_1, t)

    def _simulate_factor_garch(self, factor, j_, norm, parameters, t_stationary):
        alpha, beta, mu, omega = self._get_garch_parameters(parameters)
        epsi, sig = self._initialize_arch_simulations(j_, omega, t_stationary)
        for t in range(1, t_stationary):
            sig2 = self._get_next_step_sig2_arch(alpha, beta, epsi, omega, sig, t)
            self._get_next_step_arch(epsi, factor, mu, norm, sig, sig2, t)
        self.simulations['sig'] = sig

    def _simulate_factor_tarch(self, factor, j_, norm, parameters, t_stationary):
        alpha, beta, c, gamma, mu, omega = self._get_tarch_parameters(parameters)
        epsi, sig = self._initialize_arch_simulations(j_, omega, t_stationary)
        for t in range(1, t_stationary):
            sig2 = self._get_next_step_sig2_tarch(alpha, beta, c, epsi, gamma, omega, sig, t)
            self._get_next_step_arch(epsi, factor, mu, norm, sig, sig2, t)
        self.simulations['sig'] = sig

    def _simulate_factor_ar_tarch(self, factor, j_, norm, parameters, t_stationary):
        B, alpha, beta, c, gamma, mu, omega = self._get_ar_tarch_parameters(parameters)
        epsi, sig = self._initialize_arch_simulations(j_, omega, t_stationary)
        for t in range(1, t_stationary):
            sig2 = self._get_next_step_sig2_tarch(alpha, beta, c, epsi, gamma, omega, sig, t)

            self._get_next_step_ar_arch(B, epsi, factor, mu, norm, sig, sig2, t)
        self.simulations['sig'] = sig

    def _simulate_risk_driver_and_pnl_and_price(self):

        self._simulate_risk_driver()
        self._simulate_pnl()
        self._simulate_price()

    def _simulate_risk_driver(self):

        riskDriverDynamicsType, parameters = self.marketDynamics.get_riskDriverDynamicsType_and_parameters()
        self.simulations['risk-driver'] = self._simulate_risk_driver_impl(riskDriverDynamicsType, parameters)

    def _simulate_risk_driver_impl(self, riskDriverDynamicsType, parameters):

        factor, norm, risk_driver, t_ = self._initialize_risk_driver_simulations()

        if riskDriverDynamicsType == RiskDriverDynamicsType.Linear:

            self._simulate_risk_driver_linear(risk_driver, factor, norm, parameters)

        elif riskDriverDynamicsType == RiskDriverDynamicsType.NonLinear:

            self._simulate_risk_driver_non_linear(risk_driver, factor, norm, parameters, t_)

        else:
            raise NameError('Invalid riskDriverDynamicsType')

        return risk_driver

    def _simulate_risk_driver_linear(self, risk_driver, factor, norm, parameters):
        sig2 = parameters['sig2']
        risk_driver[:, 1:] = self._next_step_risk_driver(factor[:, :-1]) + np.sqrt(sig2) * norm[:, 1:]

    def _simulate_risk_driver_non_linear(self, risk_driver, factor, norm, parameters, t_):
        c = parameters['c']
        sig2_0 = parameters['sig2_0']
        sig2_1 = parameters['sig2_1']
        for t in range(1, t_):
            ind_0, ind_1 = self._get_threshold_indexes(c, factor, t)
            self._get_next_step_risk_driver(risk_driver, factor, ind_0, norm, sig2_0, t)
            self._get_next_step_risk_driver(risk_driver, factor, ind_1, norm, sig2_1, t)

    def _simulate_pnl(self):

        if self.riskDriverType == RiskDriverType.PnL:

            self.simulations['pnl'] = self.simulations['risk-driver']

        elif self.riskDriverType == RiskDriverType.Return:

            ret = self.simulations['risk-driver']

            self.simulations['pnl'] = self.start_price * np.cumprod(1 + ret, axis=1) / (1 + ret) * ret

        else:
            raise NameError('Invalid riskDriverType: ' + self.riskDriverType.value)

        self.simulations['average_past_pnl'] =\
            np.array(pd.DataFrame(self.simulations['pnl']).rolling(window=self.financialTimeSeries.window,
                                                                   min_periods=1,
                                                                   axis=1).mean())

    def _simulate_price(self):

        self._get_price_from_pnl()

    def _initialize_risk_driver_simulations(self):
        factor = self.simulations['factor']
        j_, t_ = factor.shape
        risk_driver = np.zeros((j_, t_))
        norm = np.random.randn(j_, t_)
        return factor, norm, risk_driver, t_

    @staticmethod
    def _initialize_factor_simulations(delta_stationary, j_, t_):
        t_stationary = t_ + delta_stationary
        factor = np.zeros((j_, t_stationary))
        norm = np.random.randn(j_, t_stationary)
        return factor, norm, t_stationary

    @staticmethod
    def _initialize_arch_simulations(j_, omega, t_stationary):
        sig = np.zeros((j_, t_stationary))
        sig[:, 0] = np.sqrt(omega)
        epsi = np.zeros((j_, t_stationary))
        return epsi, sig

    @staticmethod
    def _get_next_step_sig2_arch(alpha, beta, epsi, omega, sig, t):

        sig2 = omega + alpha * (epsi[:, t - 1] ** 2) + beta * (sig[:, t - 1] ** 2)

        return sig2

    def _get_next_step_sig2_tarch(self, alpha, beta, c, epsi, gamma, omega, sig, t):
        sig2 = self._get_next_step_sig2_arch(alpha, beta, epsi, omega, sig, t)
        sig2[epsi[:, t - 1] < c] += gamma * (epsi[epsi[:, t - 1] < c, t - 1] ** 2)
        return sig2

    @staticmethod
    def _get_next_step_arch(epsi, factor, mu, norm, sig, sig2, t):

        sig[:, t] = np.sqrt(sig2)
        epsi[:, t] = sig[:, t] * norm[:, t]
        factor[:, t] = mu + factor[:, t - 1] + epsi[:, t]

    def _get_next_step_ar_arch(self, B, epsi, factor, mu, norm, sig, sig2, t):
        self._get_next_step_arch(epsi, factor, mu, norm, sig, sig2, t)
        factor[:, t] += (B - 1) * factor[:, t - 1]

    def _get_ar_tarch_parameters(self, parameters):
        alpha, beta, c, gamma, mu, omega = self._get_tarch_parameters(parameters)
        B = parameters['B']
        return B, alpha, beta, c, gamma, mu, omega

    def _get_tarch_parameters(self, parameters):
        alpha, beta, mu, omega = self._get_garch_parameters(parameters)
        gamma = parameters['gamma']
        c = parameters['c']
        return alpha, beta, c, gamma, mu, omega

    @staticmethod
    def _get_garch_parameters(parameters):
        mu = parameters['mu']
        omega = parameters['omega']
        alpha = parameters['alpha']
        beta = parameters['beta']
        return alpha, beta, mu, omega

    @staticmethod
    def _get_threshold_indexes(c, f, t):
        ind_0 = f[:, t - 1] < c
        ind_1 = f[:, t - 1] >= c
        return ind_0, ind_1

    @staticmethod
    def _get_threshold_params(parameters):
        c = parameters['c']
        mu_0 = parameters['mu_0']
        B_0 = parameters['B_0']
        sig2_0 = parameters['sig2_0']
        mu_1 = parameters['mu_1']
        B_1 = parameters['B_1']
        sig2_1 = parameters['sig2_1']
        return B_0, B_1, c, mu_0, mu_1, sig2_0, sig2_1

    @staticmethod
    def _get_linear_params(parameters):
        mu = parameters['mu']
        B = parameters['B']
        sig2 = parameters['sig2']
        return B, mu, sig2

    def _get_price_from_pnl(self):

        self.simulations['price'] = self.start_price + np.cumsum(self.simulations['pnl'], axis=1)

    def _get_next_step_risk_driver(self, risk_driver, f, ind_0, norm, sig2_0, t):
        risk_driver[ind_0, t] = np.vectorize(self._next_step_risk_driver,
                                             otypes=[float])(f[ind_0, t - 1]) + np.sqrt(sig2_0) * norm[ind_0, t]

    @staticmethod
    def _next_step_factor_linear(B, f, ind, mu, norm, sig2, t):
        f[ind, t] = mu + B * f[ind, t - 1] + np.sqrt(sig2) * norm[ind, t]

    def _set_market_attributes(self):

        self._set_market_id()
        self._set_riskDriverType()
        self._set_start_price()
        self.simulations = {}
        self.ticker = self.financialTimeSeries.ticker

    def _set_riskDriverType(self):

        self.riskDriverType = self.marketDynamics.riskDriverType

    def _set_market_id(self):

        riskDriverDynamicsType = self.marketDynamics.riskDriverDynamics.riskDriverDynamicsType
        factorDynamicsType = self.marketDynamics.factorDynamics.factorDynamicsType
        self.market_id = f'{riskDriverDynamicsType.value}-{factorDynamicsType.value}'

    def _set_start_price(self):

        self.start_price = self.marketDynamics.start_price


def instantiate_market(riskDriverDynamicsType: RiskDriverDynamicsType,
                       factorDynamicsType: FactorDynamicsType,
                       ticker: str,
                       riskDriverType: RiskDriverType,
                       modeType: ModeType = ModeType.InSample):
    """
    Instantiates a :class:`~dynamic_trading.market_utils.market.Market` object.

    Parameters
    ----------
    riskDriverDynamicsType : :class:`~dynamic_trading.enums.enums.RiskDriverDynamicsType`
        Dynamics type for the risk-driver, see :class:`~dynamic_trading.enums.enums.RiskDriverDynamicsType` for more details.
    factorDynamicsType : :class:`~dynamic_trading.enums.enums.FactorDynamicsType`
        Dynamics type for the factor, see :class:`~dynamic_trading.enums.enums.FactorDynamicsType` for more details.
    ticker : str
        An ID to identify the traded security. If this ID is present in the list of available securities, the code
        will read its time series from the source data. Otherwise, it will try to download the time series from
        Yahoo finance via the :obj:`yfinance` module.
    riskDriverType : :class:`~dynamic_trading.enums.enums.RiskDriverType`
        Type of risk-driver, see :class:`~dynamic_trading.enums.enums.RiskDriverType` for more details.
    modeType : :class:`~dynamic_trading.enums.enums.ModeType`
        Whether to use the time series for in-sample or out-of-sample purposes. See :class:`~dynamic_trading.enums.enums.ModeType` for more details.

    Returns
    -------
    market : Market
        Instance of :class:`~dynamic_trading.market_utils.market.Market` object with given attributes. Refer to the documentation of
            :class:`~dynamic_trading.market_utils.market.Market` for more details.
    """

    # Instantiate financialTimeSeries
    financialTimeSeries = FinancialTimeSeries(ticker, modeType=modeType)

    # Instantiate dynamics
    riskDriverDynamics = RiskDriverDynamics(riskDriverDynamicsType)
    factorDynamics = FactorDynamics(factorDynamicsType)

    # Read calibrated parameters
    riskDriverDynamics.set_parameters_from_file(ticker, riskDriverType)
    factorDynamics.set_parameters_from_file(ticker, riskDriverType)

    # Set dynamics
    marketDynamics = MarketDynamics(riskDriverDynamics, factorDynamics)

    # Instantiate market
    market = Market(financialTimeSeries, marketDynamics)

    return market


def read_trading_parameters_market():
    """
    Reads trading parameters from input file.

    Returns
    -------
    ticker : str
        An ID to identify the traded security. If this ID is present in the list of available securities, the code
        will read its time series from the source data. Otherwise, it will try to download the time series from
        Yahoo finance via the :obj:`yfinance` module.
    riskDriverDynamicsType : :class:`~dynamic_trading.enums.enums.RiskDriverDynamicsType`
        Dynamics type for the risk-driver, see :class:`~dynamic_trading.enums.enums.RiskDriverDynamicsType` for more details.
    factorDynamicsType : :class:`~dynamic_trading.enums.enums.FactorDynamicsType`
        Dynamics type for the factor, see :class:`~dynamic_trading.enums.enums.FactorDynamicsType` for more details.
    riskDriverType : :class:`~dynamic_trading.enums.enums.RiskDriverType`
        Type of risk-driver, see :class:`~dynamic_trading.enums.enums.RiskDriverType` for more details.

    """

    filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename += '/resources/data/data_source/settings.csv'
    df_trad_params = pd.read_csv(filename, index_col=0)
    ticker = df_trad_params.loc['ticker'][0]
    riskDriverDynamicsType = RiskDriverDynamicsType(df_trad_params.loc['riskDriverDynamicsType'][0])
    factorDynamicsType = FactorDynamicsType(df_trad_params.loc['factorDynamicsType'][0])
    riskDriverType = RiskDriverType(df_trad_params.loc['riskDriverType'][0])

    return ticker, riskDriverDynamicsType, factorDynamicsType, riskDriverType
