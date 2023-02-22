from typing import Union

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from tqdm import tqdm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.ar_model import AutoReg

from dynamic_trading.enums.enums import RiskDriverDynamicsType, FactorDynamicsType
from dynamic_trading.gen_utils.utils import get_available_futures_tickers
from dynamic_trading.market_utils.financial_time_series import FinancialTimeSeries

import warnings

warnings.filterwarnings('ignore', message='A date index has been provided, but it has no associated frequency')


class DynamicsCalibrator:
    """
    General class for executing the calibration of all the models considered for a given security.

    Attributes
    ----------
    financialTimeSeries : :class:`~dynamic_trading.market_utils.financial_time_series.FinancialTimeSeries`
        The financial time series to be calibrated. Refer to
        :class:`~dynamic_trading.market_utils.financial_time_series.FinancialTimeSeries` for more details.
    riskDriverType : :class:`~dynamic_trading.enums.enums.RiskDriverType`
        Can be RiskDriverType.PnL (the security's model is on PnLs) or RiskDriverType.Return (the security's model is on
        returns).
    all_dynamics_param_dict : dict
        containing all the fitted parameters.
    all_dynamics_model_dict : dict
        Dictionary containing all the fitted models.
    all_dynamics_resid_dict : dict
        Dictionary containing all the fitted residuals.

    """

    def __init__(self):
        """
        Class constructor.

        """

        self.financialTimeSeries = None
        self.all_dynamics_param_dict = {}
        self.all_dynamics_model_dict = {}
        self.all_dynamics_resid_dict = {}

    def fit_all_dynamics_param(self, financialTimeSeries: FinancialTimeSeries,
                               scale: float = 1,
                               scale_f: float = 1,
                               c: float = None):
        """
        Fits all the dynamics considered on the given financial time series.

        Parameters
        ----------
        financialTimeSeries : FinancialTimeSeries
            Financial time series to fit.
        scale : float
            Factor for rescaling the security time series.
        scale_f : float
            Factor for rescaling the factor time series.
        c : float
            Threshold for threshold models.

        """

        self.financialTimeSeries = financialTimeSeries
        self._fit_all_risk_driver_dynamics_param(scale, c)
        self._fit_all_factor_dynamics_param(scale_f, c)
        self._set_riskDriverType()
        self.print_results()

    def get_params_dict(self, var_type: str, dynamicsType: Union[RiskDriverDynamicsType, FactorDynamicsType]) -> dict:
        """
        For a given var_type = ('risk-driver', 'factor') and dynamicsType, returns the dict containing the fitted
        parameters for that model.

        Parameters
        ----------
        var_type : str
            Can be 'risk-driver', 'factor'.
        dynamicsType : Union[RiskDriverDynamicsType, FactorDynamicsType]
            An instance of RiskDriverDynamicsType or FactorDynamicsType, depending on var_type.

        Returns
        -------
        param_dict : dict
            Dictionary with parameters.

        """

        param_dict = self.all_dynamics_param_dict[var_type][dynamicsType]

        return param_dict

    def print_results(self):
        """
        Prints the results of the models fitting for the risk-driver and the factor. Results are stored in
        resources/data/financial_time_series_data and in resources/reports/calibrations.

        """

        self._print_results_impl('risk-driver')
        self._print_results_impl('factor')

    def _set_riskDriverType(self):

        self.riskDriverType = self.financialTimeSeries.riskDriverType

    def _fit_all_risk_driver_dynamics_param(self, scale: float, c: float):

        self.all_dynamics_param_dict['risk-driver'] = {}
        self.all_dynamics_model_dict['risk-driver'] = {}
        self.all_dynamics_resid_dict['risk-driver'] = {}

        for riskDriverDynamicsType in RiskDriverDynamicsType:
            try:
                self._fit_risk_driver_dynamics_param(riskDriverDynamicsType, scale, c)
            except RuntimeError as e:
                raise Exception(f'Could not fit {riskDriverDynamicsType} for {self.financialTimeSeries.ticker}') from e

    def _fit_all_factor_dynamics_param(self, scale_f: float, c: float):

        self.all_dynamics_param_dict['factor'] = {}
        self.all_dynamics_model_dict['factor'] = {}
        self.all_dynamics_resid_dict['factor'] = {}

        for factorDynamicsType in FactorDynamicsType:
            try:
                self._fit_factor_dynamics_param(factorDynamicsType, scale_f, c)
            except RuntimeError as e:
                raise Exception(f'Could not fit {factorDynamicsType} for {self.financialTimeSeries.ticker}') from e

    def _fit_risk_driver_dynamics_param(self, riskDriverDynamicsType: RiskDriverDynamicsType, scale: float, c: float):

        if riskDriverDynamicsType == RiskDriverDynamicsType.Linear:

            self._execute_general_linear_regression(tgt_key=riskDriverDynamicsType, var_type='risk-driver', scale=scale)

        elif riskDriverDynamicsType == RiskDriverDynamicsType.NonLinear:

            self._execute_general_threshold_regression(tgt_key=riskDriverDynamicsType, var_type='risk-driver',
                                                       scale=scale, c=c)

        else:
            raise NameError(f'Invalid riskDriverDynamicsType: {riskDriverDynamicsType.value}')

    def _fit_factor_dynamics_param(self, factorDynamicsType: FactorDynamicsType, scale_f: float, c: float):

        if factorDynamicsType == FactorDynamicsType.AR:

            self._execute_general_linear_regression(tgt_key=factorDynamicsType, var_type='factor', scale=scale_f)

        elif factorDynamicsType == FactorDynamicsType.SETAR:

            self._execute_general_threshold_regression(tgt_key=factorDynamicsType, var_type='factor', scale=scale_f,
                                                       c=c)

        elif factorDynamicsType in (FactorDynamicsType.GARCH, FactorDynamicsType.TARCH, FactorDynamicsType.AR_TARCH):

            self._execute_garch_tarch_ar_tarch(factorDynamicsType, scale_f)

        else:

            raise NameError(f'Invalid factorDynamicsType: {factorDynamicsType.value}')

    def _execute_general_linear_regression(self, tgt_key, var_type: str, scale: float):

        self._check_var_type(var_type)

        self.all_dynamics_param_dict[var_type][tgt_key] = {}

        # regression data
        df_reg = self._prepare_df_reg(var_type)

        ind = df_reg.index

        # regression
        B, mu, model_fit, sig2, abs_epsi_autocorr, epsi = self._execute_ols(df_reg, ind, scale, var_type)

        self.all_dynamics_param_dict[var_type][tgt_key]['mu'] = mu
        self.all_dynamics_param_dict[var_type][tgt_key]['B'] = B
        self.all_dynamics_param_dict[var_type][tgt_key]['sig2'] = sig2
        self.all_dynamics_param_dict[var_type][tgt_key]['abs_epsi_autocorr'] = abs_epsi_autocorr

        self.all_dynamics_model_dict[var_type][tgt_key] = [model_fit]

        self.all_dynamics_resid_dict[var_type][tgt_key] = epsi

    def _execute_general_threshold_regression(self, tgt_key, var_type: str, scale: float, c: float):

        self._check_var_type(var_type)

        self.all_dynamics_param_dict[var_type][tgt_key] = {}

        # regression data
        df_reg = self._prepare_df_reg(var_type)

        if c is None:
            c = df_reg['factor'].mean()

        ind_0 = df_reg['factor'] < c
        ind_1 = df_reg['factor'] >= c
        ind_lst = [ind_0, ind_1]
        p = ind_0.sum() / np.array(ind_lst).sum()

        if p < 0 or p > 1:
            raise NameError('p should be between 0 and 1')

        self.all_dynamics_param_dict[var_type][tgt_key]['c'] = c
        self.all_dynamics_param_dict[var_type][tgt_key]['p'] = p

        model_lst = []
        epsi_lst = []

        for i in range(len(ind_lst)):
            ind = ind_lst[i]

            # regression
            B, mu, model_fit, sig2, abs_epsi_autocorr, epsi = self._execute_ols(df_reg, ind, scale, var_type)

            self.all_dynamics_param_dict[var_type][tgt_key]['mu_%d' % i] = mu
            self.all_dynamics_param_dict[var_type][tgt_key]['B_%d' % i] = B
            self.all_dynamics_param_dict[var_type][tgt_key]['sig2_%d' % i] = sig2
            model_lst.append(model_fit)
            epsi_lst.append(epsi)

        self.all_dynamics_model_dict[var_type][tgt_key] = model_lst

        all_epsi = pd.concat(epsi_lst, axis=0)
        all_epsi.sort_index(inplace=True)

        self.all_dynamics_param_dict[var_type][tgt_key]['abs_epsi_autocorr'] =\
            [np.abs(all_epsi).autocorr(lag) for lag in range(20)]
        self.all_dynamics_resid_dict[var_type][tgt_key] = all_epsi

    def _execute_ols(self, df_reg: pd.DataFrame, ind: pd.Index, scale: float, var_type: str):

        if var_type == 'risk-driver':
            model_fit = OLS(df_reg['risk-driver'].loc[ind], add_constant(df_reg['factor'].loc[ind])).fit(disp=0)
            B, mu, sig2 = self._extract_B_mu_sig2_from_reg(model_fit, scale)
        else:
            model_fit = AutoReg(df_reg['factor'].loc[ind], lags=1).fit()
            B, mu, sig2 = self._extract_B_mu_sig2_from_auto_reg(model_fit, scale)

        epsi = model_fit.resid
        abs_epsi = np.abs(epsi)
        abs_epsi_autocorr = [abs_epsi.autocorr(lag) for lag in range(20)]

        return B, mu, model_fit, sig2, abs_epsi_autocorr, epsi

    def _execute_garch_tarch_ar_tarch(self, factorDynamicsType: FactorDynamicsType, scale_f: float):

        self.all_dynamics_param_dict['factor'][factorDynamicsType] = {}
        self.all_dynamics_model_dict['factor'][factorDynamicsType] = {}

        if factorDynamicsType in (FactorDynamicsType.GARCH, FactorDynamicsType.TARCH):

            df_model = self._prepare_df_model_factor_diff()

            if factorDynamicsType == FactorDynamicsType.GARCH:
                from arch import arch_model
                model = arch_model(df_model, p=1, q=1, rescale=False)
            else:
                from arch import arch_model
                model = arch_model(df_model, p=1, o=1, q=1, rescale=False)

            model_fit = model.fit(disp=False)
            params = model_fit.params.copy()

            abs_epsi_autocorr, epsi = self._get_arch_abs_epsi_autocorr(model_fit, scale_f)

            if factorDynamicsType == FactorDynamicsType.GARCH:
                alpha, beta, mu, omega = self._extract_garch_params_from_model_fit(params, scale_f)
                self._set_garch_params(alpha, beta, factorDynamicsType, mu, omega, abs_epsi_autocorr, epsi)
            else:
                alpha, beta, gamma, mu, omega = self._extract_tarch_params_from_model_fit(params, scale_f)
                self._set_tarch_params(alpha, beta, factorDynamicsType, gamma, mu, omega, abs_epsi_autocorr, epsi)

        elif factorDynamicsType == FactorDynamicsType.AR_TARCH:

            from arch.univariate import ARX, GARCH

            df_model = self._prepare_df_model_factor()

            model = ARX(df_model, lags=1, rescale=False)
            model.volatility = GARCH(p=1, o=1, q=1)

            model_fit = model.fit(disp=False)
            params = model_fit.params.copy()
            params.rename(index={'Const': 'mu'}, inplace=True)

            abs_epsi_autocorr, epsi = self._get_arch_abs_epsi_autocorr(model_fit, scale_f)

            B, alpha, beta, gamma, mu, omega = self._extract_ar_tarch_params_from_model_fit(params, scale_f)

            self._set_ar_tarch_params(B, alpha, beta, factorDynamicsType, gamma, mu, omega, abs_epsi_autocorr, epsi)

        else:
            raise NameError(f'Invalid factorDynamicsType: {factorDynamicsType.value}')

        self.all_dynamics_model_dict['factor'][factorDynamicsType] = [model_fit]

    @staticmethod
    def _get_arch_abs_epsi_autocorr(model_fit, scale_f):
        resid = model_fit.resid / scale_f
        sigma = model_fit.conditional_volatility / scale_f
        epsi = np.divide(resid, sigma)
        abs_epsi = np.abs(epsi)
        abs_epsi_autocorr = [abs_epsi.autocorr(lag) for lag in range(20)]
        return abs_epsi_autocorr, epsi

    def _prepare_df_reg(self, var_type: str):

        if var_type == 'risk-driver':
            df_reg = self._prepare_df_model_risk_driver()
        else:
            df_reg = self._prepare_df_model_factor()
        return df_reg

    def _prepare_df_model_risk_driver(self):

        df_model = self.financialTimeSeries.time_series[['factor', 'risk-driver']].copy()
        df_model['risk-driver'] = df_model['risk-driver'].shift(-1)
        df_model.dropna(inplace=True)

        return df_model

    def _prepare_df_model_factor(self):

        df_model = self.financialTimeSeries.time_series['factor'].copy()
        df_model = df_model.to_frame()
        df_model.dropna(inplace=True)

        return df_model

    def _prepare_df_model_factor_diff(self):

        df_model = self.financialTimeSeries.time_series['factor'].diff().dropna().copy()

        return df_model

    @staticmethod
    def _extract_B_mu_sig2_from_reg(model_fit, scale: float):

        B = model_fit.params['factor']
        mu = model_fit.params['const'] / scale
        sig2 = model_fit.mse_resid / scale ** 2

        return B, mu, sig2

    @staticmethod
    def _extract_B_mu_sig2_from_auto_reg(auto_reg, scale_f: float):

        B = auto_reg.params.iloc[1]
        mu = auto_reg.params.iloc[0] / scale_f
        sig2 = auto_reg.sigma2 / scale_f ** 2

        return B, mu, sig2

    def _extract_tarch_params_from_model_fit(self, params: pd.Series, scale_f: float):

        alpha, beta, mu, omega = self._extract_garch_params_from_model_fit(params, scale_f)
        gamma = params['gamma[1]'] / scale_f ** 2

        return alpha, beta, gamma, mu, omega

    @staticmethod
    def _extract_garch_params_from_model_fit(params, scale_f):

        mu = params['mu'] / scale_f
        omega = params['omega'] / scale_f ** 2
        alpha = params['alpha[1]'] / scale_f ** 2
        beta = params['beta[1]'] / scale_f ** 2

        return alpha, beta, mu, omega

    def _extract_ar_tarch_params_from_model_fit(self, params: pd.Series, scale_f: float):

        alpha, beta, gamma, mu, omega = self._extract_tarch_params_from_model_fit(params, scale_f)
        B = params['factor[1]']

        return B, alpha, beta, gamma, mu, omega

    def _set_garch_params(self, alpha: float, beta: float, factorDynamicsType: FactorDynamicsType, mu: float,
                          omega: float, abs_epsi_autocorr: list, epsi: pd.Series):

        self.all_dynamics_param_dict['factor'][factorDynamicsType]['mu'] = mu
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['omega'] = omega
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['alpha'] = alpha
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['beta'] = beta
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['abs_epsi_autocorr'] = abs_epsi_autocorr

        self.all_dynamics_resid_dict['factor'][factorDynamicsType] = epsi

    def _set_tarch_params(self, alpha: float, beta: float, factorDynamicsType: FactorDynamicsType, gamma: float,
                          mu: float, omega: float, abs_epsi_autocorr: list, epsi: pd.Series):

        self._set_garch_params(alpha, beta, factorDynamicsType, mu, omega, abs_epsi_autocorr, epsi)
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['gamma'] = gamma
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['c'] = 0

    def _set_ar_tarch_params(self, B: float, alpha: float, beta: float, factorDynamicsType: FactorDynamicsType,
                             gamma: float, mu: float, omega: float, abs_epsi_autocorr: list, epsi: pd.Series):

        self._set_tarch_params(alpha, beta, factorDynamicsType, gamma, mu, omega, abs_epsi_autocorr, epsi)
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['B'] = B

    def _print_results_impl(self, var_type: str):

        self._check_var_type(var_type)
        ticker = self.financialTimeSeries.ticker
        riskDriverType = self.riskDriverType

        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += f'/resources/data/financial_time_series_data/financial_time_series_calibrations/{ticker}'
        filename += f'-riskDriverType-{riskDriverType.value}-{var_type}-calibrations.xlsx'

        writer = pd.ExcelWriter(filename)
        workbook = writer.book

        df_riskDriverType = pd.DataFrame(data=[riskDriverType.value], columns=['riskDriverType'])
        df_riskDriverType.to_excel(writer, sheet_name='riskDriverType', index=False)

        if var_type == 'risk-driver':
            df_start_price = pd.DataFrame(data=[self.financialTimeSeries.info.loc['start_price'][0]],
                                          columns=['start_price'])
            df_start_price.to_excel(writer, sheet_name='start_price', index=False)

        for dynamicsType, param_dict in self.all_dynamics_param_dict[var_type].items():

            try:
                # parameters
                worksheet = workbook.add_worksheet(dynamicsType.value)
                writer.sheets[dynamicsType.value] = worksheet
                df_params_out = pd.DataFrame.from_dict(data=param_dict,
                                                       orient='index',
                                                       columns=['param'])
                df_params_out.to_excel(writer, sheet_name=dynamicsType.value)

                # reports
                for i in range(len(self.all_dynamics_model_dict[var_type][dynamicsType])):
                    model = self.all_dynamics_model_dict[var_type][dynamicsType][i]
                    filename = self._set_report_filename(dynamicsType, i, var_type)

                    with open(filename, 'w+') as fh:
                        fh.write(model.summary().as_text())

            except:
                print(f'Could not write report for {var_type}, {dynamicsType}')

        writer.close()

    def _set_report_filename(self, dynamicsType, i: int, var_type: str):

        riskDriverType = self.riskDriverType

        if dynamicsType in (RiskDriverDynamicsType.Linear, FactorDynamicsType.AR, FactorDynamicsType.GARCH,
                            FactorDynamicsType.TARCH, FactorDynamicsType.AR_TARCH):
            filename = os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))) + '/resources/reports/calibrations/' +\
                       self.financialTimeSeries.ticker + '-riskDriverType-' + riskDriverType.value +\
                       '-' + var_type + '-' + dynamicsType.value + '.txt'
        elif dynamicsType in (RiskDriverDynamicsType.NonLinear, FactorDynamicsType.SETAR):
            filename = os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))) + '/resources/reports/calibrations/' +\
                       self.financialTimeSeries.ticker + '-riskDriverType-' + riskDriverType.value +\
                       '-' + var_type + '-' + dynamicsType.value + str(i) + '.txt'
        else:
            raise NameError('Invalid dynamicsType: ' + dynamicsType.value)

        return filename

    @staticmethod
    def _check_var_type(var_type: str):

        if var_type not in ('risk-driver', 'factor'):
            raise NameError('var_type must be equal to risk-driver or factor')


class AllSeriesDynamicsCalibrator:
    """
    General class for executing the calibration of all the models considered for all the considered securities.

    """

    def __init__(self):
        """
        Class constructor.

        """

        self._all_series_dynamics_calibrators = {}
        self._best_factorDynamicsType = {}
        self._best_factorDynamicsType_resid = {}
        self._non_best_factorDynamicsType_resid = {}
        self._average_price_per_contract = {}
        self._std_price_changes = {}

    def fit_all_series_dynamics(self):
        """
        Fits all dynamics for all securities.

        """

        for ticker in tqdm(get_available_futures_tickers(), 'Fitting all time series'):
            self._set_dynamicsCalibrator(ticker)
            self._get_best_factorDynamicsType_and_resid(ticker)

    def print_all_series_dynamics_results(self):
        """
        Prints the results of the models fitting for all the risk-drivers and the factors. Results are stored in
        resources/data/financial_time_series_data and in resources/reports/calibrations.

        """

        self._print_financial_time_series_summaries()
        self._plot_financial_time_series()
        self._plot_residuals()
        self._print_residuals_analysis()
        self._print_prices_and_stds()
        self._print_calibration_results()

    def _print_financial_time_series_summaries(self):

        ll = []
        for ticker, dynamicsCalibrator in self._all_series_dynamics_calibrators.items():
            time_series = dynamicsCalibrator.financialTimeSeries.time_series
            start_date = time_series.index[0]
            end_date = time_series.index[-1]
            price_average = time_series[ticker].mean()
            price_std = time_series[ticker].std()
            price_min = time_series[ticker].min()
            price_max = time_series[ticker].max()
            pnl_average = time_series['pnl'].mean()
            pnl_std = time_series['pnl'].std()
            pnl_min = time_series['pnl'].min()
            pnl_max = time_series['pnl'].max()

            ll.append([ticker, start_date, end_date,
                       price_average, price_std, price_min, price_max,
                       pnl_average, pnl_std, pnl_min, pnl_max])

        df_out = pd.DataFrame(data=ll, columns=['ticker', 'start_date', 'end_date',
                                                'price_average', 'price_std', 'price_min', 'price_max',
                                                'pnl_average', 'pnl_std', 'pnl_min', 'pnl_max'])
        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += f'/resources/reports/calibrations/assets_summaries.xlsx'
        df_out.to_excel(filename, sheet_name='assets_summaries', index=False)

    def _print_calibration_results(self):

        ll_model_summary = [['ticker', 'var_type', 'dynamicsType', 'i', 'aic', 'bic', 'nobs', 'ess', 'f_pvalue',
                             'fvalue', 'llf', 'mse_model', 'mse_resid', 'mse_total', 'rsquared', 'rsquared_adj']]

        for ticker, dynamicsCalibrator in self._all_series_dynamics_calibrators.items():

            for var_type in ('risk-driver', 'factor'):

                for dynamicsType, models_lst in dynamicsCalibrator.all_dynamics_model_dict[var_type].items():

                    for i in range(len(models_lst)):

                        current_ll_model_summary = []

                        aic = models_lst[i].aic
                        bic = models_lst[i].aic
                        nobs = models_lst[i].nobs

                        current_ll_model_summary += [ticker, var_type, dynamicsType.value, i, aic, bic, nobs]

                        if var_type == 'risk-driver':
                            ess = models_lst[i].ess
                            f_pvalue = models_lst[i].f_pvalue
                            fvalue = models_lst[i].fvalue
                            llf = models_lst[i].llf
                            mse_model = models_lst[i].mse_model
                            mse_resid = models_lst[i].mse_resid
                            mse_total = models_lst[i].mse_total
                            rsquared = models_lst[i].rsquared
                            rsquared_adj = models_lst[i].rsquared_adj

                            current_ll_model_summary += [ess, f_pvalue, fvalue, llf, mse_model, mse_resid, mse_total,
                                                         rsquared, rsquared_adj]
                        else:
                            current_ll_model_summary += [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                         np.nan, np.nan]

                        ll_model_summary.append(current_ll_model_summary)

        df_model_summary = pd.DataFrame(data=ll_model_summary[1:],
                                        columns=ll_model_summary[0])
        info = f'{self._riskDriverType.value}' +\
               f'-{self._factor_ticker}-{self._factorTransformationType.value}'
        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += f'/resources/reports/calibrations/models_summary_{info}.xlsx'
        df_model_summary.to_excel(filename, sheet_name='models_summary', index=False)

    def _plot_financial_time_series(self):

        for ticker, dynamicsCalibrator in self._all_series_dynamics_calibrators.items():
            financialTimeSeries = dynamicsCalibrator.financialTimeSeries
            time_series = financialTimeSeries.time_series[ticker]

            dpi = plt.rcParams['figure.dpi']

            fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
            plt.plot(time_series, label=ticker)
            plt.title(ticker + ' time series')
            plt.xlabel('Date')
            plt.ylabel('Value [$]')
            plt.savefig(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        + '/resources/figures/residuals/'
                        + ticker + '-time-series.png')
            plt.close(fig)

            fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
            factor = financialTimeSeries.time_series['factor']
            c = dynamicsCalibrator.all_dynamics_param_dict['risk-driver'][RiskDriverDynamicsType.NonLinear]['c']
            factor0 = factor[factor < c]
            factor1 = factor[factor >= c]
            risk_driver = financialTimeSeries.time_series['risk-driver']
            linear_model = dynamicsCalibrator.all_dynamics_model_dict['risk-driver'][RiskDriverDynamicsType.Linear][0]
            nonlinear_model0 =\
                dynamicsCalibrator.all_dynamics_model_dict['risk-driver'][RiskDriverDynamicsType.NonLinear][0]
            nonlinear_model1 =\
                dynamicsCalibrator.all_dynamics_model_dict['risk-driver'][RiskDriverDynamicsType.NonLinear][1]
            xlim = [np.quantile(factor, 0.01), np.quantile(factor, 0.99)]
            ylim = [np.quantile(risk_driver, 0.01), np.quantile(risk_driver, 0.99)]
            plt.scatter(factor, risk_driver, s=2)
            plt.plot(factor, linear_model.params['const'] + linear_model.params['factor'] * factor, label='Linear',
                     color='k')
            plt.plot(factor0, nonlinear_model0.params['const'] + nonlinear_model0.params['factor'] * factor0,
                     label='Non Linear', color='r')
            plt.plot(factor1, nonlinear_model1.params['const'] + nonlinear_model1.params['factor'] * factor1, color='r')
            plt.legend()
            plt.xlabel('Factor')
            plt.ylabel(f'{financialTimeSeries.riskDriverType.value}')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.savefig(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        + '/resources/figures/residuals/'
                        + ticker + '-prediction.png')
            plt.close(fig)

    def _print_residuals_analysis(self):

        ll = []
        abs_epsi_autocorr = None

        for ticker, d1 in self._best_factorDynamicsType_resid.items():

            factorDynamicsType = d1['factorDynamicsType']
            abs_epsi_autocorr = d1['abs_epsi_autocorr']

            ll.append([ticker, factorDynamicsType.value] + [a for a in abs_epsi_autocorr])

            for factorDynamicsType, d2 in self._non_best_factorDynamicsType_resid[ticker].items():
                abs_epsi_autocorr = d2['abs_epsi_autocorr']

                ll.append([ticker, factorDynamicsType.value] + [a for a in abs_epsi_autocorr])

        df_report = pd.DataFrame(data=ll,
                                 columns=['ticker', 'factorDynamicsType'] +
                                         ['autocorr_lag_%d' % a for a in range(len(abs_epsi_autocorr))])

        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += '/resources/reports/model_choice/residuals_analysis.csv'
        df_report.to_csv(filename, index=False)

    def _print_prices_and_stds(self):

        average_prices_per_contract_df = pd.DataFrame.from_dict(self._average_price_per_contract,
                                                                orient='index',
                                                                columns=['Average Price Per Contract'])
        std_price_changes_df = pd.DataFrame.from_dict(self._std_price_changes,
                                                      orient='index',
                                                      columns=['Standard Deviation of Price Changes'])
        out_dict = pd.concat([average_prices_per_contract_df, std_price_changes_df], axis=1)

        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += '/resources/reports/model_choice/prices_and_stds.csv'
        out_dict.to_csv(filename, index=True)

    def _plot_residuals(self):
        self._plot_best_residuals()
        self._plot_non_best_residuals()

    def _plot_best_residuals(self):

        for ticker, d in self._best_factorDynamicsType_resid.items():
            factorDynamicsType = d['factorDynamicsType']
            resid = d['resid']
            abs_epsi_autocorr = d['abs_epsi_autocorr']

            s = ticker + ', ' + factorDynamicsType.value

            dpi = plt.rcParams['figure.dpi']
            fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
            plt.subplot2grid((2, 1), (0, 0))
            plt.plot(resid, '.', alpha=0.5, markersize=2, label=s)
            plt.legend()
            plt.title('Residuals for ' + s)
            plt.xlabel('Date')
            plt.ylabel('Residual')

            ax2 = plt.subplot2grid((2, 1), (1, 0))
            plt.bar(range(len(abs_epsi_autocorr)), abs_epsi_autocorr)
            plt.title('Autocorrelation')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.xticks(range(len(abs_epsi_autocorr)))
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.))

            plt.tight_layout()

            plt.savefig(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        + '/resources/figures/residuals/'
                        + ticker + '-residuals-best-' + factorDynamicsType.value + '.png')

            plt.close(fig)

    def _plot_non_best_residuals(self):

        for ticker, d1 in self._non_best_factorDynamicsType_resid.items():

            for factorDynamicsType, d in d1.items():
                resid = d['resid']
                abs_epsi_autocorr = d['abs_epsi_autocorr']

                s = ticker + ', ' + factorDynamicsType.value

                dpi = plt.rcParams['figure.dpi']
                fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
                plt.subplot2grid((2, 1), (0, 0))
                plt.plot(resid, '.', alpha=0.5, markersize=2, label=s)
                plt.legend()
                plt.title('Residuals for ' + s)
                plt.xlabel('Date')
                plt.ylabel('Residual')

                ax2 = plt.subplot2grid((2, 1), (1, 0))
                plt.bar(range(len(abs_epsi_autocorr)), abs_epsi_autocorr)
                plt.title('Autocorrelation')
                plt.xlabel('Lag')
                plt.ylabel('Autocorrelation')
                plt.xticks(range(len(abs_epsi_autocorr)))
                ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.))

                plt.tight_layout()

                plt.savefig(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                            + '/resources/figures/residuals/'
                            + ticker + '-residuals-non-best-' + factorDynamicsType.value + '.png')

                plt.close(fig)

    def _get_best_factorDynamicsType_and_resid(self, ticker):

        self._best_factorDynamicsType_resid[ticker] = {}
        self._non_best_factorDynamicsType_resid[ticker] = {}

        all_factor_params = self._all_series_dynamics_calibrators[ticker].all_dynamics_param_dict['factor']
        all_factor_resids = self._all_series_dynamics_calibrators[ticker].all_dynamics_resid_dict['factor']

        factorDynamics_best = self._get_best_factorDynamicsType_and_resid_impl(all_factor_params, all_factor_resids,
                                                                               ticker)

        self._get_non_best_factorDynamicsType_and_resid(all_factor_params, all_factor_resids, factorDynamics_best,
                                                        ticker)

    def _get_best_factorDynamicsType_and_resid_impl(self, all_factor_params, all_factor_resids, ticker):

        abs_epsi_autocorr_best = None
        abs_epsi_autocorr_best_norm = np.inf

        factorDynamics_best = None
        for factorDynamicsType in FactorDynamicsType:

            abs_epsi_autocorr = all_factor_params[factorDynamicsType]['abs_epsi_autocorr']

            if np.linalg.norm(abs_epsi_autocorr) <= np.abs(abs_epsi_autocorr_best_norm):
                abs_epsi_autocorr_best = abs_epsi_autocorr
                abs_epsi_autocorr_best_norm = np.linalg.norm(abs_epsi_autocorr)

                factorDynamics_best = factorDynamicsType

        self._best_factorDynamicsType[ticker] = factorDynamics_best
        self._best_factorDynamicsType_resid[ticker]['factorDynamicsType'] = factorDynamics_best
        self._best_factorDynamicsType_resid[ticker]['resid'] = all_factor_resids[factorDynamics_best]
        self._best_factorDynamicsType_resid[ticker]['abs_epsi_autocorr'] = abs_epsi_autocorr_best
        return factorDynamics_best

    def _get_non_best_factorDynamicsType_and_resid(self, all_factor_params, all_factor_resids, factorDynamics_best,
                                                   ticker):

        non_best_factorDynamicsType = [factorDynamicsType for factorDynamicsType in FactorDynamicsType
                                       if factorDynamicsType != factorDynamics_best]

        for factorDynamicsType in non_best_factorDynamicsType:
            self._non_best_factorDynamicsType_resid[ticker][factorDynamicsType] =\
                {'resid': all_factor_resids[factorDynamicsType],
                 'abs_epsi_autocorr': all_factor_params[factorDynamicsType]['abs_epsi_autocorr']}

    def _set_dynamicsCalibrator(self, ticker):
        financialTimeSeries = FinancialTimeSeries(ticker=ticker)
        dynamicsCalibrator = DynamicsCalibrator()
        dynamicsCalibrator.fit_all_dynamics_param(financialTimeSeries)
        self._all_series_dynamics_calibrators[ticker] = dynamicsCalibrator
        self._riskDriverType = financialTimeSeries.riskDriverType
        self._factorComputationType = financialTimeSeries.factorComputationType
        self._factor_ticker = financialTimeSeries.factor_ticker
        self._factorTransformationType = financialTimeSeries.factorTransformationType

        self._average_price_per_contract[ticker] = financialTimeSeries.time_series[ticker].mean()
        self._std_price_changes[ticker] = financialTimeSeries.time_series[ticker].diff().std()


def _build_filename_calibrations(riskDriverType, ticker, var_type):
    filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) +\
               '/resources/data/financial_time_series_data/financial_time_series_calibrations/' +\
               ticker + '-riskDriverType-' + riskDriverType.value + '-' + var_type + '-calibrations.xlsx'

    return filename


def _get_futures_data_filename():
    filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename += '/resources/data/data_source/market_data/assets_data.xlsx'
    return filename


def _read_futures_data_by_ticker(filename, ticker):
    time_series = pd.read_excel(filename, sheet_name=ticker, index_col=0)
    return time_series


# ----------------------------------------- TESTS

if __name__ == '__main__':
    allSeriesDynamicsCalibrator = AllSeriesDynamicsCalibrator()
    allSeriesDynamicsCalibrator.fit_all_series_dynamics()
    allSeriesDynamicsCalibrator.print_all_series_dynamics_results()
