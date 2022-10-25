import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from tqdm import tqdm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.ar_model import AutoReg

from enums import RiskDriverDynamicsType, FactorDynamicsType
from market_utils.financial_time_series import FinancialTimeSeries

import warnings

warnings.filterwarnings('ignore', message='A date index has been provided, but it has no associated frequency')


class DynamicsCalibrator:

    def __init__(self):

        self.financialTimeSeries = None
        self.all_dynamics_param_dict = {}
        self.all_dynamics_model_dict = {}
        self.all_dynamics_resid_dict = {}

    def fit_all_dynamics_param(self, financialTimeSeries: FinancialTimeSeries,
                               scale: float = 1,
                               scale_f: float = 1,
                               c: float = None):

        self.financialTimeSeries = financialTimeSeries
        self._fit_all_risk_driver_dynamics_param(scale, c)
        self._fit_all_factor_dynamics_param(scale_f, c)
        self._set_riskDriverType()
        self.print_results()

    def _get_params_dict(self, var_type, dynamicsType):

        param_dict = self.all_dynamics_param_dict[var_type][dynamicsType]

        return param_dict

    def print_results(self):

        self._print_results_impl('risk-driver')
        self._print_results_impl('factor')

    def _set_riskDriverType(self):

        self.riskDriverType = self.financialTimeSeries.riskDriverType

    def _fit_all_risk_driver_dynamics_param(self, scale: float, c: float):

        self.all_dynamics_param_dict['risk-driver'] = {}
        self.all_dynamics_model_dict['risk-driver'] = {}
        self.all_dynamics_resid_dict['risk-driver'] = {}

        for riskDriverDynamicsType in RiskDriverDynamicsType:
            self._fit_risk_driver_dynamics_param(riskDriverDynamicsType, scale, c)

    def _fit_all_factor_dynamics_param(self, scale_f: float, c: float):

        self.all_dynamics_param_dict['factor'] = {}
        self.all_dynamics_model_dict['factor'] = {}
        self.all_dynamics_resid_dict['factor'] = {}

        for factorDynamicsType in FactorDynamicsType:
            self._fit_factor_dynamics_param(factorDynamicsType, scale_f, c)

    def _fit_risk_driver_dynamics_param(self, riskDriverDynamicsType: RiskDriverDynamicsType, scale: float, c: float):

        if riskDriverDynamicsType == RiskDriverDynamicsType.Linear:

            self._execute_general_linear_regression(tgt_key=riskDriverDynamicsType, var_type='risk-driver', scale=scale)

        elif riskDriverDynamicsType == RiskDriverDynamicsType.NonLinear:

            self._execute_general_threshold_regression(tgt_key=riskDriverDynamicsType, var_type='risk-driver',
                                                       scale=scale, c=c)

        else:
            raise NameError('Invalid riskDriverDynamicsType: ' + riskDriverDynamicsType.value)

    def _fit_factor_dynamics_param(self, factorDynamicsType: FactorDynamicsType, scale_f: float, c: float):

        if factorDynamicsType == FactorDynamicsType.AR:

            self._execute_general_linear_regression(tgt_key=factorDynamicsType, var_type='factor', scale=scale_f)

        elif factorDynamicsType == FactorDynamicsType.SETAR:

            self._execute_general_threshold_regression(tgt_key=factorDynamicsType, var_type='factor', scale=scale_f,
                                                       c=c)

        elif factorDynamicsType in (FactorDynamicsType.GARCH, FactorDynamicsType.TARCH, FactorDynamicsType.AR_TARCH):

            self._execute_garch_tarch_ar_tarch(factorDynamicsType, scale_f)

        else:

            raise NameError('Invalid factorDynamicsType: ' + factorDynamicsType.value)

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
            c = df_reg[var_type].mean()

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

        self.all_dynamics_param_dict[var_type][tgt_key]['abs_epsi_autocorr'] = \
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

            model_fit = model.fit(disp=0)
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

            model_fit = model.fit(disp=0)
            params = model_fit.params.copy()
            params.rename(index={'Const': 'mu'}, inplace=True)

            abs_epsi_autocorr, epsi = self._get_arch_abs_epsi_autocorr(model_fit, scale_f)

            B, alpha, beta, gamma, mu, omega = self._extract_ar_tarch_params_from_model_fit(params, scale_f)

            self._set_ar_tarch_params(B, alpha, beta, factorDynamicsType, gamma, mu, omega, abs_epsi_autocorr, epsi)

        else:
            raise NameError('Invalid factorDynamicsType: ' + factorDynamicsType.value)

        self.all_dynamics_model_dict['factor'][factorDynamicsType] = [model_fit]

    def _get_arch_abs_epsi_autocorr(self, model_fit, scale_f):
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

    def _extract_B_mu_sig2_from_reg(self, model_fit, scale: float):

        B = model_fit.params['factor']
        mu = model_fit.params['const'] / scale
        sig2 = model_fit.mse_resid / scale ** 2

        return B, mu, sig2

    def _extract_B_mu_sig2_from_auto_reg(self, auto_reg, scale_f: float):

        B = auto_reg.params.iloc[1]
        mu = auto_reg.params.iloc[0] / scale_f
        sig2 = auto_reg.sigma2 / scale_f ** 2

        return B, mu, sig2

    def _extract_tarch_params_from_model_fit(self, params: pd.Series, scale_f: float):

        alpha, beta, mu, omega = self._extract_garch_params_from_model_fit(params, scale_f)
        gamma = params['gamma[1]'] / scale_f ** 2

        return alpha, beta, gamma, mu, omega

    def _extract_garch_params_from_model_fit(self, params, scale_f):

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

        filename = os.path.dirname(os.path.dirname(__file__)) + \
            '/data/financial_time_series_data/financial_time_series_calibrations/' + \
            ticker + '-riskDriverType-' + riskDriverType.value + '-' + var_type + '-calibrations.xlsx'

        writer = pd.ExcelWriter(filename)
        workbook = writer.book

        df_riskDriverType = pd.DataFrame(data=[riskDriverType.value], columns=['riskDriverType'])
        df_riskDriverType.to_excel(writer, sheet_name='riskDriverType', index=False)

        if var_type == 'risk-driver':
            df_start_price = pd.DataFrame(data=[self.financialTimeSeries.info.loc['start_price'][0]],
                                          columns=['start_price'])
            df_start_price.to_excel(writer, sheet_name='start_price', index=False)

        for dynamicsType, param_dict in self.all_dynamics_param_dict[var_type].items():

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

        writer.close()

    def _set_report_filename(self, dynamicsType, i: int, var_type: str):

        riskDriverType = self.riskDriverType

        if dynamicsType in (RiskDriverDynamicsType.Linear, FactorDynamicsType.AR, FactorDynamicsType.GARCH,
                            FactorDynamicsType.TARCH, FactorDynamicsType.AR_TARCH):
            filename = os.path.dirname(
                os.path.dirname(__file__)) + '/reports/calibrations/' + self.financialTimeSeries.ticker + \
                       '-riskDriverType-' + riskDriverType.value + \
                       '-' + var_type + \
                       '-' + dynamicsType.value + '.txt'
        elif dynamicsType in (RiskDriverDynamicsType.NonLinear, FactorDynamicsType.SETAR):
            filename = os.path.dirname(
                os.path.dirname(__file__)) + '/reports/calibrations/' + self.financialTimeSeries.ticker + \
                       '-riskDriverType-' + riskDriverType.value + \
                       '-' + var_type + \
                       '-' + dynamicsType.value + str(i) + '.txt'
        else:
            raise NameError('Invalid dynamicsType: ' + dynamicsType.value)

        return filename

    def _check_var_type(self, var_type: str):

        if var_type not in ('risk-driver', 'factor'):
            raise NameError('var_type must be equal to risk-driver or factor')


class AllSeriesDynamicsCalibrator:

    def __init__(self):

        self.all_series_dynamics_calibrators = {}
        self.best_factorDynamicsType = {}
        self.best_factorDynamicsType_resid = {}
        self.non_best_factorDynamicsType_resid = {}
        self.average_price_per_contract = {}
        self.std_price_changes = {}

    def fit_all_series_dynamics(self):

        for ticker in tqdm(get_available_futures_tickers(), 'Fitting all time series'):
            self._set_dynamicsCalibrator(ticker)

            self._get_best_factorDynamicsType_and_resid(ticker)

    def print_all_series_dynamics_results(self):

        self._plot_financial_time_series()
        self._plot_residuals()
        self._print_report()
        self._print_statistics()

    def _plot_financial_time_series(self):

        for ticker, dynamicsCalibrator in self.all_series_dynamics_calibrators.items():
            financialTimeSeries = dynamicsCalibrator.financialTimeSeries
            time_series = financialTimeSeries.time_series[ticker]

            dpi = plt.rcParams['figure.dpi']
            fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
            plt.plot(time_series, label=ticker)
            plt.title(ticker + ' time series')
            plt.xlabel('Date')
            plt.ylabel('Value [$]')

            plt.savefig(os.path.dirname(os.path.dirname(__file__))
                        + '/figures/residuals/'
                        + ticker + '-time-series.png')

            plt.close(fig)

    def _print_report(self):

        ll = []

        for ticker, d1 in self.best_factorDynamicsType_resid.items():

            factorDynamicsType = d1['factorDynamicsType']
            abs_epsi_autocorr = d1['abs_epsi_autocorr']

            ll.append([ticker, factorDynamicsType.value] + [a for a in abs_epsi_autocorr])

            for factorDynamicsType, d2 in self.non_best_factorDynamicsType_resid[ticker].items():
                abs_epsi_autocorr = d2['abs_epsi_autocorr']

                ll.append([ticker, factorDynamicsType.value] + [a for a in abs_epsi_autocorr])

        df_report = pd.DataFrame(data=ll,
                                 columns=['ticker', 'factorDynamicsType']
                                         + ['autocorr_lag_%d' % a for a in range(len(abs_epsi_autocorr))])

        filename = os.path.dirname(os.path.dirname(__file__)) + '/reports/model_choice/residuals_analysis.csv'
        df_report.to_csv(filename, index=False)

    def _print_statistics(self):

        average_prices_per_contract_df = pd.DataFrame.from_dict(self.average_price_per_contract,
                                                                orient='index',
                                                                columns=['Average Price Per Contract'])
        std_price_changes_df = pd.DataFrame.from_dict(self.std_price_changes,
                                                      orient='index',
                                                      columns=['Standard Deviation of Price Changes'])
        out_dict = pd.concat([average_prices_per_contract_df, std_price_changes_df], axis=1)

        filename = os.path.dirname(os.path.dirname(__file__)) + '/reports/model_choice/prices_and_stds.csv'
        out_dict.to_csv(filename, index=True)

    def _plot_residuals(self):
        self._plot_best_residuals()
        self._plot_non_best_residuals()

    def _plot_best_residuals(self):

        for ticker, d in self.best_factorDynamicsType_resid.items():
            factorDynamicsType = d['factorDynamicsType']
            resid = d['resid']
            abs_epsi_autocorr = d['abs_epsi_autocorr']

            s = ticker + ', ' + factorDynamicsType.value

            dpi = plt.rcParams['figure.dpi']
            fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
            ax1 = plt.subplot2grid((2, 1), (0, 0))
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

            plt.savefig(os.path.dirname(os.path.dirname(__file__))
                        + '/figures/residuals/'
                        + ticker + '-residuals-best-' + factorDynamicsType.value + '.png')

            plt.close(fig)

    def _plot_non_best_residuals(self):

        for ticker, d1 in self.non_best_factorDynamicsType_resid.items():

            for factorDynamicsType, d in d1.items():
                resid = d['resid']
                abs_epsi_autocorr = d['abs_epsi_autocorr']

                s = ticker + ', ' + factorDynamicsType.value

                dpi = plt.rcParams['figure.dpi']
                fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
                ax1 = plt.subplot2grid((2, 1), (0, 0))
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

                plt.savefig(os.path.dirname(os.path.dirname(__file__))
                            + '/figures/residuals/'
                            + ticker + '-residuals-non-best-' + factorDynamicsType.value + '.png')

                plt.close(fig)

    def _get_best_factorDynamicsType_and_resid(self, ticker):

        self.best_factorDynamicsType_resid[ticker] = {}
        self.non_best_factorDynamicsType_resid[ticker] = {}

        all_factor_params = self.all_series_dynamics_calibrators[ticker].all_dynamics_param_dict['factor']
        all_factor_resids = self.all_series_dynamics_calibrators[ticker].all_dynamics_resid_dict['factor']

        factorDynamics_best = self._get_best_factorDynamicsType_and_resid_impl(all_factor_params, all_factor_resids,
                                                                               ticker)

        self._get_non_best_factorDynamicsType_and_resid(all_factor_params, all_factor_resids, factorDynamics_best,
                                                        ticker)

    def _get_best_factorDynamicsType_and_resid_impl(self, all_factor_params, all_factor_resids, ticker):

        abs_epsi_autocorr_best = None
        abs_epsi_autocorr_best_lag1 = 1.

        factorDynamics_best = None
        for factorDynamicsType in FactorDynamicsType:

            abs_epsi_autocorr = all_factor_params[factorDynamicsType]['abs_epsi_autocorr']

            if np.abs(abs_epsi_autocorr[1]) <= np.abs(abs_epsi_autocorr_best_lag1):
                abs_epsi_autocorr_best = abs_epsi_autocorr
                abs_epsi_autocorr_best_lag1 = abs_epsi_autocorr_best[1]

                factorDynamics_best = factorDynamicsType

        self.best_factorDynamicsType[ticker] = factorDynamics_best
        self.best_factorDynamicsType_resid[ticker]['factorDynamicsType'] = factorDynamics_best
        self.best_factorDynamicsType_resid[ticker]['resid'] = all_factor_resids[factorDynamics_best]
        self.best_factorDynamicsType_resid[ticker]['abs_epsi_autocorr'] = abs_epsi_autocorr_best
        return factorDynamics_best

    def _get_non_best_factorDynamicsType_and_resid(self, all_factor_params, all_factor_resids, factorDynamics_best,
                                                   ticker):

        non_best_factorDynamicsType = [factorDynamicsType for factorDynamicsType in FactorDynamicsType
                                       if factorDynamicsType != factorDynamics_best]

        for factorDynamicsType in non_best_factorDynamicsType:
            self.non_best_factorDynamicsType_resid[ticker][factorDynamicsType] = \
                {'resid': all_factor_resids[factorDynamicsType],
                 'abs_epsi_autocorr': all_factor_params[factorDynamicsType]['abs_epsi_autocorr']}

    def _set_dynamicsCalibrator(self, ticker):
        financialTimeSeries = FinancialTimeSeries(ticker=ticker)
        dynamicsCalibrator = DynamicsCalibrator()
        dynamicsCalibrator.fit_all_dynamics_param(financialTimeSeries)
        self.all_series_dynamics_calibrators[ticker] = dynamicsCalibrator
        self.riskDriverType = financialTimeSeries.riskDriverType
        self.factorComputationType = financialTimeSeries.factorComputationType

        self.average_price_per_contract[ticker] = financialTimeSeries.time_series[ticker].mean()
        self.std_price_changes[ticker] = financialTimeSeries.time_series[ticker].diff().std()


def build_filename_calibrations(riskDriverType, ticker, var_type):
    filename = os.path.dirname(os.path.dirname(__file__)) + \
               '/data/financial_time_series_data/financial_time_series_calibrations/' + \
               ticker + '-riskDriverType-' + riskDriverType.value + '-' + var_type + '-calibrations.xlsx'

    return filename


def get_available_futures_tickers():
    lst = ['cocoa', 'coffee', 'copper', 'WTI', 'gasoil', 'gold', 'lead', 'nat-gas-rngc1d', 'nat-gas-reuter', 'nickel',
           'silver', 'sugar', 'tin', 'unleaded', 'zinc']

    return lst


def get_futures_data_filename():
    filename = os.path.dirname(os.path.dirname(__file__)) + '/data/data_source/market_data/futures_data.xlsx'
    return filename


def read_futures_data_by_ticker(filename, ticker):
    time_series = pd.read_excel(filename, sheet_name=ticker, index_col=0).fillna(method='pad')
    return time_series


# ----------------------------------------- TESTS

if __name__ == '__main__':
    allSeriesDynamicsCalibrator = AllSeriesDynamicsCalibrator()
    allSeriesDynamicsCalibrator.fit_all_series_dynamics()
    allSeriesDynamicsCalibrator.print_all_series_dynamics_results()

    # financialTimeSeries = FinancialTimeSeries(ticker='WTI')
    # dynamicsCalibrator = DynamicsCalibrator()
    # dynamicsCalibrator.fit_all_dynamics_param(financialTimeSeries, scale=1, scale_f=1, c=0)
    # dynamicsCalibrator.print_results()
