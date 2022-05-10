import pandas as pd
from arch import arch_model
from arch.univariate import ARX, GARCH
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.ar_model import AutoReg

from enums import RiskDriverDynamicsType, FactorDynamicsType
from market_utils.financial_time_series import FinancialTimeSeries


class DynamicsCalibrator:

    def __init__(self):

        self._all_dynamics_param_dict = {}
        self._all_dynamics_model_dict = {}

    def fit_all_dynamics_param(self, financialTimeSeries: FinancialTimeSeries,
                               scale: float = 1,
                               scale_f: float = 1,
                               c : float = 0):

        self.financialTimeSeries = financialTimeSeries
        self._fit_all_risk_driver_dynamics_param(scale, c)
        self._fit_all_factor_dynamics_param(scale_f, c)
        self._set_riskDriverType()

    def _set_riskDriverType(self):

        self.riskDriverType = self.financialTimeSeries.riskDriverType

    def get_params_dict(self, var_type, dynamicsType):

        param_dict = self._all_dynamics_param_dict[var_type][dynamicsType]

        return param_dict

    def _fit_all_risk_driver_dynamics_param(self, scale: float, c: float):

        self._all_dynamics_param_dict['risk-driver'] = {}
        self._all_dynamics_model_dict['risk-driver'] = {}

        for riskDriverDynamicsType in RiskDriverDynamicsType:

            self._fit_risk_driver_dynamics_param(riskDriverDynamicsType, scale, c)

    def _fit_all_factor_dynamics_param(self, scale_f: float, c: float):

        self._all_dynamics_param_dict['factor'] = {}
        self._all_dynamics_model_dict['factor'] = {}

        for factorDynamicsType in FactorDynamicsType:

            self._fit_factor_dynamics_param(factorDynamicsType, scale_f, c)

    def _fit_risk_driver_dynamics_param(self, riskDriverDynamicsType: RiskDriverDynamicsType, scale: float, c: float):

        if riskDriverDynamicsType == RiskDriverDynamicsType.Linear:

            self._execute_general_linear_regression(tgt_key=riskDriverDynamicsType, var_type='risk-driver', scale=scale)

        elif riskDriverDynamicsType == RiskDriverDynamicsType.NonLinear:

            self._execute_general_threshold_regression(tgt_key=riskDriverDynamicsType, var_type='risk-driver', scale=scale,
                                                       c=c)

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

        self._all_dynamics_param_dict[var_type][tgt_key] = {}

        # regression data
        df_reg = self._prepare_df_reg(var_type)

        ind = df_reg.index

        # regression
        B, mu, model_fit, sig2 = self._execute_ols(df_reg, ind, scale, var_type)

        self._all_dynamics_param_dict[var_type][tgt_key]['mu'] = mu
        self._all_dynamics_param_dict[var_type][tgt_key]['B'] = B
        self._all_dynamics_param_dict[var_type][tgt_key]['sig2'] = sig2

        self._all_dynamics_model_dict[var_type][tgt_key] = [model_fit]

    def _execute_general_threshold_regression(self, tgt_key, var_type: str, scale: float, c: float):

        self._check_var_type(var_type)

        self._all_dynamics_param_dict[var_type][tgt_key] = {}

        # regression data
        df_reg = self._prepare_df_reg(var_type)

        ind_0 = df_reg['factor'] < c
        ind_1 = df_reg['factor'] >= c
        ind_lst = [ind_0, ind_1]
        p = ind_0.sum() / ind_1.sum()

        self._all_dynamics_param_dict[var_type][tgt_key]['c'] = c
        self._all_dynamics_param_dict[var_type][tgt_key]['p'] = p

        model_lst = []

        for i in range(len(ind_lst)):

            ind = ind_lst[i]

            # regression
            B, mu, model_fit, sig2 = self._execute_ols(df_reg, ind, scale, var_type)

            self._all_dynamics_param_dict[var_type][tgt_key]['mu_%d' % i] = mu
            self._all_dynamics_param_dict[var_type][tgt_key]['B_%d' % i] = B
            self._all_dynamics_param_dict[var_type][tgt_key]['sig2_%d' % i] = sig2

            model_lst.append(model_fit)

        self._all_dynamics_model_dict[var_type][tgt_key] = model_lst

    def _execute_ols(self, df_reg: pd.DataFrame, ind: pd.Index, scale: float, var_type: str):

        if var_type == 'risk-driver':
            model_fit = OLS(df_reg['risk-driver'].loc[ind], add_constant(df_reg['factor'].loc[ind])).fit()
            B, mu, sig2 = self._extract_B_mu_sig2_from_reg(model_fit, scale)

        else:
            model_fit = AutoReg(df_reg['factor'].loc[ind], lags=1, old_names=False).fit()
            B, mu, sig2 = self._extract_B_mu_sig2_from_auto_reg(model_fit, scale)

        return B, mu, model_fit, sig2

    def _execute_garch_tarch_ar_tarch(self, factorDynamicsType: FactorDynamicsType, scale_f: float):

        self._all_dynamics_param_dict['factor'][factorDynamicsType] = {}
        self._all_dynamics_model_dict['factor'][factorDynamicsType] = {}

        if factorDynamicsType in (FactorDynamicsType.GARCH, FactorDynamicsType.TARCH):

            df_model = self._prepare_df_model_factor_diff()

            if factorDynamicsType == FactorDynamicsType.GARCH:
                model = arch_model(df_model, p=1, q=1, rescale=False)
            else:
                model = arch_model(df_model, p=1, o=1, q=1, rescale=False)

            model_fit = model.fit()
            params = model_fit.params.copy()

            if factorDynamicsType == FactorDynamicsType.GARCH:
                alpha, beta, mu, omega = self._extract_garch_params_from_model_fit(params, scale_f)
                self._set_garch_params(alpha, beta, factorDynamicsType, mu, omega)
            else:
                alpha, beta, gamma, mu, omega = self._extract_tarch_params_from_model_fit(params, scale_f)
                self._set_tarch_params(alpha, beta, factorDynamicsType, gamma, mu, omega)

        elif factorDynamicsType == FactorDynamicsType.AR_TARCH:

            df_model = self._prepare_df_model_factor()

            model = ARX(df_model, lags=1, rescale=False)
            model.volatility = GARCH(p=1, o=1, q=1)

            model_fit = model.fit()
            params = model_fit.params.copy()
            params.rename(index={'Const': 'mu'}, inplace=True)

            B, alpha, beta, gamma, mu, omega = self._extract_ar_tarch_params_from_model_fit(params, scale_f)

            self._set_ar_tarch_params(B, alpha, beta, factorDynamicsType, gamma, mu, omega)

        else:
            raise NameError('Invalid factorDynamicsType: ' + factorDynamicsType.value)

        self._all_dynamics_model_dict['factor'][factorDynamicsType] = [model_fit]

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

    def _extract_B_mu_sig2_from_reg(self, model_fit, scale: str):

        B = model_fit.params['factor']
        mu = model_fit.params['const'] / scale
        sig2 = model_fit.mse_resid / scale ** 2

        return B, mu, sig2

    def _extract_B_mu_sig2_from_auto_reg(self, auto_reg, scale_f: str):

        B = auto_reg.params.iloc[1]
        mu = auto_reg.params.iloc[0] / scale_f
        sig2 = auto_reg.sigma2 / scale_f ** 2

        return B, mu, sig2

    def _extract_tarch_params_from_model_fit(self, params: dict, scale_f: str):

        alpha, beta, mu, omega = self._extract_garch_params_from_model_fit(params, scale_f)
        gamma = params['gamma[1]'] / scale_f ** 2

        return alpha, beta, gamma, mu, omega

    def _extract_garch_params_from_model_fit(self, params, scale_f):

        mu = params['mu'] / scale_f
        omega = params['omega'] / scale_f ** 2
        alpha = params['alpha[1]'] / scale_f ** 2
        beta = params['beta[1]'] / scale_f ** 2

        return alpha, beta, mu, omega

    def _extract_ar_tarch_params_from_model_fit(self, params: dict, scale_f: float):

        alpha, beta, gamma, mu, omega = self._extract_tarch_params_from_model_fit(params, scale_f)
        B = params['factor[1]']

        return B, alpha, beta, gamma, mu, omega

    def _set_garch_params(self, alpha: float, beta: float, factorDynamicsType: FactorDynamicsType, mu: float,
                          omega: float):

        self._all_dynamics_param_dict['factor'][factorDynamicsType]['mu'] = mu
        self._all_dynamics_param_dict['factor'][factorDynamicsType]['omega'] = omega
        self._all_dynamics_param_dict['factor'][factorDynamicsType]['alpha'] = alpha
        self._all_dynamics_param_dict['factor'][factorDynamicsType]['beta'] = beta

    def _set_tarch_params(self, alpha: float, beta: float, factorDynamicsType: FactorDynamicsType, gamma: float, mu: float,
                          omega: float):

        self._set_garch_params(alpha, beta, factorDynamicsType, mu, omega)
        self._all_dynamics_param_dict['factor'][factorDynamicsType]['gamma'] = gamma
        self._all_dynamics_param_dict['factor'][factorDynamicsType]['c'] = 0

    def _set_ar_tarch_params(self, B: float, alpha: float, beta: float, factorDynamicsType: FactorDynamicsType,
                             gamma: float, mu: float, omega: float):

        self._set_tarch_params(alpha, beta, factorDynamicsType, gamma, mu, omega)
        self._all_dynamics_param_dict['factor'][factorDynamicsType]['B'] = B

    def print_results(self):

        self._print_results_impl('risk-driver')
        self._print_results_impl('factor')

    def _print_results_impl(self, var_type: str):

        self._check_var_type(var_type)
        ticker = self.financialTimeSeries.ticker
        riskDriverType = self.riskDriverType

        filename = '../data/data_tmp/' + ticker + '-riskDriverType-' + riskDriverType.value + '-' + var_type + \
                   '-calibrations.xlsx'

        writer = pd.ExcelWriter(filename)
        workbook = writer.book

        df_riskDriverType = pd.DataFrame(data=[riskDriverType.value], columns=['riskDriverType'])
        df_riskDriverType.to_excel(writer, sheet_name='riskDriverType', index=False)

        if var_type == 'risk-driver':
            df_start_price = pd.DataFrame(data=[self.financialTimeSeries.info.loc['start_price'][0]],
                                          columns=['start_price'])
            df_start_price.to_excel(writer, sheet_name='start_price', index=False)

        for dynamicsType, param_dict in self._all_dynamics_param_dict[var_type].items():

            # parameters
            worksheet = workbook.add_worksheet(dynamicsType.value)
            writer.sheets[dynamicsType.value] = worksheet
            df_params_out = pd.DataFrame.from_dict(data=param_dict,
                                                   orient='index',
                                                   columns=['param'])
            df_params_out.to_excel(writer, sheet_name=dynamicsType.value)

            # reports
            for i in range(len(self._all_dynamics_model_dict[var_type][dynamicsType])):

                model = self._all_dynamics_model_dict[var_type][dynamicsType][i]
                filename = self._set_report_filename(dynamicsType, i, var_type)

                with open(filename, 'w+') as fh:
                    fh.write(model.summary().as_text())

        writer.close()

    def _set_report_filename(self, dynamicsType, i: int, var_type: str):

        riskDriverType = self.riskDriverType

        if dynamicsType in (RiskDriverDynamicsType.Linear, FactorDynamicsType.AR, FactorDynamicsType.GARCH,
                            FactorDynamicsType.TARCH, FactorDynamicsType.AR_TARCH):
            filename = '../reports/' + self.financialTimeSeries.ticker +\
                       '-riskDriverType-' + riskDriverType.value +\
                       '-' + var_type +\
                       '-' + dynamicsType.value + '.txt'
        elif dynamicsType in (RiskDriverDynamicsType.NonLinear, FactorDynamicsType.SETAR):
            filename = '../reports/' + self.financialTimeSeries.ticker +\
                       '-riskDriverType-' + riskDriverType.value +\
                       '-' + var_type +\
                       '-' + dynamicsType.value + str(i) +'.txt'

        return filename

    def _check_var_type(self, var_type: str):

        if var_type not in ('risk-driver', 'factor'):
            raise NameError('var_type must be equal to risk-driver or factor')


def build_filename_calibrations(riskDriverType, ticker, var_type):

    filename = '../data/data_tmp/' + ticker + '-riskDriverType-' + riskDriverType.value + '-' + var_type + \
               '-calibrations.xlsx'

    return filename


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':

    financialTimeSeries = FinancialTimeSeries('WTI')
    financialTimeSeries.set_time_series()

    dynamicsCalibrator = DynamicsCalibrator()
    dynamicsCalibrator.fit_all_dynamics_param(financialTimeSeries, scale=1, scale_f=1, c=0)
    dynamicsCalibrator.print_results()
