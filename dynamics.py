import pandas as pd
from enums import AssetDynamicsType, FactorDynamicsType
from financial_time_series import FinancialTimeSeries
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model
from arch.univariate import ARX, GARCH


class DynamicsParametersCalibrator:

    def __init__(self):

        self.all_dynamics_param_dict = {}
        self._all_dynamics_model_dict = {}

    def fit_all_dynamics_param(self, financialTimeSeries: FinancialTimeSeries,
                               scale: float = 1,
                               scale_f: float = 1,
                               c : float = 0):

        self.financialTimeSeries = financialTimeSeries
        self._fit_all_asset_dynamics_param(scale, c)
        self._fit_all_factor_dynamics_param(scale_f, c)

    def use_pnl(self):

        return self.financialTimeSeries.use_pnl

    def _fit_all_asset_dynamics_param(self, scale: float, c: float):

        self.all_dynamics_param_dict['asset'] = {}
        self._all_dynamics_model_dict['asset'] = {}

        if self.financialTimeSeries.use_pnl:
            self._tgt = 'pnl'
        else:
            self._tgt = 'return'

        for assetDynamicsType in AssetDynamicsType:

            self._fit_asset_dynamics_param(assetDynamicsType, scale, c)

    def _fit_all_factor_dynamics_param(self, scale_f: float, c: float):

        self.all_dynamics_param_dict['factor'] = {}
        self._all_dynamics_model_dict['factor'] = {}

        for factorDynamicsType in FactorDynamicsType:

            self._fit_factor_dynamics_param(factorDynamicsType, scale_f, c)

    def _fit_asset_dynamics_param(self, assetDynamicsType: AssetDynamicsType, scale: float, c: float):

        tgt = self._tgt

        if assetDynamicsType == AssetDynamicsType.Linear:

            self._execute_general_linear_regression(tgt_key=assetDynamicsType, var_type='asset', scale=scale, tgt=tgt)

        elif assetDynamicsType == AssetDynamicsType.NonLinear:

            self._execute_general_threshold_regression(tgt_key=assetDynamicsType, var_type='asset', scale=scale,
                                                       tgt=tgt, c=c)

        else:
            raise NameError('Invalid assetDynamicsType: ' + assetDynamicsType.value)

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

    def _execute_general_linear_regression(self, tgt_key, var_type: str, scale: float, tgt: str = None):

        self._check_var_type(var_type)

        self.all_dynamics_param_dict[var_type][tgt_key] = {}

        # regression data
        df_reg = self._prepare_df_reg(var_type, tgt)

        ind = df_reg.index

        # regression
        B, mu, model_fit, sig2 = self._execute_ols(df_reg, ind, scale, tgt, var_type)

        self.all_dynamics_param_dict[var_type][tgt_key]['mu'] = mu
        self.all_dynamics_param_dict[var_type][tgt_key]['B'] = B
        self.all_dynamics_param_dict[var_type][tgt_key]['sig2'] = sig2

        self._all_dynamics_model_dict[var_type][tgt_key] = [model_fit]

    def _execute_general_threshold_regression(self, tgt_key, var_type: str, scale: float, c: float, tgt: str = None):

        self._check_var_type(var_type)

        self.all_dynamics_param_dict[var_type][tgt_key] = {}

        # regression data
        df_reg = self._prepare_df_reg(var_type, tgt)

        ind_0 = df_reg['f'] < c
        ind_1 = df_reg['f'] >= c
        ind_lst = [ind_0, ind_1]

        model_lst = []

        for i in range(len(ind_lst)):

            ind = ind_lst[i]

            # regression
            B, mu, model_fit, sig2 = self._execute_ols(df_reg, ind, scale, tgt, var_type)

            self.all_dynamics_param_dict[var_type][tgt_key]['mu_%d' % i] = mu
            self.all_dynamics_param_dict[var_type][tgt_key]['B_%d' % i] = B
            self.all_dynamics_param_dict[var_type][tgt_key]['sig2_%d' % i] = sig2
            self.all_dynamics_param_dict[var_type][tgt_key]['c'] = c

            model_lst.append(model_fit)

        self._all_dynamics_model_dict[var_type][tgt_key] = model_lst

    def _execute_ols(self, df_reg: pd.DataFrame, ind: pd.Index, scale: float, tgt: str, var_type: str):

        if var_type == 'asset':
            model_fit = OLS(df_reg[tgt].loc[ind], add_constant(df_reg['f'].loc[ind])).fit()
            B, mu, sig2 = self._extract_B_mu_sig2_from_reg(model_fit, scale)

        else:
            model_fit = AutoReg(df_reg['f'].loc[ind], lags=1, old_names=False).fit()
            B, mu, sig2 = self._extract_B_mu_sig2_from_auto_reg(model_fit, scale)

        return B, mu, model_fit, sig2

    def _execute_garch_tarch_ar_tarch(self, factorDynamicsType: FactorDynamicsType, scale_f: float):

        self.all_dynamics_param_dict['factor'][factorDynamicsType] = {}
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

    def _prepare_df_reg(self, var_type: str, tgt: str):

        if var_type == 'asset':
            df_reg = self._prepare_df_model_asset(tgt)
        else:
            df_reg = self._prepare_df_model_factor()
        return df_reg

    def _prepare_df_model_asset(self, tgt: str):

        df_model = self.financialTimeSeries.time_series[['f', tgt]].copy()
        df_model[tgt] = df_model[tgt].shift(-1)
        df_model.dropna(inplace=True)

        return df_model

    def _prepare_df_model_factor(self):

        df_model = self.financialTimeSeries.time_series['f'].copy()
        df_model = df_model.to_frame()
        df_model.dropna(inplace=True)

        return df_model

    def _prepare_df_model_factor_diff(self):

        df_model = self.financialTimeSeries.time_series['f'].diff().dropna().copy()

        return df_model

    def _extract_B_mu_sig2_from_reg(self, model_fit, scale: str):

        B = model_fit.params['f']
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
        B = params['f[1]']

        return B, alpha, beta, gamma, mu, omega

    def _set_garch_params(self, alpha: float, beta: float, factorDynamicsType: FactorDynamicsType, mu: float,
                          omega: float):

        self.all_dynamics_param_dict['factor'][factorDynamicsType]['mu'] = mu
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['omega'] = omega
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['alpha'] = alpha
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['beta'] = beta

    def _set_tarch_params(self, alpha: float, beta: float, factorDynamicsType: FactorDynamicsType, gamma: float, mu: float,
                          omega: float):

        self._set_garch_params(alpha, beta, factorDynamicsType, mu, omega)
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['gamma'] = gamma
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['c'] = 0

    def _set_ar_tarch_params(self, B: float, alpha: float, beta: float, factorDynamicsType: FactorDynamicsType,
                             gamma: float, mu: float, omega: float):

        self._set_tarch_params(alpha, beta, factorDynamicsType, gamma, mu, omega)
        self.all_dynamics_param_dict['factor'][factorDynamicsType]['B'] = B

    def print_results(self):

        self._print_results_impl('asset')
        self._print_results_impl('factor')

    def _print_results_impl(self, var_type: str):

        self._check_var_type(var_type)
        ticker = self.financialTimeSeries.ticker

        writer = pd.ExcelWriter('data_tmp/' + ticker + '_' + var_type + '_calibrations.xlsx')
        workbook = writer.book

        df_use_pnl = pd.DataFrame(data=[str(self.use_pnl())], columns=['use_pnl'])
        df_use_pnl.to_excel(writer, sheet_name='use_pnl', index=False)

        if var_type == 'asset':
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
            for i in range(len(self._all_dynamics_model_dict[var_type][dynamicsType])):

                model = self._all_dynamics_model_dict[var_type][dynamicsType][i]
                filename = self._set_report_filename(dynamicsType, i, var_type)

                with open(filename, 'w+') as fh:
                    fh.write(model.summary().as_text())

        writer.close()

    def _set_report_filename(self, dynamicsType, i: int, var_type: str):

        if len(self._all_dynamics_model_dict[var_type][dynamicsType]) > 0:
            filename = 'reports/' + self.financialTimeSeries.ticker +\
                       '-use_pnl-' + str(self.use_pnl()) +\
                       '-' + var_type +\
                       '-' + dynamicsType.value + str(i) +'.txt'
        else:
            filename = 'reports/' + self.financialTimeSeries.ticker +\
                       '-use_pnl-' + str(self.use_pnl()) +\
                       '-' + var_type +\
                       '-' + dynamicsType.value + '.txt'

        return filename

    def _check_var_type(self, var_type: str):

        if var_type not in ('asset', 'factor'):
            raise NameError('var_type must be equal to asset or factor')


class Dynamics:

    def __init__(self):

        self.parameters = {}

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

    def _read_parameters(self, ticker):

        if type(self) == AssetDynamics:
            filename = 'data_tmp/' + ticker + '_asset_calibrations.xlsx'
            sheet_name = self.assetDynamicsType.value

        elif type(self) == FactorDynamics:
            filename = 'data_tmp/' + ticker + '_factor_calibrations.xlsx'
            sheet_name = self.factorDynamicsType.value

        else:
            raise NameError('dynamicsType not properly set')

        params = pd.read_excel(filename,
                               sheet_name=sheet_name,
                               index_col=0)

        return params['param'].to_dict()

    def _read_use_pnl(self, ticker):

        sheet_name = 'use_pnl'

        if type(self) == AssetDynamics:
            filename = 'data_tmp/' + ticker + '_asset_calibrations.xlsx'
        elif type(self) == FactorDynamics:
            filename = 'data_tmp/' + ticker + '_factor_calibrations.xlsx'
        else:
            raise NameError('dynamicsType not properly set')

        use_pnl_df = pd.read_excel(filename, sheet_name=sheet_name)
        use_pnl_str = str(use_pnl_df['use_pnl'].iloc[0])

        return use_pnl_str == 'True'


class AssetDynamics(Dynamics):

    def __init__(self, assetDynamicsType: AssetDynamicsType):

        super().__init__()
        self.assetDynamicsType = assetDynamicsType

    def set_parameters_from_calibrator(self, dynamicsParametersCalibrator: DynamicsParametersCalibrator):

        self.use_pnl = dynamicsParametersCalibrator.use_pnl()
        param_dict = dynamicsParametersCalibrator.all_dynamics_param_dict['asset'][self.assetDynamicsType]
        self._set_parameters_from_dict(param_dict)

    def read_asset_parameters(self, ticker):

        self.use_pnl = super()._read_use_pnl(ticker)
        param_dict = super()._read_parameters(ticker)
        self._set_parameters_from_dict(param_dict)

    def _set_parameters_from_dict(self, param_dict):
        if self.assetDynamicsType == AssetDynamicsType.Linear:

            self._set_linear_parameters(param_dict)

        elif self.assetDynamicsType == AssetDynamicsType.NonLinear:

            self._set_threshold_parameters(param_dict)

        else:
            raise NameError('Invalid asset dynamics')

    def read_asset_start_price(self, ticker):

        filename = 'data_tmp/' + ticker + '_asset_calibrations.xlsx'
        sheet_name = 'start_price'

        start_price_df = pd.read_excel(filename, sheet_name=sheet_name)

        return start_price_df['start_price'].iloc[0]

class FactorDynamics(Dynamics):

    def __init__(self, factorDynamicsType: FactorDynamicsType):

        super().__init__()
        self.factorDynamicsType = factorDynamicsType

    def set_parameters_from_calibrator(self, dynamicsParametersCalibrator: DynamicsParametersCalibrator):

        self.use_pnl = dynamicsParametersCalibrator.use_pnl()
        param_dict = dynamicsParametersCalibrator.all_dynamics_param_dict['factor'][self.factorDynamicsType]
        self._set_parameters_from_dict(param_dict)

    def read_factor_parameters(self, ticker):

        self.use_pnl = super()._read_use_pnl(ticker)
        param_dict = super()._read_parameters(ticker)
        self._set_parameters_from_dict(param_dict)

    def _set_parameters_from_dict(self, param_dict):
        if self.factorDynamicsType == FactorDynamicsType.AR:

            self._set_linear_parameters(param_dict)

        elif self.factorDynamicsType == FactorDynamicsType.SETAR:

            self._set_threshold_parameters(param_dict)

        elif self.factorDynamicsType == FactorDynamicsType.GARCH:

            self._set_garch_parameters(param_dict)

        elif self.factorDynamicsType == FactorDynamicsType.TARCH:

            self._set_tarch_parameters(param_dict)

        elif self.factorDynamicsType == FactorDynamicsType.AR_TARCH:

            self._set_artarch_parameters(param_dict)

        else:
            raise NameError('Invalid factor dynamics')

    def _set_artarch_parameters(self, param_dict: dict):

        self._set_tarch_parameters(param_dict)
        self.parameters['B'] = param_dict['B']

    def _set_tarch_parameters(self, param_dict: dict):

        self._set_garch_parameters(param_dict)
        self.parameters['gamma'] = param_dict['gamma']
        self.parameters['c'] = param_dict['c']

    def _set_garch_parameters(self, param_dict: dict):

        self.parameters['mu'] = param_dict['mu']
        self.parameters['omega'] = param_dict['omega']
        self.parameters['alpha'] = param_dict['alpha']
        self.parameters['beta'] = param_dict['beta']


class MarketDynamics:

    def __init__(self, assetDynamics: AssetDynamics, factorDynamics: FactorDynamics):

        self.assetDynamics = assetDynamics
        self.factorDynamics = factorDynamics
        self._set_use_pnl()

    def get_assetDynamicsType_and_parameters(self):

        return self.assetDynamics.assetDynamicsType, self.assetDynamics.parameters

    def _set_use_pnl(self):

        if self.assetDynamics.use_pnl == self.factorDynamics.use_pnl:

            self.use_pnl = self.assetDynamics.use_pnl

        else:
            raise NameError('use_pnl different for asset and factor dynamics')


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':

    financialTimeSeries = FinancialTimeSeries('WTI')
    financialTimeSeries.set_time_series()

    dynamicsParametersCalibrator = DynamicsParametersCalibrator()
    dynamicsParametersCalibrator.fit_all_dynamics_param(financialTimeSeries, scale=1, scale_f=1, c=0)
    dynamicsParametersCalibrator.print_results()

    assetDynamics = AssetDynamics(AssetDynamicsType.NonLinear)
    factorDynamics = FactorDynamics(FactorDynamicsType.AR_TARCH)

    assetDynamics.set_parameters_from_calibrator(dynamicsParametersCalibrator)
    factorDynamics.set_parameters_from_calibrator(dynamicsParametersCalibrator)

    marketDynamics = MarketDynamics(assetDynamics=assetDynamics,
                                    factorDynamics=factorDynamics)
