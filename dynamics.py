import pandas as pd
from enums import AssetDynamicsType, FactorDynamicsType
from financial_time_series import FinancialTimeSeries
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.ar_model import AutoReg



class DynamicsParametersCalibrator:

    def __init__(self):

        self.all_dynamics_param_dict = {}
        self._all_dynamics_model_dict = {}

    def fit_all_dynamics_param(self, financialTimeSeries: FinancialTimeSeries, scale=1, scale_f=1, c=0):

        self._financialTimeSeries = financialTimeSeries
        self._fit_all_asset_dynamics_param(scale, c)
        self._fit_all_factor_dynamics_param(scale_f, c)

    def _fit_all_asset_dynamics_param(self, scale, c):

        self.all_dynamics_param_dict['asset'] = {}
        self._all_dynamics_model_dict['asset'] = {}

        if self._financialTimeSeries.use_pnl:
            self._tgt = 'pnl'
        else:
            self._tgt = 'return'

        for assetDynamicsType in AssetDynamicsType:

            self._fit_asset_dynamics_param(assetDynamicsType, scale, c)

    def _fit_asset_dynamics_param(self, assetDynamicsType, scale, c):

        tgt = self._tgt

        if assetDynamicsType == AssetDynamicsType.Linear:

            self._execute_general_linear_regression(tgt_key=assetDynamicsType, var_type='asset', scale=scale, tgt=tgt)

        elif assetDynamicsType == AssetDynamicsType.NonLinear:

            self._execute_general_threshold_regression(tgt_key=assetDynamicsType, var_type='asset', scale=scale,
                                                       tgt=tgt, c=c)

        else:
            raise NameError('Invalid assetDynamicsType: ' + assetDynamicsType.value)

    def _extract_B_mu_sig2_from_reg(self, reg, scale):

        B = reg.params['f']
        mu = reg.params['const'] / scale
        sig2 = reg.mse_resid / scale ** 2

        return B, mu, sig2

    def _fit_all_factor_dynamics_param(self, scale_f, c):

        self.all_dynamics_param_dict['factor'] = {}
        self._all_dynamics_model_dict['factor'] = {}

        for factorDynamicsType in FactorDynamicsType:

            self._fit_factor_dynamics_param(factorDynamicsType, scale_f, c)

    def _fit_factor_dynamics_param(self, factorDynamicsType, scale_f, c):

        if factorDynamicsType == FactorDynamicsType.AR:

            self._execute_general_linear_regression(tgt_key=factorDynamicsType, var_type='factor', scale=scale_f)

        elif factorDynamicsType == FactorDynamicsType.SETAR:

            self._execute_general_threshold_regression(tgt_key=factorDynamicsType, var_type='factor', scale=scale_f,
                                                       c=c)

        elif factorDynamicsType == FactorDynamicsType.GARCH:
            pass
        elif factorDynamicsType == FactorDynamicsType.TARCH:
            pass
        elif factorDynamicsType == FactorDynamicsType.AR_TARCH:
            pass
        else:
            raise NameError('Invalid factorDynamicsType: ' + factorDynamicsType.value)

    def _execute_general_linear_regression(self, tgt_key, var_type, scale, tgt=None):

        self._check_var_type(var_type)

        self.all_dynamics_param_dict[var_type][tgt_key] = {}

        # regression data
        df_reg = self._prepare_df_reg(var_type, tgt)

        ind = df_reg.index

        # regression
        B, mu, reg, sig2 = self._execute_general_linear_regression_impl(df_reg, ind, scale, tgt, var_type)

        self.all_dynamics_param_dict[var_type][tgt_key]['mu'] = mu
        self.all_dynamics_param_dict[var_type][tgt_key]['B'] = B
        self.all_dynamics_param_dict[var_type][tgt_key]['sig2'] = sig2

        self._all_dynamics_model_dict[var_type][tgt_key] = [reg]

    def _execute_general_threshold_regression(self, tgt_key, var_type, scale, c, tgt=None):

        self._check_var_type(var_type)

        self.all_dynamics_param_dict[var_type][tgt_key] = {}

        # regression data
        df_reg = self._prepare_df_reg(var_type, tgt)

        ind_0 = df_reg['f'] < c
        ind_1 = df_reg['f'] >= c
        ind_lst = [ind_0, ind_1]

        reg_lst = []

        for i in range(len(ind_lst)):

            ind = ind_lst[i]

            # regression
            B, mu, reg, sig2 = self._execute_general_linear_regression_impl(df_reg, ind, scale, tgt, var_type)

            self.all_dynamics_param_dict[var_type][tgt_key]['mu_%d' % i] = mu
            self.all_dynamics_param_dict[var_type][tgt_key]['B_%d' % i] = B
            self.all_dynamics_param_dict[var_type][tgt_key]['sig2_%d' % i] = sig2
            self.all_dynamics_param_dict[var_type][tgt_key]['c'] = c

            reg_lst.append(reg)

        self._all_dynamics_model_dict[var_type][tgt_key] = reg_lst

    def _execute_general_linear_regression_impl(self, df_reg, ind, scale, tgt, var_type):
        if var_type == 'asset':
            reg = OLS(df_reg[tgt].loc[ind], add_constant(df_reg['f'].loc[ind])).fit()
            B, mu, sig2 = self._extract_B_mu_sig2_from_reg(reg, scale)
        else:
            reg = AutoReg(df_reg['f'].loc[ind], lags=1, old_names=False).fit()
            B, mu, sig2 = self._extract_B_mu_sig2_from_auto_reg(reg, scale)
        return B, mu, reg, sig2

    def _prepare_df_reg(self, var_type, tgt):
        if var_type == 'asset':
            df_reg = self._prepare_df_reg_asset(tgt)
        else:
            df_reg = self._prepare_df_reg_factor()
        return df_reg

    def _check_var_type(self, var_type):
        if var_type not in ('asset', 'factor'):
            raise NameError('var_type must be equal to asset or factor')

    def _prepare_df_reg_asset(self, tgt):

        df_reg = self._financialTimeSeries.time_series[['f', tgt]].copy()
        df_reg[tgt] = df_reg[tgt].shift(-1)
        df_reg.dropna(inplace=True)

        return df_reg

    def _prepare_df_reg_factor(self):

        df_reg = self._financialTimeSeries.time_series['f'].copy()
        df_reg = df_reg.to_frame()
        df_reg.dropna(inplace=True)

        return df_reg

    def _extract_B_mu_sig2_from_auto_reg(self, auto_reg, scale_f):

        B = auto_reg.params.iloc[1]
        mu = auto_reg.params.iloc[0] / scale_f
        sig2 = auto_reg.sigma2 / scale_f ** 2

        return B, mu, sig2

    def print_results(self):

        self._print_results_asset()
        self._print_results_factor()

    def _print_results_asset(self):

        writer = pd.ExcelWriter('data_tmp/asset_calibrations.xlsx')
        workbook = writer.book

        for assetDynamicsType, param_dict in self.all_dynamics_param_dict['asset'].items():

            # parameters
            worksheet = workbook.add_worksheet(assetDynamicsType.value)
            writer.sheets[assetDynamicsType.value] = worksheet
            df_params_out = pd.DataFrame.from_dict(data=param_dict,
                                                   orient='index',
                                                   columns=['param'])
            df_params_out.to_excel(writer, sheet_name=assetDynamicsType.value)

            # reports
            for i in range(len(self._all_dynamics_model_dict['asset'][assetDynamicsType])):
                model = self._all_dynamics_model_dict['asset'][assetDynamicsType][i]

                with open('reports/'
                          + self._financialTimeSeries.ticker
                          + '-asset_'
                          + assetDynamicsType.value
                          + str(i)
                          + '.txt', 'w+') as fh:
                    fh.write(model.summary().as_text())

        writer.close()

    def _print_results_factor(self):

        pass


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

    def set_parameters(self, dynamicsParametersCalibrator: DynamicsParametersCalibrator):

        param_dict = dynamicsParametersCalibrator.all_dynamics_param_dict['asset'][self._assetDynamicsType]

        if self._assetDynamicsType == AssetDynamicsType.Linear:

            self._set_linear_parameters(param_dict)

        elif self._assetDynamicsType == AssetDynamicsType.NonLinear:

            self._set_threshold_parameters(param_dict)

        else:
            raise NameError('Invalid return dynamics')

    def read_asset_parameters(self):

        if self._assetDynamicsType == AssetDynamicsType.Linear:

            params = pd.read_excel('data_tmp/asset_calibrations.xlsx',
                                   sheet_name='linear',
                                   index_col=0)

        elif self._assetDynamicsType == AssetDynamicsType.NonLinear:

            params = pd.read_excel('data_tmp/asset_calibrations.xlsx',
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


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':

    financialTimeSeries = FinancialTimeSeries('WTI')
    financialTimeSeries.set_time_series()

    dynamicsParametersCalibrator = DynamicsParametersCalibrator()
    dynamicsParametersCalibrator.fit_all_dynamics_param(financialTimeSeries, scale=1, scale_f=1, c=0)
    dynamicsParametersCalibrator.print_results()

    assetDynamics_linear = AssetDynamics(AssetDynamicsType.Linear)
    assetDynamics_linear.set_parameters(dynamicsParametersCalibrator)

    assetDynamics_nonlinear = AssetDynamics(AssetDynamicsType.NonLinear)
    assetDynamics_nonlinear.set_parameters(dynamicsParametersCalibrator)

    factorDynamics_ar = FactorDynamics(FactorDynamicsType.AR)
