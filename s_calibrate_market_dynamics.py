from market_utils.calibrator import DynamicsCalibrator
from market_utils.financial_time_series import FinancialTimeSeries
from enums import RiskDriverType, FactorDefinitionType


if __name__ == '__main__':

    # -------------------- Input parameters
    ticker = 'WTI'
    scale = 1
    scale_f = 1
    c = 0

    # -------------------- Execution
    financialTimeSeries = FinancialTimeSeries(ticker=ticker)
    financialTimeSeries.set_time_series()

    dynamicsCalibrator = DynamicsCalibrator()
    dynamicsCalibrator.fit_all_dynamics_param(financialTimeSeries, scale=scale, scale_f=scale_f, c=c)
    dynamicsCalibrator.print_results()
