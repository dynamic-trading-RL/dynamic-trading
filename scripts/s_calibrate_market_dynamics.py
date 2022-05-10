from market_utils.calibrator import DynamicsCalibrator
from enums import RiskDriverType, FactorDefinitionType
from market_utils.financial_time_series import FinancialTimeSeries


# -------------------- Input parameters
ticker = 'WTI'
t_past = 8000
riskDriverType = RiskDriverType.PnL
factorDefinitionType = FactorDefinitionType.MovingAverage
window = 5
scale = 1
scale_f = 1
c = 0


# -------------------- Execution
financialTimeSeries = FinancialTimeSeries(ticker=ticker)
financialTimeSeries.set_time_series(t_past=t_past, riskDriverType=riskDriverType,
                                    factorDefinitionType=factorDefinitionType, window=window)

dynamicsCalibrator = DynamicsCalibrator()
dynamicsCalibrator.fit_all_dynamics_param(financialTimeSeries, scale=scale, scale_f=scale_f, c=c)
dynamicsCalibrator.print_results()
