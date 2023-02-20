from src.enums import ModeType
from src.gen_utils.utils import read_ticker
from src.market_utils.calibrator import DynamicsCalibrator
from src.market_utils.financial_time_series import FinancialTimeSeries


if __name__ == '__main__':

    # -------------------- Input parameters
    ticker = read_ticker()
    scale = 1
    scale_f = 1
    c = None  # can be None, in this case, uses mean of the process

    # -------------------- Execution
    financialTimeSeries = FinancialTimeSeries(ticker=ticker, modeType=ModeType.InSample)
    dynamicsCalibrator = DynamicsCalibrator()
    dynamicsCalibrator.fit_all_dynamics_param(financialTimeSeries, scale=scale, scale_f=scale_f, c=c)

    # PnL must be done for benchmark agents
    financialTimeSeries = FinancialTimeSeries(ticker=ticker, modeType=ModeType.InSample, forcePnL=True)
    dynamicsCalibrator = DynamicsCalibrator()
    dynamicsCalibrator.fit_all_dynamics_param(financialTimeSeries, scale=scale, scale_f=scale_f, c=c)

    print('--- End s_calibrate_specific_market_dynamics.py')
