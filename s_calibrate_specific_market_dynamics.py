import os
from joblib import dump

from gen_utils.utils import read_ticker
from market_utils.calibrator import DynamicsCalibrator
from market_utils.financial_time_series import FinancialTimeSeries


if __name__ == '__main__':

    # -------------------- Input parameters
    ticker = read_ticker()
    scale = 1
    scale_f = 1
    c = 0

    # -------------------- Execution
    financialTimeSeries = FinancialTimeSeries(ticker=ticker)

    dynamicsCalibrator = DynamicsCalibrator()
    dynamicsCalibrator.fit_all_dynamics_param(financialTimeSeries, scale=scale, scale_f=scale_f, c=c)

    print('--- End s_calibrate_specific_market_dynamics.py')
