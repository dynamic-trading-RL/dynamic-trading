
from backtesting_utils.backtesting import Backtester

if __name__ == '__main__':

    # -------------------- Input parameters
    ticker = 'WTI'

    # -------------------- Execution
    backtester = Backtester(ticker=ticker)
    backtester.execute_backtesting()
    backtester.make_plots()

    print('--- End s_in_sample_testing.py')
