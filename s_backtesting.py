
from backtesting_utils.backtesting import Backtester

if __name__ == '__main__':

    # -------------------- Input parameters
    ticker = 'WTI'
    t_past = 50

    # -------------------- Execution
    backtester = Backtester(ticker=ticker, t_past=t_past)
    backtester.execute_backtesting()
    backtester.make_plots()

    print('--- End s_in_sample_testing.py')
