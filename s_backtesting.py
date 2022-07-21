from testing_utils.testing import BackTester
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # -------------------- Input parameters
    ticker = 'WTI'

    # -------------------- Execution
    backtester = BackTester(ticker=ticker)
    backtester.execute_backtesting()
    backtester.make_plots()

    print('--- End s_in_sample_testing.py')
