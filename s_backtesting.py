import numpy as np
from testing_utils.testers import BackTester
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    np.random.seed(789)

    # -------------------- Input parameters
    ticker = 'WTI'

    # -------------------- Execution
    backtester = BackTester(ticker=ticker)
    backtester.execute_backtesting()
    backtester.make_plots()

    print('--- End s_backtesting.py')
