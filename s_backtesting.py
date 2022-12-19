import numpy as np

from gen_utils.utils import read_ticker
from testing_utils.testers import BackTester
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    np.random.seed(789)

    # If true, then the traders split their strategy into sub-periods of the same length used to calibrate the RL agent
    split_strategy = True

    # -------------------- Input parameters
    ticker = read_ticker()

    # -------------------- Execution
    backtester = BackTester(ticker=ticker, split_strategy=split_strategy)
    backtester.execute_backtesting()
    backtester.make_plots()

    print('--- End s_backtesting.py')
