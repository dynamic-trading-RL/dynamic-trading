import numpy as np

from dynamic_trading.testing_utils.testers import BackTester
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    np.random.seed(789)

    # -------------------- Input parameters
    # If true, then the traders split their strategy into sub-periods of the same length used to calibrate the RL agent
    split_strategy = True

    # -------------------- Execution
    backtester = BackTester(split_strategy=split_strategy)
    backtester.execute_backtesting()
    backtester.make_plots()

    print('--- End s_backtesting.py')
