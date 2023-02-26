
import numpy as np

from dynamic_trading.testing_utils.testers import SimulationTester, read_out_of_sample_parameters
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    np.random.seed(789)

    # -------------------- Input parameters
    j_oos, t_, t_test_mode = read_out_of_sample_parameters()

    # -------------------- Execution
    simulationTester = SimulationTester()
    simulationTester.execute_simulation_testing(j_=j_oos, t_=t_, t_test_mode=t_test_mode)
    simulationTester.make_plots(j_trajectories_plot=5)

    print('--- End s_simulationtesting.py')
