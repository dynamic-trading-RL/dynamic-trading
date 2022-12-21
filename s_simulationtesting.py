
import numpy as np

from gen_utils.utils import read_ticker
from testing_utils.testers import SimulationTester, read_out_of_sample_parameters
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    np.random.seed(789)

    # -------------------- Input parameters
    j_oos, t_ = read_out_of_sample_parameters()

    # -------------------- Execution
    simulationTester = SimulationTester()
    simulationTester.execute_simulation_testing(j_=j_oos, t_=t_)
    simulationTester.make_plots(j_trajectories_plot=10)

    print('--- End s_simulationtesting.py')
