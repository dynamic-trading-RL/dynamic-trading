from testing_utils.testing import SimulationTester
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # -------------------- Input parameters
    ticker = 'WTI'
    j_ = 1000
    t_ = 50
    j_trajectories_plot = 20

    # -------------------- Execution
    simulationTester = SimulationTester(ticker=ticker)
    simulationTester.execute_simulation_testing(j_=j_, t_=t_)
    simulationTester.make_plots(j_trajectories_plot=j_trajectories_plot)

    print('--- End s_simulationtesting.py')
