import numpy as np

from market_utils.market import read_trading_parameters_market
from reinforcement_learning_utils.agent_trainer import AgentTrainer, read_trading_parameters_training

# TODO: understand why with the use_GP_to_train I get all this warnings on optimize
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    np.random.seed(789)

    # -------------------- Input parameters
    compare_to_GP = False
    train_using_GP = False
    plot_regressor = True
    # Market parameters
    ticker = 'WTI'
    riskDriverDynamicsType, factorDynamicsType, riskDriverType, factorType = read_trading_parameters_market(ticker)

    # Training parameters
    shares_scale, j_episodes, n_batches, t_, parallel_computing, n_cores = read_trading_parameters_training(ticker)

    # -------------------- Execution
    agentTrainer = AgentTrainer(riskDriverDynamicsType=riskDriverDynamicsType,
                                factorDynamicsType=factorDynamicsType,
                                ticker=ticker,
                                riskDriverType=riskDriverType,
                                factorType=factorType,
                                shares_scale=shares_scale,
                                compare_to_GP=compare_to_GP,
                                train_using_GP=train_using_GP,
                                plot_regressor=plot_regressor)
    agentTrainer.train(j_episodes=j_episodes, n_batches=n_batches, t_=t_, parallel_computing=parallel_computing,
                       n_cores=n_cores, eps_start=0.01)

    agentTrainer.agent.dump_q_value_models()

    print('--- End s_train_agent.py')
