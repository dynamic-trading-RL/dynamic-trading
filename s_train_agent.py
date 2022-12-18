import numpy as np

from enums import OptimizerType, SupervisedRegressorType, InitialEstimateType
from gen_utils.utils import read_ticker
from market_utils.market import read_trading_parameters_market
from reinforcement_learning_utils.agent_trainer import AgentTrainer, read_trading_parameters_training

import warnings

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    np.random.seed(789)

    # -------------------- Input parameters
    # todo: all of these inputs should be read from settings.csv via a dedicated function

    # if zero, the initial estimate of the qvalue function is 0; if random, it is N(0,1)
    initialEstimateType = InitialEstimateType.random
    # if True, the agent uses the model to predict the next step pnl and sig2 for the reward; else, uses the realized
    predict_pnl_for_reward = False
    # if True, the agent averages across supervised regressors in its definition of q_value; else, uses the last one
    average_across_models = True
    # if True, then the agent considers the supervised regressors only up to n<=n_batches, where n is the batch that
    # provided the best reward in the training phase
    use_best_n_batch = True
    # if True, the agent observes the reward GP would obtain and forces its strategy to be GP's if such reward is higher
    # than the one learned automatically
    train_benchmarking_GP_reward = False
    # which optimizer to use in greedy policy
    optimizerType = OptimizerType.shgo
    # whether to make plots of regressor for the training phase
    plot_regressor = True
    # choose which model to use for supervised regression
    supervisedRegressorType = SupervisedRegressorType.ann

    # initial epsilon for eps-greedy policy: at each batch iteration, we do eps <- eps/3
    eps_start = 0.01

    # Market parameters
    ticker = read_ticker()
    riskDriverDynamicsType, factorDynamicsType, riskDriverType = read_trading_parameters_market()

    # Training parameters
    shares_scale, j_episodes, n_batches, t_, parallel_computing, n_cores = read_trading_parameters_training(ticker)

    # -------------------- Execution
    agentTrainer = AgentTrainer(riskDriverDynamicsType=riskDriverDynamicsType,
                                factorDynamicsType=factorDynamicsType,
                                ticker=ticker,
                                riskDriverType=riskDriverType,
                                predict_pnl_for_reward=predict_pnl_for_reward,
                                optimizerType=optimizerType,
                                average_across_models=average_across_models,
                                use_best_n_batch=use_best_n_batch,
                                shares_scale=shares_scale,
                                train_benchmarking_GP_reward=train_benchmarking_GP_reward,
                                plot_regressor=plot_regressor,
                                supervisedRegressorType=supervisedRegressorType,
                                initialEstimateType=initialEstimateType)
    agentTrainer.train(j_episodes=j_episodes, n_batches=n_batches, t_=t_, parallel_computing=parallel_computing,
                       n_cores=n_cores, eps_start=eps_start)

    agentTrainer.agent.dump_q_value_models()

    print('--- End s_train_agent.py')
