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
    # Market parameters
    ticker, riskDriverDynamicsType, factorDynamicsType, riskDriverType = read_trading_parameters_market()

    # Training parameters
    (shares_scale, j_episodes, n_batches, t_, parallel_computing, n_cores,
     initialEstimateType, predict_pnl_for_reward, average_across_models, use_best_n_batch, train_benchmarking_GP_reward,
     optimizerType, supervisedRegressorType, eps_start) = read_trading_parameters_training()

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
                                supervisedRegressorType=supervisedRegressorType,
                                initialEstimateType=initialEstimateType)
    agentTrainer.train(j_episodes=j_episodes, n_batches=n_batches, t_=t_, parallel_computing=parallel_computing,
                       n_cores=n_cores, eps_start=eps_start)

    agentTrainer.agent.dump_q_value_models()

    print('--- End s_train_agent.py')
