import numpy as np

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
     initialQvalueEstimateType, predict_pnl_for_reward, average_across_models, use_best_n_batch, train_benchmarking_GP_reward,
     optimizerType, supervisedRegressorType, eps_start, ann_architecture, early_stopping, max_iter, n_iter_no_change,
     activation, alpha_sarsa, decrease_eps, random_initial_state) = read_trading_parameters_training()

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
                                initialQvalueEstimateType=initialQvalueEstimateType,
                                ann_architecture=ann_architecture,
                                early_stopping=early_stopping,
                                max_iter=max_iter,
                                n_iter_no_change=n_iter_no_change,
                                activation=activation,
                                alpha_sarsa=alpha_sarsa,
                                decrease_eps=decrease_eps,
                                random_initial_state=random_initial_state)
    agentTrainer.train(j_episodes=j_episodes, n_batches=n_batches, t_=t_, parallel_computing=parallel_computing,
                       n_cores=n_cores, eps_start=eps_start)

    agentTrainer.agent.dump_q_value_models()

    print('--- End s_train_agent.py')
