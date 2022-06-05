from market_utils.market import read_trading_parameters_market
from reinforcement_learning_utils.agent_trainer import AgentTrainer, read_trading_parameters_training

if __name__ == '__main__':

    # -------------------- Input parameters
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
                                shares_scale=shares_scale)
    agentTrainer.train(j_episodes=j_episodes, n_batches=n_batches, t_=t_,
                       parallel_computing=parallel_computing,
                       n_cores=n_cores,
                       eps_start=0.001)

    agentTrainer.agent.dump_q_value_models()

    print('--- End s_train_agent.py')
