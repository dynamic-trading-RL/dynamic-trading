from market_utils.market import read_market_parameters
from reinforcement_learning_utils.agent_trainer import AgentTrainer, read_training_parameters

if __name__ == '__main__':

    # Market parameters
    ticker = 'WTI'
    riskDriverDynamicsType, factorDynamicsType, riskDriverType, factorType = read_market_parameters(ticker)

    # Training parameters
    shares_scale, j_episodes, n_batches, t_, parallel_computing, n_cores = read_training_parameters(ticker)

    agentTrainer = AgentTrainer(riskDriverDynamicsType=riskDriverDynamicsType,
                                factorDynamicsType=factorDynamicsType,
                                ticker=ticker,
                                riskDriverType=riskDriverType,
                                factorType=factorType,
                                shares_scale=shares_scale)
    agentTrainer.train(j_episodes=j_episodes, n_batches=n_batches, t_=t_,
                       parallel_computing=parallel_computing,
                       n_cores=n_cores)

    agentTrainer.agent.dump_q_value_models()
    print(agentTrainer.reward)