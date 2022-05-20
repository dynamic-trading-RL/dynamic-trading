import matplotlib.pyplot as plt

from benchmark_agents.agents import AgentMarkowitz, AgentGP
from market_utils.market import instantiate_market, read_trading_parameters_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.agent_trainer import read_trading_parameters_training
from reinforcement_learning_utils.state_action_utils import State

if __name__ == '__main__':

    # Market parameters
    ticker = 'WTI'
    riskDriverDynamicsType, factorDynamicsType, riskDriverType, factorType = read_trading_parameters_market(ticker)

    # Tradind parameters
    shares_scale, _, n_batches, _, _, _ = read_trading_parameters_training(ticker)

    # Instantiate market and environment
    market = instantiate_market(riskDriverDynamicsType=riskDriverDynamicsType,
                                factorDynamicsType=factorDynamicsType,
                                ticker=ticker,
                                riskDriverType=riskDriverType,
                                factorType=factorType)
    environment = Environment(market)

    # Instantiate agents
    agentMarkowitz = AgentMarkowitz(market)
    agentGP = AgentGP(market)
    agentRL = Agent(environment)
    agentRL.load_q_value_models(n_batches)

    # Get factor_series
    factor_series = market.financialTimeSeries.time_series['factor'].iloc[-50:]

    # Strategies
    plt.figure()

    # Markowitz strategy
    markowitz_strategy = []
    current_rescaled_shares = 0.
    for factor in factor_series:
        rescaled_trade = agentMarkowitz.policy(current_factor=factor,
                                               current_rescaled_shares=current_rescaled_shares,
                                               shares_scale=shares_scale)
        current_rescaled_shares += rescaled_trade
        markowitz_strategy.append(current_rescaled_shares * shares_scale)

    plt.plot(factor_series.index, markowitz_strategy, color='m', label='Markowitz')

    # GP strategy
    GP_strategy = []
    current_rescaled_shares = 0.
    for factor in factor_series:
        rescaled_trade = agentGP.policy(current_factor=factor,
                                        current_rescaled_shares=current_rescaled_shares,
                                        shares_scale=shares_scale)
        current_rescaled_shares += rescaled_trade
        GP_strategy.append(current_rescaled_shares * shares_scale)

    plt.plot(factor_series.index, GP_strategy, color='g', label='GP')

    # RL strategy
    RL_strategy = []
    current_rescaled_shares = 0.
    for factor in factor_series:
        state = State()
        state.set_trading_attributes(current_factor=factor,
                                     current_rescaled_shares=current_rescaled_shares,
                                     current_other_observable=None,
                                     shares_scale=shares_scale,
                                     current_price=None)
        action = agentRL.policy(state=state)
        rescaled_trade = action.rescaled_trade
        current_rescaled_shares += rescaled_trade
        RL_strategy.append(current_rescaled_shares * shares_scale)

    plt.plot(factor_series.index, RL_strategy, color='r', label='RL')

    plt.legend()

    plt.savefig('../figures/backtesting.png')
