import numpy as np
from tqdm import tqdm

from benchmark_agents.agents import AgentMarkowitz
from market_utils.market import read_trading_parameters_market, instantiate_market

if __name__ == '__main__':

    ticker = 'WTI'
    j_episodes = 10000
    t_ = 50

    riskDriverDynamicsType, factorDynamicsType, riskDriverType, factorType = read_trading_parameters_market(ticker)
    market = instantiate_market(riskDriverDynamicsType=riskDriverDynamicsType,
                                factorDynamicsType=factorDynamicsType,
                                ticker=ticker,
                                riskDriverType=riskDriverType)
    market.simulate_market_trading(n=0, j_episodes=j_episodes, t_=t_)

    agentMarkowitz = AgentMarkowitz(market=market)

    rescaled_trade_lst = []
    rescaled_shares_lst = []

    for j in tqdm(range(j_episodes), desc='Computing trajectories of Markowitz strategy'):

        current_rescaled_shares = 0.

        for t in range(t_):

            current_factor = market.simulations['factor'][j, t]

            rescaled_trade = agentMarkowitz.policy(current_factor=current_factor,
                                                   current_rescaled_shares=current_rescaled_shares)

            current_rescaled_shares += rescaled_trade

            rescaled_trade_lst.append(rescaled_trade)
            rescaled_shares_lst.append(current_rescaled_shares)

    M = np.quantile(a=np.abs(np.array(rescaled_shares_lst)), q=0.99)

    print(f'M = {M}')
