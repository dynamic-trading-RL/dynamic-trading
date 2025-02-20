import os
from joblib import dump
import numpy as np
from tqdm import tqdm

from dynamic_trading.benchmark_agents.agents import AgentMarkowitz
from dynamic_trading.enums.enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType
from dynamic_trading.market_utils.market import read_trading_parameters_market, instantiate_market

if __name__ == '__main__':

    np.random.seed(789)

    j_episodes = 10000
    t_ = 50

    ticker, _, _, _ = read_trading_parameters_market()
    market = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear,
                                factorDynamicsType=FactorDynamicsType.AR,
                                ticker=ticker,
                                riskDriverType=RiskDriverType.PnL)
    market.simulate_market_trading(n=0, j_episodes=j_episodes, t_=t_)

    agentMarkowitz = AgentMarkowitz(market=market)

    rescaled_trade_lst = []
    rescaled_shares_lst = []

    for j in tqdm(range(j_episodes), desc='Computing trajectories of Markowitz strategy'):

        rescaled_shares = 0.

        for t in range(t_):

            factor = market.simulations['factor'][j, t]

            rescaled_trade = agentMarkowitz.policy(factor=factor,
                                                   rescaled_shares=rescaled_shares)

            rescaled_shares += rescaled_trade

            rescaled_trade_lst.append(rescaled_trade)
            rescaled_shares_lst.append(rescaled_shares)

    if agentMarkowitz.use_quadratic_cost_in_markowitz:
        shares_scale = 2.15 * np.quantile(a=np.abs(np.array(rescaled_shares_lst)), q=0.95)
    else:
        shares_scale = np.quantile(a=np.abs(np.array(rescaled_shares_lst)), q=0.95)

    print(f'shares_scale = {shares_scale}')

    dump(shares_scale, os.path.dirname(__file__) + '/resources/data/data_tmp/shares_scale.joblib')
