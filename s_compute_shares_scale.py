import os
from joblib import dump
import numpy as np
from tqdm import tqdm

from benchmark_agents.agents import AgentMarkowitz
from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType
from gen_utils.utils import read_ticker
from market_utils.market import read_trading_parameters_market, instantiate_market

if __name__ == '__main__':

    np.random.seed(789)

    ticker = read_ticker()
    j_episodes = 10000
    t_ = 50

    riskDriverDynamicsType, factorDynamicsType, riskDriverType = read_trading_parameters_market()
    market = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear,
                                factorDynamicsType=FactorDynamicsType.AR,
                                ticker=ticker,
                                riskDriverType=RiskDriverType.PnL)
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

    shares_scale = np.quantile(a=np.abs(np.array(rescaled_shares_lst)), q=0.99)

    print(f'shares_scale = {shares_scale}')

    dump(shares_scale, os.path.dirname(__file__) + '/data/data_tmp/shares_scale.joblib')
