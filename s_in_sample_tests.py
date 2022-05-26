import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

import numpy as np
import pandas as pd

from benchmark_agents.agents import AgentMarkowitz, AgentGP
from market_utils.market import instantiate_market, read_trading_parameters_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.agent_trainer import read_trading_parameters_training
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import State


def instantiate_agents_benchmark_and_RL(market):

    environment = Environment(market)

    agentMarkowitz = AgentMarkowitz(market)
    agentGP = AgentGP(market)
    agentRL = Agent(environment)
    agentRL.load_q_value_models(n_batches)

    return agentMarkowitz, agentGP, agentRL, environment


def get_factor_pnl_price(market):

    factor_series = market.financialTimeSeries.time_series['factor'].iloc[-t_past:].copy()
    pnl_series = market.financialTimeSeries.time_series['pnl'].copy()
    price_series = market.financialTimeSeries.time_series[ticker].copy()
    factor_pnl_and_price = pd.concat([factor_series, pnl_series, price_series], axis=1)
    factor_pnl_and_price.dropna(inplace=True)

    return factor_pnl_and_price


def compute_in_sample_output():

    global agents, colors

    agents = {'Markowitz': agentMarkowitz, 'GP': agentGP, 'RL': agentRL}
    colors = {'Markowitz': 'm', 'GP': 'g', 'RL': 'r'}
    strategies_all = {}
    trades_all = {}
    cum_values_all = {}
    cum_costs_all = {}
    cum_wealth_all = {}
    sharpe_ratio_all = {}

    for agent_type in ('Markowitz', 'GP', 'RL'):

        strategy = []
        trades = []
        cum_value = []
        cum_cost = []

        current_rescaled_shares = 0.

        for i in range(len(factor_pnl_and_price) - 1):

            factor = factor_pnl_and_price['factor'].iloc[i]
            pnl = factor_pnl_and_price['pnl'].iloc[i + 1]
            price = factor_pnl_and_price[ticker].iloc[i]

            if agent_type == 'RL':
                state = State()
                state.set_trading_attributes(current_factor=factor,
                                             current_rescaled_shares=current_rescaled_shares,
                                             current_other_observable=None,
                                             shares_scale=shares_scale,
                                             current_price=None)
                action = agentRL.policy(state=state)
                rescaled_trade = action.rescaled_trade

                sig2 = market.next_step_sig2(factor=factor, price=price)
                cost_trade = environment.compute_trading_cost(action, sig2)

            else:
                rescaled_trade = agents[agent_type].policy(current_factor=factor,
                                                           current_rescaled_shares=current_rescaled_shares,
                                                           shares_scale=shares_scale)
                cost_trade = agents[agent_type].get_cost_trade(trade=rescaled_trade * shares_scale,
                                                               current_factor=factor,
                                                               price=price)

            current_rescaled_shares += rescaled_trade
            trades.append(rescaled_trade * shares_scale)
            strategy.append(current_rescaled_shares * shares_scale)
            cum_value.append(strategy[-1] * pnl)
            cum_cost.append(cost_trade)

        strategies_all[agent_type] = strategy
        trades_all[agent_type] = trades
        cum_values_all[agent_type] = list(np.cumsum(cum_value))
        cum_costs_all[agent_type] = list(np.cumsum(cum_cost))
        cum_wealth_all[agent_type] = list(np.cumsum(cum_value) - np.cumsum(cum_cost))

        pnl_net = np.diff(np.array(cum_wealth_all[agent_type]))

        sharpe_ratio_all[agent_type] = np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252)

    return strategies_all, trades_all, cum_values_all, cum_costs_all, cum_wealth_all, sharpe_ratio_all


if __name__ == '__main__':

    # -------------------- Input parameters
    # Market parameters
    ticker = 'WTI'
    t_past = 500
    riskDriverDynamicsType, factorDynamicsType, riskDriverType, factorType = read_trading_parameters_market(ticker)

    # Training parameters
    shares_scale, _, n_batches, _, _, _ = read_trading_parameters_training(ticker)

    # -------------------- Execution
    # Instantiate market and environment
    market = instantiate_market(riskDriverDynamicsType=riskDriverDynamicsType,
                                factorDynamicsType=factorDynamicsType,
                                ticker=ticker,
                                riskDriverType=riskDriverType,
                                factorType=factorType)

    # Instantiate agents
    agentMarkowitz, agentGP, agentRL, environment = instantiate_agents_benchmark_and_RL(market)

    # Get factor_series and price_series
    factor_pnl_and_price = get_factor_pnl_price(market)

    # Output
    strategies_all, trades_all, cum_values_all, cum_costs_all, cum_wealth_all, sharpe_ratio_all =\
        compute_in_sample_output()

    # -------------------- Plots
    plt.figure()
    for agent_type in agents.keys():
        plt.plot(factor_pnl_and_price.index[:-1], strategies_all[agent_type], color=colors[agent_type], label=agent_type)
    plt.title('Shares')
    plt.xlabel('Date')
    plt.ylabel('Shares [#]')
    plt.legend()
    plt.savefig('figures/backtesting-shares.png')

    plt.figure()
    for agent_type in agents.keys():
        plt.plot(factor_pnl_and_price.index[:-1], cum_values_all[agent_type], color=colors[agent_type], label=agent_type)
    plt.title('Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio value [$]')
    plt.legend()
    plt.savefig('figures/backtesting-value.png')

    plt.figure()
    for agent_type in agents.keys():
        plt.plot(factor_pnl_and_price.index[:-1], cum_costs_all[agent_type], color=colors[agent_type], label=agent_type)
    plt.title('Cost')
    plt.xlabel('Date')
    plt.ylabel('Cost [$]')
    plt.legend()
    plt.savefig('figures/backtesting-cost.png')

    plt.figure()
    for agent_type in agents.keys():
        plt.plot(factor_pnl_and_price.index[:-1], np.array(cum_values_all[agent_type]) - np.array(cum_costs_all[agent_type]),
                 color=colors[agent_type], label=agent_type)
    plt.title('Wealth')
    plt.xlabel('Date')
    plt.ylabel('Wealth [$]')
    plt.legend()
    plt.savefig('figures/backtesting-wealth.png')

    plt.figure()
    plt.scatter(trades_all['GP'], trades_all['RL'], s=2, alpha=0.5)
    plt.title('GP vs RL trades')
    plt.xlabel('GP trades [#]')
    plt.ylabel('RL trades [#]')
    plt.axis('equal')
    plt.xlim([np.quantile(trades_all['GP'], 0.02), np.quantile(trades_all['GP'], 0.98)])
    plt.ylim([np.quantile(trades_all['RL'], 0.02), np.quantile(trades_all['RL'], 0.98)])
    plt.savefig('figures/backtesting-trades-scatter.png')

    plt.figure()
    plt.bar(sharpe_ratio_all.keys(), sharpe_ratio_all.values()),
    plt.xlabel('Agent')
    plt.ylabel('Sharpe ratio (annualized)')
    plt.title('Sharpe ratio')
    plt.grid()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.))
    plt.savefig('figures/backtesting-sharpe-ratio.png')

    print('--- End s_in_sample_testing.py')
