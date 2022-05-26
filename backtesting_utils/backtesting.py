import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

from benchmark_agents.agents import AgentMarkowitz, AgentGP
from market_utils.market import read_trading_parameters_market, instantiate_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.agent_trainer import read_trading_parameters_training
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import State


class Backtester:

    def __init__(self, ticker: str, t_past: int):

        self._ticker = ticker
        self._t_past = t_past
        self._read_parameters()

    def _read_parameters(self):

        # Trading parameters
        riskDriverDynamicsType, factorDynamicsType, riskDriverType, factorType =\
            read_trading_parameters_market(self._ticker)

        # Training parameters
        shares_scale, _, n_batches, _, _, _ = read_trading_parameters_training(self._ticker)

        self._riskDriverDynamicsType = riskDriverDynamicsType
        self._factorDynamicsType = factorDynamicsType
        self._riskDriverType = riskDriverType
        self._factorType = factorType
        self._shares_scale = shares_scale
        self._n_batches = n_batches

    def execute_backtesting(self):

        # Instantiate agents
        self._instantiate_agents_and_environment()

        # Get factor_series and price_series
        self._get_factor_pnl_price()

        # Output
        self._compute_in_sample_output()

    def make_plots(self):

        self._plot_shares()
        self._plot_value()
        self._plot_cost()
        self._plot_wealth()
        self._plot_trades_scatter()
        self._plot_sharpe_ratio()

    def _plot_shares(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1], self._strategy_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Shares')
        plt.xlabel('Date')
        plt.ylabel('Shares [#]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting-shares.png')

    def _plot_value(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1], self._cum_value_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Value')
        plt.xlabel('Date')
        plt.ylabel('Portfolio value [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting-value.png')

    def _plot_cost(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1], self._cum_cost_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Cost')
        plt.xlabel('Date')
        plt.ylabel('Cost [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting-cost.png')

    def _plot_wealth(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1],
                     self._cum_wealth_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Wealth')
        plt.xlabel('Date')
        plt.ylabel('Wealth [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting-wealth.png')

    def _plot_trades_scatter(self):
        plt.figure()
        plt.scatter(self._trade_all['GP'], self._trade_all['RL'], s=2, alpha=0.5)
        plt.title('GP vs RL trades')
        plt.xlabel('GP trades [#]')
        plt.ylabel('RL trades [#]')
        plt.axis('equal')
        plt.xlim([np.quantile(self._trade_all['GP'], 0.02), np.quantile(self._trade_all['GP'], 0.98)])
        plt.ylim([np.quantile(self._trade_all['RL'], 0.02), np.quantile(self._trade_all['RL'], 0.98)])
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting-trades-scatter.png')

    def _plot_sharpe_ratio(self):
        plt.figure()
        plt.bar(self._sharpe_ratio_all.keys(), self._sharpe_ratio_all.values()),
        plt.xlabel('Agent')
        plt.ylabel('Sharpe ratio (annualized)')
        plt.title('Sharpe ratio')
        plt.grid()
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.))
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting-sharpe-ratio.png')

    def _instantiate_agents_and_environment(self):

        # Instantiate market, environment and agents
        market = instantiate_market(riskDriverDynamicsType=self._riskDriverDynamicsType,
                                    factorDynamicsType=self._factorDynamicsType,
                                    ticker=self._ticker,
                                    riskDriverType=self._riskDriverType,
                                    factorType=self._factorType)

        environment = Environment(market)

        agentMarkowitz = AgentMarkowitz(market)
        agentGP = AgentGP(market)
        agentRL = Agent(environment)
        agentRL.load_q_value_models(self._n_batches)

        self._agents = {'Markowitz': agentMarkowitz, 'GP': agentGP, 'RL': agentRL}
        self._environment = environment
        self._market = market

    def _get_factor_pnl_price(self):

        factor_series = self._market.financialTimeSeries.time_series['factor'].iloc[-self._t_past:].copy()
        pnl_series = self._market.financialTimeSeries.time_series['pnl'].copy()
        price_series = self._market.financialTimeSeries.time_series[self._ticker].copy()
        factor_pnl_and_price = pd.concat([factor_series, pnl_series, price_series], axis=1)
        factor_pnl_and_price.dropna(inplace=True)

        self._factor_pnl_and_price = factor_pnl_and_price

    def _compute_in_sample_output(self):

        self._colors = {'Markowitz': 'm', 'GP': 'g', 'RL': 'r'}

        self._strategy_all = {}
        self._trade_all = {}
        self._cum_value_all = {}
        self._cum_cost_all = {}
        self._cum_wealth_all = {}
        self._sharpe_ratio_all = {}

        for agent_type in ('Markowitz', 'GP', 'RL'):

            strategy = []
            trades = []
            value = []
            cost = []

            current_rescaled_shares = 0.

            for i in range(len(self._factor_pnl_and_price) - 1):

                factor = self._factor_pnl_and_price['factor'].iloc[i]
                pnl = self._factor_pnl_and_price['pnl'].iloc[i + 1]
                price = self._factor_pnl_and_price[self._ticker].iloc[i]

                if agent_type == 'RL':
                    state = State()
                    state.set_trading_attributes(current_factor=factor,
                                                 current_rescaled_shares=current_rescaled_shares,
                                                 current_other_observable=None,
                                                 shares_scale=self._shares_scale,
                                                 current_price=None)
                    action = self._agents[agent_type].policy(state=state)
                    rescaled_trade = action.rescaled_trade

                    sig2 = self._market.next_step_sig2(factor=factor, price=price)
                    cost_trade = self._environment.compute_trading_cost(action, sig2)

                else:
                    rescaled_trade = self._agents[agent_type].policy(current_factor=factor,
                                                                    current_rescaled_shares=current_rescaled_shares,
                                                                    shares_scale=self._shares_scale)
                    cost_trade = self._agents[agent_type].get_cost_trade(trade=rescaled_trade * self._shares_scale,
                                                                        current_factor=factor,
                                                                        price=price)

                current_rescaled_shares += rescaled_trade

                strategy.append(current_rescaled_shares * self._shares_scale)
                trades.append(rescaled_trade * self._shares_scale)
                value.append(strategy[-1] * pnl)
                cost.append(cost_trade)

            self._strategy_all[agent_type] = strategy
            self._trade_all[agent_type] = trades
            self._cum_value_all[agent_type] = list(np.cumsum(value))
            self._cum_cost_all[agent_type] = list(np.cumsum(cost))
            self._cum_wealth_all[agent_type] = list(np.cumsum(value) - np.cumsum(cost))

            pnl_net = np.diff(np.array(self._cum_wealth_all[agent_type]))

            self._sharpe_ratio_all[agent_type] = np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252)
