import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

from tqdm import tqdm

from benchmark_agents.agents import AgentMarkowitz, AgentGP
from market_utils.market import read_trading_parameters_market, instantiate_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.agent_trainer import read_trading_parameters_training
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import State
from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType, FactorType


class Tester:

    def __init__(self, ticker):

        self._ticker = ticker
        self._read_parameters()

        self._colors = {'Markowitz': 'm', 'GP': 'g', 'RL': 'r'}

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

    def _instantiate_agents_and_environment(self):

        # Instantiate market for benchmark agents: market for them is necessarily Linear - AR
        market_benchmark = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear,
                                              factorDynamicsType=FactorDynamicsType.AR,
                                              ticker=self._ticker,
                                              riskDriverType=RiskDriverType.PnL,
                                              factorType=FactorType.Observable)

        agentMarkowitz = AgentMarkowitz(market_benchmark)
        agentGP = AgentGP(market_benchmark)

        # Instantiate market for RL, environment and RL agent
        market = instantiate_market(riskDriverDynamicsType=self._riskDriverDynamicsType,
                                    factorDynamicsType=self._factorDynamicsType,
                                    ticker=self._ticker,
                                    riskDriverType=self._riskDriverType,
                                    factorType=self._factorType)
        environment = Environment(market)
        agentRL = Agent(environment)
        agentRL.load_q_value_models(self._n_batches)

        self._agents = {'Markowitz': agentMarkowitz, 'GP': agentGP, 'RL': agentRL}
        self._environment = environment
        self._market = market
        self._market_benchmark = market_benchmark


class BackTester(Tester):

    def __init__(self, ticker: str):

        super().__init__(ticker)
        self._read_out_of_sample_proportion_len()

    def execute_backtesting(self):

        # Instantiate agents
        super()._instantiate_agents_and_environment()

        # Get factor_series and price_series
        self._get_factor_pnl_price()

        # Output
        self._compute_backtesting_output()

        # Print Sharpe ratios
        print(self._sharpe_ratio_all)

    def make_plots(self):

        self._plot_shares()
        self._plot_value()
        self._plot_cost()
        self._plot_risk()
        self._plot_wealth()
        self._plot_wealth_net_risk()
        self._plot_trades_scatter()
        self._plot_sharpe_ratio()

    def _read_out_of_sample_proportion_len(self):
        filename = os.path.dirname(os.path.dirname(__file__)) + \
                   '/data/financial_time_series_data/financial_time_series_info/' + self._ticker + '-info.csv'
        df_info = pd.read_csv(filename, index_col=0)
        self._out_of_sample_proportion_len = int(df_info.loc['out_of_sample_proportion_len'][0])

    def _get_factor_pnl_price(self):

        factor_series =\
            self._market.financialTimeSeries.time_series['factor'].iloc[-self._out_of_sample_proportion_len:].copy()
        pnl_series = self._market.financialTimeSeries.time_series['pnl'].copy()
        price_series = self._market.financialTimeSeries.time_series[self._ticker].copy()
        factor_pnl_and_price = pd.concat([factor_series, pnl_series, price_series], axis=1)
        factor_pnl_and_price.rename(columns={self._ticker: 'price'}, inplace=True)
        factor_pnl_and_price.dropna(inplace=True)

        self._factor_pnl_and_price = factor_pnl_and_price

    def _compute_backtesting_output(self):

        self._strategy_all = {}
        self._trade_all = {}
        self._cum_value_all = {}
        self._cum_cost_all = {}
        self._cum_risk_all = {}
        self._cum_wealth_all = {}
        self._cum_wealth_net_risk_all = {}
        self._sharpe_ratio_all = {}

        for agent_type in self._agents.keys():

            strategy = []
            trades = []
            value = []
            cost = []
            risk = []

            current_rescaled_shares = 0.

            for i in tqdm(range(len(self._factor_pnl_and_price) - 1),
                          desc='Computing ' + agent_type + ' strategy.'):

                factor = self._factor_pnl_and_price['factor'].iloc[i]
                pnl = self._factor_pnl_and_price['pnl'].iloc[i + 1]
                price = self._factor_pnl_and_price['price'].iloc[i]

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
                    risk_trade = self._environment.compute_trading_risk(state, sig2)

                else:
                    rescaled_trade = self._agents[agent_type].policy(current_factor=factor,
                                                                     current_rescaled_shares=current_rescaled_shares,
                                                                     shares_scale=self._shares_scale)
                    cost_trade = self._agents[agent_type].compute_trading_cost(trade=rescaled_trade * self._shares_scale,
                                                                               current_factor=factor,
                                                                               price=price)
                    risk_trade = \
                        self._agents[agent_type].compute_trading_risk(current_factor=factor,
                                                                      price=price,
                                                                      current_rescaled_shares=current_rescaled_shares,
                                                                      shares_scale=self._shares_scale)

                current_rescaled_shares += rescaled_trade

                strategy.append(current_rescaled_shares * self._shares_scale)
                trades.append(rescaled_trade * self._shares_scale)
                value.append(strategy[-1] * pnl)
                cost.append(cost_trade)
                risk.append(risk_trade)

            self._strategy_all[agent_type] = strategy
            self._trade_all[agent_type] = trades
            self._cum_value_all[agent_type] = list(np.cumsum(value))
            self._cum_cost_all[agent_type] = list(np.cumsum(cost))
            self._cum_risk_all[agent_type] = list(np.cumsum(risk))
            self._cum_wealth_all[agent_type] = list(np.cumsum(value) - np.cumsum(cost))
            self._cum_wealth_net_risk_all[agent_type] = list(np.cumsum(value) - np.cumsum(cost) - np.cumsum(risk))

            pnl_net = np.diff(np.array(self._cum_wealth_all[agent_type]))

            self._sharpe_ratio_all[agent_type] = np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252)

    def _plot_shares(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1], self._strategy_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Shares')
        plt.xlabel('Date')
        plt.ylabel('Shares [#]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-shares.png')

    def _plot_value(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1], self._cum_value_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Value')
        plt.xlabel('Date')
        plt.ylabel('Portfolio value [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-value.png')

    def _plot_cost(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1], self._cum_cost_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Cost')
        plt.xlabel('Date')
        plt.ylabel('Cost [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-cost.png')

    def _plot_risk(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1], self._cum_risk_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Risk')
        plt.xlabel('Date')
        plt.ylabel('Risk')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-risk.png')

    def _plot_wealth(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1],
                     self._cum_wealth_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Wealth')
        plt.xlabel('Date')
        plt.ylabel('Wealth = Value - Cost [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-wealth.png')

    def _plot_wealth_net_risk(self):
        plt.figure()
        for agent_type in self._agents.keys():
            plt.plot(self._factor_pnl_and_price.index[:-1],
                     self._cum_wealth_net_risk_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Wealth net Risk')
        plt.xlabel('Date')
        plt.ylabel('Wealth net Risk = Value - Cost - Risk [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-wealth-net-risk.png')

    def _plot_trades_scatter(self):
        plt.figure()
        plt.scatter(self._trade_all['GP'], self._trade_all['RL'], s=2, alpha=0.5)
        plt.title('GP vs RL trades')
        plt.xlabel('GP trades [#]')
        plt.ylabel('RL trades [#]')
        plt.axis('equal')
        xlim = [np.quantile(self._trade_all['GP'], 0.02), np.quantile(self._trade_all['GP'], 0.98)]
        ylim = [np.quantile(self._trade_all['RL'], 0.02), np.quantile(self._trade_all['RL'], 0.98)]
        plt.plot(xlim, ylim, color='r', label='45° line')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-trades-scatter.png')

    def _plot_sharpe_ratio(self):
        plt.figure()
        plt.bar(self._sharpe_ratio_all.keys(), self._sharpe_ratio_all.values()),
        plt.xlabel('Agent')
        plt.ylabel('Realized Sharpe ratio (annualized)')
        plt.title('Realized Sharpe ratio')
        plt.grid()
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.))
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-sharpe-ratio.png')


class SimulationTester(Tester):

    def __init__(self, ticker: str):

        super().__init__(ticker)

    def execute_simulation_testing(self, j_, t_):

        self.j_ = j_
        self.t_ = t_

        # Instantiate agents
        super()._instantiate_agents_and_environment()

        # Simulate factor_series and price_series
        self._simulate_factor_pnl_price()

        # Output
        self._compute_simulation_testing_output()

        # Print outputs
        print()

    def make_plots(self):

        self._plot_shares()
        self._plot_value()
        self._plot_cost()
        self._plot_risk()
        self._plot_wealth()
        self._plot_wealth_net_risk()
        self._plot_trades_scatter()
        self._plot_sharpe_ratio()

    def _simulate_factor_pnl_price(self):

        self._factor_pnl_and_price_sims = {}

        self._environment.market.simulate(j_=self.j_, t_=self.t_)

        start_date = self._market.financialTimeSeries.info.loc['end_date'].item()
        dates = pd.date_range(start=start_date, periods=self.t_)

        for data_type in ('factor', 'pnl', 'price'):

            sims = pd.DataFrame(data=self._environment.market.simulations[data_type], columns=dates)
            sims = pd.melt(sims, var_name='date', ignore_index=False)
            sims.set_index('date', append=True, inplace=True)
            sims = sims.squeeze()
            sims.name = data_type

            self._factor_pnl_and_price_sims[data_type] = sims

    def _compute_simulation_testing_output(self):

        pass # TODO: implement



if __name__ == '__main__':

    simulationTester = SimulationTester('WTI')

    simulationTester.execute_simulation_testing(100, 20)
