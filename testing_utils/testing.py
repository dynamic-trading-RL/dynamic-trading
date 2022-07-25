from functools import partial
import multiprocessing as mp

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
from reinforcement_learning_utils.state_action_utils import State, Action
from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType, FactorType, ModeType


class Tester:

    def __init__(self, ticker, use_assessment_period: bool = False, assessment_proportion: float = 0.1):

        # TODO: evaluating the introduction of the concept of assesment period

        self._ticker = ticker
        self._use_assessment_period = use_assessment_period
        self._assessment_proportion = assessment_proportion
        self._read_parameters()

        self._colors = {'Markowitz': 'm', 'GP': 'g', 'RL': 'r'}

    def _read_parameters(self):

        # Trading parameters
        riskDriverDynamicsType, factorDynamicsType, riskDriverType, factorType =\
            read_trading_parameters_market(self._ticker)

        # Training parameters
        shares_scale, _, n_batches, _, parallel_computing, n_cores = read_trading_parameters_training(self._ticker)

        self._riskDriverDynamicsType = riskDriverDynamicsType
        self._factorDynamicsType = factorDynamicsType
        self._riskDriverType = riskDriverType
        self._factorType = factorType
        self._shares_scale = shares_scale
        self._n_batches = n_batches
        self._parallel_computing = parallel_computing
        self._n_cores = n_cores

    def _instantiate_agents_and_environment(self):

        # Instantiate market for benchmark agents: market for them is necessarily Linear - AR
        market_benchmark = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear,
                                              factorDynamicsType=FactorDynamicsType.AR,
                                              ticker=self._ticker,
                                              riskDriverType=RiskDriverType.PnL,
                                              factorType=FactorType.Observable,
                                              modeType=ModeType.OutOfSample)

        agentMarkowitz = AgentMarkowitz(market_benchmark)
        agentGP = AgentGP(market_benchmark)

        # Instantiate market for RL, environment and RL agent
        market = instantiate_market(riskDriverDynamicsType=self._riskDriverDynamicsType,
                                    factorDynamicsType=self._factorDynamicsType,
                                    ticker=self._ticker,
                                    riskDriverType=self._riskDriverType,
                                    factorType=self._factorType,
                                    modeType=ModeType.OutOfSample)
        environment = Environment(market)
        agentRL = Agent(environment)
        agentRL.load_q_value_models(self._n_batches)

        self._agents = {'Markowitz': agentMarkowitz, 'GP': agentGP, 'RL': agentRL}
        self._environment = environment
        self._market = market
        self._market_benchmark = market_benchmark

    def _initialize_output_dicts(self):
        self._strategy_all = {}
        self._trade_all = {}
        self._cum_value_all = {}
        self._cum_cost_all = {}
        self._cum_risk_all = {}
        self._cum_wealth_all = {}
        self._cum_wealth_net_risk_all = {}
        self._sharpe_ratio_all = {}

    def _get_time_series(self):
        factor_series = self._factor_pnl_and_price['factor']
        pnl_series = self._factor_pnl_and_price['pnl']
        price_series = self._factor_pnl_and_price['price']
        return factor_series, pnl_series, price_series

    def _initialize_output_list_for_agent(self):
        strategy = []
        trades = []
        value = []
        cost = []
        risk = []
        return cost, risk, strategy, trades, value

    def _get_current_factor_pnl_price(self, date, dates, factor_series, pnl_series, price_series):
        i_loc = dates.get_loc(date)
        next_date = dates[i_loc + 1]
        factor = factor_series.loc[date]
        pnl = pnl_series.loc[next_date]
        price = price_series.loc[date]
        return factor, pnl, price

    def _compute_outputs_for_time_t(self, agent_type, current_rescaled_shares, factor, price):
        if agent_type == 'RL':
            state = State()

            rescaled_trade_GP = self._agents['GP'].policy(current_factor=factor,
                                                          current_rescaled_shares=current_rescaled_shares,
                                                          shares_scale=self._shares_scale)
            action_GP = Action()
            action_GP.set_trading_attributes(rescaled_trade=rescaled_trade_GP, shares_scale=self._shares_scale)

            state.set_trading_attributes(current_factor=factor,
                                         current_rescaled_shares=current_rescaled_shares,
                                         current_other_observable=None,
                                         shares_scale=self._shares_scale,
                                         current_price=price,
                                         action_GP=action_GP)
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
            risk_trade =\
                self._agents[agent_type].compute_trading_risk(current_factor=factor,
                                                              price=price,
                                                              current_rescaled_shares=current_rescaled_shares,
                                                              shares_scale=self._shares_scale)
        current_rescaled_shares += rescaled_trade
        return cost_trade, current_rescaled_shares, rescaled_trade, risk_trade

    def _update_lists(self, cost, cost_trade, current_rescaled_shares, pnl, rescaled_trade, risk, risk_trade,
                      strategy, trades, value):
        strategy.append(current_rescaled_shares * self._shares_scale)
        trades.append(rescaled_trade * self._shares_scale)
        value.append(strategy[-1] * pnl)
        cost.append(cost_trade)
        risk.append(risk_trade)


class BackTester(Tester):

    def __init__(self, ticker: str):

        super().__init__(ticker)
        self._read_out_of_sample_proportion_len()

    def execute_backtesting(self):

        # Instantiate agents
        self._instantiate_agents_and_environment()

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

        length = self._out_of_sample_proportion_len

        factor_series = self._market.financialTimeSeries.time_series['factor'].iloc[-length:].copy()
        pnl_series = self._market.financialTimeSeries.time_series['pnl'].copy()
        price_series = self._market.financialTimeSeries.time_series[self._ticker].copy()
        factor_pnl_and_price = pd.concat([factor_series, pnl_series, price_series], axis=1)
        factor_pnl_and_price.rename(columns={self._ticker: 'price'}, inplace=True)
        factor_pnl_and_price.dropna(inplace=True)

        self._factor_pnl_and_price = factor_pnl_and_price
        self.t_ = len(self._factor_pnl_and_price)
        self.t_assessment = int(self.t_ * self._assessment_proportion)

    def _compute_backtesting_output(self):
        # TODO: unify this and the corresponding in SimulationTester and move as much as possible to superclass

        # initialize output dicts: {agent_type: list}
        self._initialize_output_dicts()

        # get time series
        factor_series, pnl_series, price_series = self._get_time_series()

        # get dates
        dates = factor_series.index

        for agent_type in self._agents.keys():

            cost, risk, strategy, trades, value = self._initialize_output_list_for_agent()

            current_rescaled_shares = 0.

            for date in tqdm(dates[:-1], desc='Computing ' + agent_type + ' strategy'):

                factor, pnl, price = self._get_current_factor_pnl_price(date, dates, factor_series, pnl_series,
                                                                        price_series)

                cost_trade, current_rescaled_shares, rescaled_trade, risk_trade = self._compute_outputs_for_time_t(
                    agent_type, current_rescaled_shares, factor, price)

                self._update_lists(cost, cost_trade, current_rescaled_shares, pnl, rescaled_trade, risk, risk_trade,
                                   strategy, trades, value)

            strategy = np.array(strategy)
            trades = np.array(trades)

            self._strategy_all[agent_type] = strategy
            self._trade_all[agent_type] = trades
            self._cum_value_all[agent_type] = np.cumsum(value)
            self._cum_cost_all[agent_type] = np.cumsum(cost)
            self._cum_risk_all[agent_type] = np.cumsum(risk)
            self._cum_wealth_all[agent_type] = np.cumsum(value) - np.cumsum(cost)
            self._cum_wealth_net_risk_all[agent_type] = np.cumsum(value) - np.cumsum(cost) - np.cumsum(risk)

            pnl_net = np.diff(self._cum_wealth_all[agent_type])

            self._sharpe_ratio_all[agent_type] = np.mean(pnl_net) / np.std(pnl_net) * np.sqrt(252)

    def _get_dates_plot(self):
        if self._use_assessment_period:
            dates = self._factor_pnl_and_price.index[self.t_assessment:-1]
        else:
            dates = self._factor_pnl_and_price.index[:-1]
        return dates

    def _plot_shares(self):

        dates = self._get_dates_plot()

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
        for agent_type in self._agents.keys():
            plt.plot(dates, self._strategy_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)

        plt.title('Shares')
        plt.xlabel('Date')
        plt.ylabel('Shares [#]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-shares.png')

    def _plot_value(self):

        dates = self._get_dates_plot()

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
        for agent_type in self._agents.keys():
            plt.plot(dates, self._cum_value_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Value')
        plt.xlabel('Date')
        plt.ylabel('Portfolio value [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-value.png')

    def _plot_cost(self):

        dates = self._get_dates_plot()

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
        for agent_type in self._agents.keys():
            plt.plot(dates, self._cum_cost_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Cost')
        plt.xlabel('Date')
        plt.ylabel('Cost [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-cost.png')

    def _plot_risk(self):

        dates = self._get_dates_plot()

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
        for agent_type in self._agents.keys():
            plt.plot(dates, self._cum_risk_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Risk')
        plt.xlabel('Date')
        plt.ylabel('Risk')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-risk.png')

    def _plot_wealth(self):

        dates = self._get_dates_plot()

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
        for agent_type in self._agents.keys():
            plt.plot(dates,
                     self._cum_wealth_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Wealth')
        plt.xlabel('Date')
        plt.ylabel('Wealth = Value - Cost [$]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-wealth.png')

    def _plot_wealth_net_risk(self):

        dates = self._get_dates_plot()

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
        for agent_type in self._agents.keys():
            plt.plot(dates,
                     self._cum_wealth_net_risk_all[agent_type],
                     color=self._colors[agent_type], label=agent_type)
        plt.title('Wealth net Risk')
        plt.xlabel('Date')
        plt.ylabel('Wealth net Risk = Value - Cost - Risk')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-wealth-net-risk.png')

    def _plot_trades_scatter(self):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
        plt.scatter(self._trade_all['GP'], self._trade_all['RL'], s=2, alpha=0.5)
        plt.title('GP vs RL trades')
        plt.xlabel('GP trades [#]')
        plt.ylabel('RL trades [#]')
        plt.axis('equal')
        xlim = [min(np.quantile(self._trade_all['GP'], 0.02), np.quantile(self._trade_all['RL'], 0.02)),
                max(np.quantile(self._trade_all['GP'], 0.98), np.quantile(self._trade_all['RL'], 0.98))]
        plt.plot(xlim, xlim, color='r', label='45° line')
        plt.xlim(xlim)
        plt.ylim(xlim)
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/backtesting/' + self._ticker
                    + '-backtesting-trades-scatter.png')

    def _plot_sharpe_ratio(self):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
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

    # TODO: Finalize implementation of this class

    def __init__(self, ticker: str):

        super().__init__(ticker)

    def execute_simulation_testing(self, j_, t_):

        self.j_ = j_
        self.t_ = t_

        # Instantiate agents
        self._instantiate_agents_and_environment()

        # Simulate factor_series and price_series
        self._simulate_factor_pnl_price()

        # Output
        self._compute_simulation_testing_output()

        # Print outputs
        print()

    def make_plots(self, j_trajectories_plot):

        self._plot_shares(j_trajectories_plot)
        self._plot_value()
        self._plot_cost()
        self._plot_risk()
        self._plot_wealth()
        self._plot_wealth_net_risk()
        self._plot_trades_scatter()
        self._plot_sharpe_ratio()

    def _plot_shares(self, j_trajectories_plot):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)

        dates = self._factor_pnl_and_price['factor'].index.get_level_values('date').unique()

        for agent_type in self._agents.keys():

            plt.plot(dates[:-1], self._strategy_all[agent_type][0], color=self._colors[agent_type], label=agent_type,
                     alpha=0.5)

            for j in range(1, min(j_trajectories_plot, self.j_)):

                plt.plot(dates[:-1], self._strategy_all[agent_type][j], color=self._colors[agent_type])

        plt.title('Shares')
        plt.xlabel('Date')
        plt.ylabel('Shares [#]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/simulationtesting/' + self._ticker
                    + '-simulationtesting-shares.png')

    def _plot_value(self):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)

        for agent_type in self._agents.keys():

            values = self._cum_value_all[agent_type][:, -1]
            mean = values.mean()
            std = values.std()

            plt.hist(values, color=self._colors[agent_type], alpha=0.3,
                     label=f'{agent_type} : (mean, std) = ({mean:.2f}, {std:.2f})', density=True)

        plt.title('Value')
        plt.xlabel('Portfolio Value [$]')
        plt.yticks([])
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/simulationtesting/' + self._ticker
                    + '-simulationtesting-value.png')

    def _plot_cost(self):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)

        for agent_type in self._agents.keys():

            values = self._cum_cost_all[agent_type][:, -1]
            mean = values.mean()
            std = values.std()

            plt.hist(values, color=self._colors[agent_type], alpha=0.3,
                     label=f'{agent_type} : (mean, std) = ({mean:.2f}, {std:.2f})', density=True)

        plt.title('Cost')
        plt.xlabel('Cost [$]')
        plt.yticks([])
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/simulationtesting/' + self._ticker
                    + '-simulationtesting-cost.png')

    def _plot_risk(self):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)

        for agent_type in self._agents.keys():

            values = self._cum_risk_all[agent_type][:, -1]
            mean = values.mean()
            std = values.std()

            plt.hist(values, color=self._colors[agent_type], alpha=0.3,
                     label=f'{agent_type} : (mean, std) = ({mean:.2f}, {std:.2f})', density=True)

        plt.title('Risk')
        plt.xlabel('Risk')
        plt.yticks([])
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/simulationtesting/' + self._ticker
                    + '-simulationtesting-risk.png')

    def _plot_wealth(self):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)

        for agent_type in self._agents.keys():

            values = self._cum_wealth_all[agent_type][:, -1]
            mean = values.mean()
            std = values.std()

            plt.hist(values, color=self._colors[agent_type], alpha=0.3,
                     label=f'{agent_type} : (mean, std) = ({mean:.2f}, {std:.2f})', density=True)

        plt.title('Wealth')
        plt.xlabel('Wealth = Value - Cost [$]')
        plt.yticks([])
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/simulationtesting/' + self._ticker
                    + '-simulationtesting-wealth.png')

    def _plot_wealth_net_risk(self):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)

        for agent_type in self._agents.keys():

            values = self._cum_wealth_net_risk_all[agent_type][:, -1]
            mean = values.mean()
            std = values.std()

            plt.hist(values, color=self._colors[agent_type], alpha=0.3,
                     label=f'{agent_type} : (mean, std) = ({mean:.2f}, {std:.2f})', density=True)

        plt.title('Wealth net Risk')
        plt.xlabel('Wealth net Risk = Value - Cost - Risk')
        plt.yticks([])
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/simulationtesting/' + self._ticker
                    + '-simulationtesting-wealth-net-risk.png')

    def _plot_trades_scatter(self):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)

        xx = self._trade_all['GP'].flatten()
        yy = self._trade_all['RL'].flatten()

        xlim = [min(np.quantile(xx, 0.02), np.quantile(yy, 0.02)),
                max(np.quantile(xx, 0.98), np.quantile(yy, 0.98))]

        plt.scatter(xx, yy, s=2, alpha=0.5)
        plt.plot(xlim, xlim, label='45° line', color='r')

        plt.title('GP vs RL trades')
        plt.xlabel('GP trades [#]')
        plt.ylabel('RL trades [#]')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/simulationtesting/' + self._ticker
                    + '-simulationtesting-trades-scatter.png')

    def _plot_sharpe_ratio(self):

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)

        for agent_type in self._agents.keys():

            values = self._sharpe_ratio_all[agent_type]
            mean = values.mean()
            std = values.std()

            plt.hist(values, color=self._colors[agent_type], alpha=0.3,
                     label=f'{agent_type} : (mean, std) = ({mean:.2f}, {std:.2f})', density=True)

        plt.title('Realized Sharpe ratio')
        plt.xlabel('Realized Sharpe ratio (annualized)')
        plt.yticks([])
        plt.legend()
        plt.savefig(os.path.dirname(os.path.dirname(__file__)) + '/figures/simulationtesting/' + self._ticker
                    + '-simulationtesting-sharpe-ratio.png')

    def _simulate_factor_pnl_price(self):

        self._factor_pnl_and_price = {}

        self._environment.market.simulate(j_=self.j_, t_=self.t_)

        start_date = self._market.financialTimeSeries.info.loc['end_date'].item()
        dates = pd.date_range(start=start_date, periods=self.t_)

        for data_type in ('factor', 'pnl', 'price'):

            sims = pd.DataFrame(data=self._environment.market.simulations[data_type], columns=dates)
            sims.index.name = 'simulation'
            sims = pd.melt(sims, var_name='date', ignore_index=False)
            sims.set_index('date', append=True, inplace=True)
            sims = sims.squeeze()
            sims.name = data_type

            self._factor_pnl_and_price[data_type] = sims

    def _compute_simulation_testing_output(self):

        # TODO: unify this and the corresponding in BackTester and move as much as possible to superclass

        # initialize output dicts: {agent_type: {j: list}}
        self._initialize_output_dicts()

        # get time series
        factor_series_all_j, pnl_series_all_j, price_series_all_j = self._get_time_series()

        # get dates
        dates = factor_series_all_j.index.get_level_values('date').unique()

        # get simulation index
        j_index = factor_series_all_j.index.get_level_values('simulation').unique()

        for agent_type in self._agents.keys():

            cost, risk, strategy, trades, value = self._initialize_output_list_for_agent()

            if self._parallel_computing:

                compute_outputs_iter_j_partial = partial(self._compute_outputs_iter_j,
                                                         factor_series_all_j=factor_series_all_j,
                                                         pnl_series_all_j=pnl_series_all_j,
                                                         price_series_all_j=price_series_all_j,
                                                         dates=dates,
                                                         agent_type=agent_type)

                p = mp.Pool(self._n_cores)

                outputs = list(tqdm(p.imap_unordered(func=compute_outputs_iter_j_partial,
                                                     iterable=j_index,
                                                     chunksize=int(self.j_ / self._n_cores)),
                                    total=self.j_,
                                    desc='Computing simulations of ' + agent_type + ' strategy'))

                p.close()
                p.join()

                for j in range(len(outputs)):
                    strategy.append(outputs[j][0])
                    trades.append(outputs[j][1])
                    value.append(outputs[j][2])
                    cost.append(outputs[j][3])
                    risk.append(outputs[j][4])

            else:

                for j in tqdm(j_index, desc='Computing simulations of ' + agent_type + ' strategy'):

                    strategy_j, trades_j, value_j, cost_j, risk_j =\
                        self._compute_outputs_iter_j(j, factor_series_all_j, pnl_series_all_j, price_series_all_j,
                                                     dates, agent_type)

                    strategy.append(strategy_j)
                    trades.append(trades_j)
                    value.append(value_j)
                    cost.append(cost_j)
                    risk.append(risk_j)

            self._strategy_all[agent_type] = np.array(strategy)
            self._trade_all[agent_type] = np.array(trades)
            self._cum_value_all[agent_type] = np.cumsum(value, axis=1)
            self._cum_cost_all[agent_type] = np.cumsum(cost, axis=1)
            self._cum_risk_all[agent_type] = np.cumsum(risk, axis=1)
            self._cum_wealth_all[agent_type] = np.cumsum(value, axis=1) -\
                                               np.cumsum(cost, axis=1)
            self._cum_wealth_net_risk_all[agent_type] = np.cumsum(value, axis=1) -\
                                                        np.cumsum(cost, axis=1) -\
                                                        np.cumsum(risk, axis=1)

            pnl_net = np.diff(self._cum_wealth_all[agent_type], axis=1)

            self._sharpe_ratio_all[agent_type] = np.mean(pnl_net, axis=1) / np.std(pnl_net, axis=1) * np.sqrt(252)

    def _compute_outputs_iter_j(self, j, factor_series_all_j, pnl_series_all_j, price_series_all_j, dates, agent_type):

        cost_j, risk_j, strategy_j, trades_j, value_j = self._initialize_output_list_for_agent()

        current_rescaled_shares = 0.

        factor_series_j = factor_series_all_j.loc[j, :]
        pnl_series_j = pnl_series_all_j.loc[j, :]
        price_series_j = price_series_all_j.loc[j, :]

        for date in dates[:-1]:
            factor, pnl, price = self._get_current_factor_pnl_price(date, dates, factor_series_j,
                                                                    pnl_series_j, price_series_j)

            cost_trade, current_rescaled_shares, rescaled_trade, risk_trade =\
                self._compute_outputs_for_time_t(agent_type, current_rescaled_shares, factor, price)

            self._update_lists(cost_j, cost_trade, current_rescaled_shares, pnl, rescaled_trade, risk_j,
                               risk_trade, strategy_j, trades_j, value_j)

        return strategy_j, trades_j, value_j, cost_j, risk_j


if __name__ == '__main__':

    simulationTester = SimulationTester('WTI')

    simulationTester.execute_simulation_testing(8, 5)

    simulationTester.make_plots(j_trajectories_plot=4)

    a = 1
