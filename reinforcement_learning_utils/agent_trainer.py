import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType, OptimizerType, SupervisedRegressorType,\
    InitialQvalueEstimateType, StrategyType
from gen_utils.utils import instantiate_polynomialFeatures, available_ann_architectures
from market_utils.market import instantiate_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import State, Action


# TODO: methods should be generalized, then specialized with a "trading" keyword in the name
from testing_utils.testers import BackTester, SimulationTester


class AgentTrainer:

    def __init__(self, riskDriverDynamicsType: RiskDriverDynamicsType, factorDynamicsType: FactorDynamicsType,
                 ticker: str,
                 riskDriverType: RiskDriverType, shares_scale: float = 1,
                 predict_pnl_for_reward: bool = False,
                 optimizerType: OptimizerType = OptimizerType.shgo,
                 average_across_models: bool = True,
                 use_best_n_batch: bool = False,
                 train_benchmarking_GP_reward: bool = False,
                 plot_regressor: bool = True,
                 supervisedRegressorType: SupervisedRegressorType = SupervisedRegressorType.ann,
                 initialQvalueEstimateType: InitialQvalueEstimateType = InitialQvalueEstimateType.zero,
                 max_ann_depth: int = 4,
                 early_stopping: bool = False,
                 max_iter: int = 10,
                 n_iter_no_change: int = 2,
                 activation: str = 'relu',
                 alpha_sarsa: float = 1.,
                 decrease_eps: bool = True,
                 random_initial_state: bool = True,
                 max_polynomial_regression_degree: int = 3,
                 max_complexity_no_gridsearch: bool = True,
                 alpha_ewma: float = 0.5,
                 use_best_n_batch_mode: str = 't_test_statistic',
                 restrict_evaluation_grid: bool = True):

        self.t_ = None
        self.n_batches = None
        self.j_episodes = None
        self.market = instantiate_market(riskDriverDynamicsType=riskDriverDynamicsType,
                                         factorDynamicsType=factorDynamicsType,
                                         ticker=ticker, riskDriverType=riskDriverType)
        self.shares_scale = shares_scale
        self._predict_pnl_for_reward = predict_pnl_for_reward
        self._optimizerType = optimizerType
        dump(self._optimizerType,
             os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/optimizerType.joblib')
        self._average_across_models = average_across_models
        dump(self._average_across_models,
             os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/average_across_models.joblib')
        self._use_best_n_batch = use_best_n_batch
        dump(self._use_best_n_batch,
             os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/use_best_n_batch.joblib')

        self.environment = Environment(market=self.market, random_initial_state=random_initial_state)
        self._add_absorbing_state = self.environment.add_absorbing_state

        if train_benchmarking_GP_reward and not self.environment.observe_GP:
            self.environment.observe_GP = True
            self.environment.instantiate_market_benchmark_and_agent_GP()

        self.state_factor = self.environment.state_factor
        self.observe_GP = self.environment.observe_GP
        self.state_GP_action = self.environment.state_GP_action
        self.state_ttm = self.environment.state_ttm
        self.state_pnl = self.environment.state_pnl
        self.train_benchmarking_GP_reward = train_benchmarking_GP_reward

        self._plot_regressor = plot_regressor
        self._supervisedRegressorType = supervisedRegressorType
        print(f'Fitting a {supervisedRegressorType.value} regressor')
        dump(self._supervisedRegressorType,
             os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/supervisedRegressorType.joblib')

        self._max_ann_depth = max_ann_depth
        self._early_stopping = early_stopping
        self._max_iter = max_iter
        self._n_iter_no_change = n_iter_no_change
        self._activation = activation
        self._alpha_sarsa = alpha_sarsa
        self._decrease_eps = decrease_eps
        self._random_initial_state = random_initial_state
        self._max_polynomial_regression_degree = max_polynomial_regression_degree
        self._max_complexity_no_gridsearch = max_complexity_no_gridsearch
        if self._max_complexity_no_gridsearch:
            self._polynomial_regression_degree = max_polynomial_regression_degree
        else:
            self._polynomial_regression_degree = None

        self._alpha_ewma = alpha_ewma
        dump(self._alpha_ewma,
             os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/alpha_ewma.joblib')

        self._available_use_best_n_batch_mode_lst = ['t_test_pvalue',
                                                     't_test_statistic',
                                                     'reward',
                                                     'average_q',
                                                     'model_convergence']
        if use_best_n_batch_mode not in self._available_use_best_n_batch_mode_lst:
            raise NameError(f'Invalid use_best_n_batch_mode: {use_best_n_batch_mode}. Should be in {self._available_use_best_n_batch_mode_lst}')
        self._use_best_n_batch_mode = use_best_n_batch_mode

        self._restrict_evaluation_grid = restrict_evaluation_grid

        self.agent = Agent(self.environment,
                           optimizerType=self._optimizerType,
                           average_across_models=self._average_across_models,
                           use_best_n_batch=self._use_best_n_batch,
                           initialQvalueEstimateType=initialQvalueEstimateType,
                           supervisedRegressorType=self._supervisedRegressorType,
                           alpha_ewma=self._alpha_ewma)

    def train(self, j_episodes: int, n_batches: int, t_: int, eps_start: float = 0.01, parallel_computing: bool = False,
              n_cores: int = None):

        self.j_episodes = j_episodes
        self.n_batches = n_batches
        self.t_ = t_

        if parallel_computing:
            if n_cores is None:
                print('Number of cores to use for parallel computing not set. Setting it to maximum available.')
                n_cores = os.cpu_count()
            elif n_cores > os.cpu_count():
                print('Number of cores set exceeds those available on this machine. Setting it to maximum available.')
                n_cores = os.cpu_count()
        else:
            n_cores = None

        self._n_cores = n_cores

        self._train_trading(eps_start, parallel_computing)

    def _train_trading(self, eps_start: float, parallel_computing: bool):

        self._generate_all_batches(eps_start, parallel_computing)

    def _generate_all_batches(self, eps_start: float, parallel_computing: bool):

        self._initialize_dicts_for_reporting()

        eps = eps_start

        for n in tqdm(range(self.n_batches), desc='Generating batches'):
            self._generate_batch(n=n, eps=eps, parallel_computing=parallel_computing)

            if self._decrease_eps:
                eps = max(eps / 3, 10 ** -5)

        # dump reports
        self._dump_training_report()

        # compute best batch
        self._compute_best_batch()

        print(f'Trained using N = {self.n_batches}, numbered (0, ..., {self.n_batches - 1}).')
        print(f'Best performance obtained by using estimate q_n for n = {self.best_n - 1}')

        print('Summaries:')
        for n in range(self.n_batches):
            print(f'Average RL reward for batch {n}: {self.reward_RL[n]}')

        for n in range(self.n_batches):
            print(f'|model_{n} - model_{n-1}|  / |model_{n-1}|: {self.model_convergence[n]}')

        for n in range(self.n_batches):
            print(f'Average RL backtesting Sharpe ratio for batch {n}: {self.backtesting_sharperatio[n]}')

        for n in range(self.n_batches):
            print(f'Average/Std RL simulation Sharpe ratio for batch {n}: {self.simulationtesting_sharperatio_av2std[n]}')

        for n in range(self.n_batches):
            print(f'Average/Std RL simulation wealth-net-risk for batch {n}: {self.simulationtesting_wealthnetrisk_av2std[n]}')

        for n in range(self.n_batches):
            stat = self.simulationtesting_ttest[n]['statistic']
            pval = self.simulationtesting_ttest[n]['pvalue']
            print(f'T-statistics for Welch\'s test WNRRL vs WRNGP for batch {n}: statistic={stat}, pvalue={pval}')

        if self._supervisedRegressorType == SupervisedRegressorType.polynomial_regression:
            self.agent.print_proportion_missing_polynomial_optima()

    def _compute_best_batch(self):

        if self._use_best_n_batch_mode in ('t_test_pvalue', 't_test_statistic'):
            self.best_n = self._get_best_n_based_on_tTest()

        elif self._use_best_n_batch_mode == 'reward':
            n_vs_reward_RL = np.array([[n, reward_RL] for n, reward_RL in self.reward_RL.items()])
            self.best_n = int(n_vs_reward_RL[np.argmax(n_vs_reward_RL[:, 1]), 0]) + 1

        elif self._use_best_n_batch_mode == 'average_q':
            average_q_per_batch = self._average_q_per_batch()
            self.best_n = max(np.argmax(average_q_per_batch), 1)

        elif self._use_best_n_batch_mode == 'model_convergence':
            # todo: should modify in such way that if model convergence is not improving more than tol, the batch
            #  iteration stops
            n_vs_model_convergence = np.array([[n, reward_RL]
                                               for n, reward_RL in self.model_convergence.items()
                                               if n > 0])
            self.best_n = int(n_vs_model_convergence[np.argmax(n_vs_model_convergence[:, 1]), 0]) + 1

        dump(self.best_n, os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/best_n.joblib')

    def _initialize_dicts_for_reporting(self):
        self.state_action_grid_dict = {}
        self.q_grid_dict = {}
        self.reward_RL = {}
        self.reward_GP = {}
        self._model_evaluations = {}
        self.model_convergence = {}
        self.backtesting_sharperatio = {}
        self.simulationtesting_sharperatio_av2std = {}
        self.simulationtesting_wealthnetrisk_av2std = {}
        self.simulationtesting_ttest = {}

    def _dump_training_report(self):
        df_reward_RL = pd.DataFrame.from_dict(data=self.reward_RL, orient='index', columns=['reward_RL'])
        df_reward_RL.index.name = 'batch'
        df_model_convergence = pd.DataFrame.from_dict(data=self.model_convergence, orient='index',
                                                      columns=['model_convergence'])
        df_model_convergence.index.name = 'batch'
        df_backtesting_sharperatio = pd.DataFrame.from_dict(data=self.backtesting_sharperatio, orient='index',
                                                            columns=['backtesting_sharperatio'])
        df_backtesting_sharperatio.index.name = 'batch'
        df_simulationtesting_sharperatio_av2std = pd.DataFrame.from_dict(data=self.simulationtesting_sharperatio_av2std,
                                                                         orient='index', columns=[
                'simulationtesting_sharperatio_av2std'])
        df_simulationtesting_sharperatio_av2std.index.name = 'batch'
        df_simulationtesting_wealthnetrisk_av2std = pd.DataFrame.from_dict(
            data=self.simulationtesting_wealthnetrisk_av2std, orient='index',
            columns=['simulationtesting_wealthnetrisk_av2std'])
        df_simulationtesting_wealthnetrisk_av2std.index.name = 'batch'
        df = pd.concat([df_reward_RL,
                        df_model_convergence,
                        df_backtesting_sharperatio,
                        df_simulationtesting_sharperatio_av2std,
                        df_simulationtesting_wealthnetrisk_av2std],
                       axis=1)
        filename = os.path.dirname(os.path.dirname(__file__)) + f'/reports/training/training_report.csv'
        df.to_csv(filename)

    def _get_best_n_based_on_tTest(self):

        marketDynamics = self.market.marketDynamics
        riskDriverDynamicsType = marketDynamics.riskDriverDynamics.riskDriverDynamicsType
        factorDynamicsType = marketDynamics.factorDynamics.factorDynamicsType
        strategyType = StrategyType.Unconstrained

        if self._use_best_n_batch_mode == 't_test_statistic':

            best_n = self._compute_best_n_based_on_statistic()

        elif self._use_best_n_batch_mode == 't_test_pvalue':

            best_n = self._compute_best_n_based_on_pvalue(factorDynamicsType, riskDriverDynamicsType,
                                                          strategyType)

        else:
            raise NameError(f'Invalid use_best_n_batch_mode: {self._use_best_n_batch_mode}')

        return best_n

    def _compute_best_n_based_on_statistic(self):
        n_vs_goodness = []
        print('Performing hypothesis testing: want statistic to be as large as possible')
        for n in range(len(self.simulationtesting_ttest)):
            n_vs_goodness.append([n, self.simulationtesting_ttest[n]['statistic']])
        n_vs_goodness = np.array(n_vs_goodness)
        best_n = int(n_vs_goodness[np.argmin(n_vs_goodness[:, 1]), 0]) + 1
        return best_n

    def _compute_best_n_based_on_pvalue(self, factorDynamicsType, riskDriverDynamicsType, strategyType):
        n_vs_goodness = []
        if (riskDriverDynamicsType == RiskDriverDynamicsType.Linear
                and factorDynamicsType == FactorDynamicsType.AR
                and strategyType == StrategyType.Unconstrained):
            print('Performing hypothesis testing for a benchmark case: want p-value to be as large as possible')
            # Benchmark
            # H0: RL = GP
            # H1: RL != GP
            # low p-value -> reject H0 -> conclude that RL != GP (RL does not recover benchmark) -> we want high p-value
            for n in range(len(self.simulationtesting_ttest)):
                n_vs_goodness.append([n, self.simulationtesting_ttest[n]['pvalue']])
            n_vs_goodness = np.array(n_vs_goodness)
            best_n = int(n_vs_goodness[np.argmax(n_vs_goodness[:, 1]), 0]) + 1
        else:
            print('Performing hypothesis testing for an alternative case: want p-value to be as small as possible')
            # Alternative
            # H0: RL <= GP
            # H1: RL > GP
            # low p-value -> reject H0 -> conclude that RL > GP (RL outperforms the benchmark)
            for n in range(len(self.simulationtesting_ttest)):
                n_vs_goodness.append([n, self.simulationtesting_ttest[n]['pvalue']])
            n_vs_goodness = np.array(n_vs_goodness)
            best_n = int(n_vs_goodness[np.argmin(n_vs_goodness[:, 1]), 0]) + 1
        return best_n

    def _average_q_per_batch(self):
        average_q_per_batch = []
        total_n_values = 0
        for key in self.q_grid_dict[0].keys():
            for _ in self.q_grid_dict[0][key]:
                total_n_values += 1

        for n in range(self.n_batches):
            q_grid_n = self.q_grid_dict[n]
            q_array_n = np.zeros(total_n_values)
            count = 0
            for j, episode in q_grid_n.items():
                for t in range(len(episode)):
                    q_array_n[count] = q_grid_n[j][t]
                    count += 1
            average_q_per_batch.append(q_array_n.mean())

        return average_q_per_batch

    def _generate_batch(self, n: int, eps: float, parallel_computing: bool):

        self._check_n(n)

        self.market.simulate_market_trading(n, self.j_episodes, self.t_ + 2)  # TODO: should go to dedicated method

        self.state_action_grid_dict[n] = {}
        self.q_grid_dict[n] = {}
        self.reward_RL[n] = 0.
        self.reward_GP[n] = 0.

        if parallel_computing:
            self._create_batch_parallel(eps, n)
        else:
            self._create_batch_sequential(eps, n)

        self._fit_supervised_regressor(n)

        # Execute on-the-fly backtesting and simulation-testing
        self._execute_otf_bkt_and_smt(n)

        print(f'Average RL reward for batch {n + 1}: {self.reward_RL[n]}')
        print(f'Average GP reward for batch {n + 1}: {self.reward_GP[n]}')
        print(f'|model_{n+1} - model_{n}|  / |model_{n}| : {self.model_convergence[n]}')
        print(f'Average RL backtesting Sharpe ratio for batch {n + 1}: {self.backtesting_sharperatio[n]}')
        print(f'Average/Std RL simulation Sharpe ratio for batch {n + 1}: {self.simulationtesting_sharperatio_av2std[n]}')
        print(f'Average/Std RL simulation wealth-net-risk for batch {n + 1}: {self.simulationtesting_wealthnetrisk_av2std[n]}')

    def _execute_otf_bkt_and_smt(self, n):
        backtester = BackTester(split_strategy=False, on_the_fly=True, n=n)
        backtester.execute_backtesting()
        backtester.make_plots()
        simulationTester = SimulationTester(on_the_fly=True, n=n)
        simulationTester.execute_simulation_testing(j_=10000, t_=self.t_)
        simulationTester.make_plots(j_trajectories_plot=5)
        self.backtesting_sharperatio[n] = backtester._sharpe_ratio_all['RL']
        self.simulationtesting_sharperatio_av2std[n] =\
            simulationTester._sharpe_ratio_all['RL'].mean() / simulationTester._sharpe_ratio_all['RL'].std()
        self.simulationtesting_wealthnetrisk_av2std[n] =\
            simulationTester._means['RL']['wealth_net_risk'] / simulationTester._stds['RL']['wealth_net_risk']
        self.simulationtesting_ttest[n] = {'statistic': simulationTester.tTester.t_test_result['statistic'],
                                           'pvalue': simulationTester.tTester.t_test_result['pvalue']}
        del backtester, simulationTester

    def _create_batch_sequential(self, eps, n):

        for j in tqdm(range(self.j_episodes), 'Creating episodes in batch %d of %d.' % (n + 1, self.n_batches)):
            state_action_grid, q_grid, reward_RL_j, reward_GP_j = self._generate_single_episode(j, n, eps)
            self._store_grids_in_dict(j, n, q_grid, state_action_grid)
            self.reward_RL[n] += reward_RL_j
            self.reward_GP[n] += reward_GP_j

        self._normalize_rewards(n)

    def _create_batch_parallel(self, eps, n):

        generate_single_episode = partial(self._generate_single_episode, n=n, eps=eps)

        p = mp.Pool(self._n_cores)

        episodes = p.map(func=generate_single_episode, iterable=range(self.j_episodes),
                         chunksize=int(self.j_episodes / self._n_cores))

        p.close()
        p.join()

        for j in range(len(episodes)):
            state_action_grid_j = episodes[j][0]
            q_grid_j = episodes[j][1]
            reward_RL_j = episodes[j][2]
            reward_GP_j = episodes[j][3]
            self._store_grids_in_dict(j, n, q_grid_j, state_action_grid_j)
            self.reward_RL[n] += reward_RL_j
            self.reward_GP[n] += reward_GP_j

        self._normalize_rewards(n)

    def _normalize_rewards(self, n):
        self.reward_RL[n] /= (self.j_episodes * self.t_)
        self.reward_GP[n] /= (self.j_episodes * self.t_)

    def _store_grids_in_dict(self, j, n, q_grid, state_action_grid):

        self.state_action_grid_dict[n][j] = state_action_grid
        self.q_grid_dict[n][j] = q_grid

    def _generate_single_episode(self, j: int, n: int, eps: float):

        self._check_j(j)
        reward_RL_j = 0.
        reward_GP_j = 0.

        # Initialize grid for supervised regressor interpolation
        state_action_grid = []
        q_grid = []

        # Observe state at t = 0
        state = self.environment.instantiate_initial_state_trading(n=n, j=j, shares_scale=self.shares_scale)

        # Choose action at t = 0
        action = self.agent.policy(state=state, eps=eps)

        for t in range(1, self.t_ + 2):

            if t == self.t_ + 1:
                if self._add_absorbing_state:
                    q = 0.
                    state_action_grid.append([state, action])
                    q_grid.append(q)
                    continue
                else:
                    continue

            # Observe reward_RL and state at time t
            reward_RL, next_state = self._get_reward_next_state_trading(state=state, action=action, n=n, j=j, t=t)
            reward_GP = self._get_reward_GP(j=j, n=n, state=state, action_GP=state.action_GP, t=t)

            if self.train_benchmarking_GP_reward:

                if reward_GP > reward_RL:  # if reward_GP > reward_RL, choose GP action

                    action = state.action_GP
                    reward_RL, next_state = \
                        self._get_reward_next_state_trading(state=state, action=action, n=n, j=j, t=t)

            reward_RL_j += reward_RL
            reward_GP_j += reward_GP

            # Choose action at time t
            next_action = self.agent.policy(state=next_state, eps=eps)

            # Observe next point on value function grid
            q = self._sarsa_updating_formula(state=state, action=action, next_state=next_state, next_action=next_action,
                                             reward=reward_RL)

            # Store point estimate
            state_action_grid.append([state, action])
            q_grid.append(q)

            # Update state and action
            state = next_state
            action = next_action

        return state_action_grid, q_grid, reward_RL_j, reward_GP_j

    def _get_reward_GP(self, j, n, state, action_GP, t):

        if self.observe_GP:
            reward_GP, _ = self._get_reward_next_state_trading(state=state, action=action_GP, n=n, j=j, t=t)

        else:
            reward_GP = 0.

        return reward_GP

    def _get_reward_next_state_trading(self, state: State, action: Action, n: int, j: int, t: int):

        next_state, reward = \
            self.environment.compute_reward_and_next_state(state=state, action=action, n=n, j=j, t=t,
                                                           predict_pnl_for_reward=self._predict_pnl_for_reward)

        return reward, next_state

    def _sarsa_updating_formula(self, state: State, action: Action, next_state: State, next_action: Action,
                                reward: float):

        q_prev = self.agent.q_value(state=state, action=action)

        q_new = self.agent.q_value(state=next_state, action=next_action)

        q = q_prev + self._alpha_sarsa * (reward + self.environment.gamma * q_new - q_prev)

        return q

    def _fit_supervised_regressor(self, n: int):

        print('    Fitting supervised regressor %d of %d.' % (n + 1, self.n_batches))

        x_array, y_array = self._prepare_data_for_supervised_regressor_fit(n)
        model = self._set_and_fit_supervised_regressor_model(x_array, y_array, n)

        self._dump_model_2_agent(model)
        self._store_model_evaluations(n)
        self._evaluate_model(n)

        del self.market.simulations_trading[n]

    def _evaluate_model(self, n):

        if n == 0:
            self.model_convergence[n] = 0.
        else:
            prev_model_evaluations = self._model_evaluations[n-1]
            curr_model_evaluations = self._model_evaluations[n]
            sup_norm_prev = np.linalg.norm(prev_model_evaluations, ord=np.inf)
            sup_norm_diff = np.linalg.norm(curr_model_evaluations - prev_model_evaluations, ord=np.inf)
            self.model_convergence[n] = sup_norm_diff / sup_norm_prev

    def _store_model_evaluations(self, n):

        model_evaluations = self.agent.qvl_from_ravel_input(np.column_stack(tuple([self._x_evaluation_meshgrid[j].ravel()
                                                                 for j in range(len(self._x_evaluation_meshgrid))])))
        self._model_evaluations[n] = model_evaluations

    def _dump_model_2_agent(self, model):
        self.agent.update_q_value_models(q_value_model=model)
        self.agent.dump_q_value_models()

    def _set_and_fit_supervised_regressor_model(self, x_array, y_array, n):
        alpha_ann, max_iter, n_iter_no_change, early_stopping, validation_fraction, activation = \
            self._set_supervised_regressor_parameters()
        model = self._fit_supervised_regressor_model(alpha_ann, max_iter, n_iter_no_change,
                                                     early_stopping, validation_fraction, activation, x_array, y_array,
                                                     n)
        return model

    def _fit_supervised_regressor_model(self, alpha_ann, max_iter, n_iter_no_change, early_stopping,
                                        validation_fraction, activation, x_array, y_array, n):

        if self._supervisedRegressorType == SupervisedRegressorType.ann:

            hidden_layer_sizes = self._perform_ann_grid_search(x_array, y_array, alpha_ann, max_iter, n_iter_no_change,
                                                               early_stopping, validation_fraction, activation)

            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                 alpha=alpha_ann,
                                 max_iter=max_iter,
                                 n_iter_no_change=n_iter_no_change,
                                 early_stopping=early_stopping,
                                 validation_fraction=validation_fraction,
                                 activation=activation,
                                 verbose=1).fit(x_array, y_array)

        elif self._supervisedRegressorType == SupervisedRegressorType.gradient_boosting:
            model = GradientBoostingRegressor(random_state=789,
                                              loss='squared_error',
                                              alpha=0.9,  # only used if loss='quantile'
                                              max_depth=int(np.log(len(y_array))),
                                              verbose=1).fit(x_array, y_array)

        elif self._supervisedRegressorType == SupervisedRegressorType.polynomial_regression:

            if n == 0:
                self._perform_polynomial_grid_search(x_array, y_array)

            poly = PolynomialFeatures(self._polynomial_regression_degree, interaction_only=False, include_bias=True)
            poly_features = poly.fit_transform(x_array)
            model = Ridge(alpha=self._ridge_alpha, fit_intercept=False).fit(poly_features, y_array)
            print(f'Score: {model.score(poly_features, y_array): .2f}')

        else:
            raise NameError(f'Invalid supervisedRegressorType: {self._supervisedRegressorType}')

        if self._plot_regressor:
            self._make_regressor_plots(model, n, x_array, y_array)

        return model

    def _perform_ann_grid_search(self, x_array, y_array, alpha_ann, max_iter, n_iter_no_change, early_stopping,
                                 validation_fraction, activation):
        ann_architectures = self._get_ann_architectures_by_depth()
        if self._max_complexity_no_gridsearch:

            return ann_architectures[self._max_ann_depth - 1]

        else:
            param_grid = {'ann__hidden_layer_sizes': ann_architectures}
            pipeline = Pipeline(steps=[('ann', MLPRegressor(alpha=alpha_ann,
                                                            max_iter=max_iter,
                                                            n_iter_no_change=n_iter_no_change,
                                                            early_stopping=early_stopping,
                                                            validation_fraction=validation_fraction,
                                                            activation=activation))])
            grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True,
                                       verbose=4).fit(x_array, y_array)
            print(f'Best parameters: {grid_search.best_params_}')

            return grid_search.best_params_['ann__hidden_layer_sizes']

    def _get_ann_architectures_by_depth(self):

        if len(available_ann_architectures) < self._max_ann_depth:
            print(f'You set max_ann_depth={self._max_ann_depth}',
                  f' but the maximum available is {len(available_ann_architectures)}')
            self._max_ann_depth = len(available_ann_architectures)

        return available_ann_architectures[:self._max_ann_depth]

    def _perform_polynomial_grid_search(self, x_array, y_array):

        if self._max_complexity_no_gridsearch:

            self._ridge_alpha = 0.
            self._polynomial_regression_degree = self._max_polynomial_regression_degree

        else:
            poly_degrees = list(np.arange(3, self._max_polynomial_regression_degree + 1))
            ridge_alphas = [0] + list(range(10, 101, 10))
            param_grid = [{'poly__degree': poly_degrees,
                           'ridge__alpha': ridge_alphas}]
            pipeline = Pipeline(steps=[('poly', PolynomialFeatures(interaction_only=False,
                                                                   include_bias=True)),
                                       ('ridge', Ridge(fit_intercept=False))])
            grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True,
                                       verbose=4).fit(x_array, y_array)
            print(f'Best joint parameters: {grid_search.best_params_}')

            best_poly_degree_1 = int(grid_search.best_params_['poly__degree'])
            best_ridge_1 = grid_search.best_params_['ridge__alpha']
            poly_degrees = np.arange(max(3, best_poly_degree_1 - 1),
                                     min(self._max_polynomial_regression_degree + 1, best_poly_degree_1 + 2))
            ridge_alphas = np.linspace(max(0, best_ridge_1 - 6), best_ridge_1 + 6, 25)
            param_grid = [{'poly__degree': poly_degrees,
                           'ridge__alpha': ridge_alphas}]
            pipeline = Pipeline(steps=[('poly', PolynomialFeatures(degree=self._polynomial_regression_degree,
                                                                   interaction_only=False,
                                                                   include_bias=True)),
                                       ('ridge', Ridge(fit_intercept=False))])
            grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True,
                                       verbose=4).fit(x_array, y_array)

            best_poly_degree_2 = int(grid_search.best_params_['poly__degree'])
            best_ridge_2 = grid_search.best_params_['ridge__alpha']
            poly_degrees = np.arange(max(3, best_poly_degree_2 - 1),
                                     min(self._max_polynomial_regression_degree + 1, best_poly_degree_2 + 2))
            ridge_alphas = np.linspace(max(0, best_ridge_2 - 3), best_ridge_2 + 3, 25)
            param_grid = [{'poly__degree': poly_degrees,
                           'ridge__alpha': ridge_alphas}]
            pipeline = Pipeline(steps=[('poly', PolynomialFeatures(degree=self._polynomial_regression_degree,
                                                                   interaction_only=False,
                                                                   include_bias=True)),
                                       ('ridge', Ridge(fit_intercept=False))])
            grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True,
                                       verbose=4).fit(x_array, y_array)

            print(f'Best joint parameters: {grid_search.best_params_}')

            self._ridge_alpha = grid_search.best_params_['ridge__alpha']
            self._polynomial_regression_degree = int(grid_search.best_params_['poly__degree'])

        self.agent.polynomial_regression_degree = self._polynomial_regression_degree
        dump(self._polynomial_regression_degree,
             os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/polynomial_regression_degree.joblib')

    def _make_regressor_plots(self, model, n, x_array, y_array):

        dpi = plt.rcParams['figure.dpi']

        inverse_state_shape = {}
        for key, value in self.environment.state_shape.items():
            inverse_state_shape[value] = key

        low_quant = 0.001
        high_quant = 0.999
        j_plot = np.random.randint(low=0,
                                   high=self.j_episodes * (self.t_ - 1),
                                   size=min(self.j_episodes * (self.t_ - 1), 10 ** 5))

        x_plot = x_array[j_plot, :]
        y_plot = y_array[j_plot]

        if self._supervisedRegressorType == SupervisedRegressorType.polynomial_regression:
            poly = instantiate_polynomialFeatures(self._polynomial_regression_degree)
            x_plot_poly = poly.fit_transform(x_plot)
            q_predicted = model.predict(x_plot_poly)
        else:
            q_predicted = model.predict(x_plot)

        # --------- Plotting y_plot vs q_predicted
        plt.figure(figsize=(1000 / dpi, 600 / dpi), dpi=dpi)
        xlim = [np.quantile(y_plot, low_quant), np.quantile(y_plot, 0.95)]
        ylim = [np.quantile(q_predicted, low_quant), np.quantile(q_predicted, 0.95)]
        plt.scatter(y_plot, q_predicted, s=1)
        plt.plot(xlim, ylim, label='45° line', color='r')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('Realized q')
        plt.ylabel('Predicted q')
        plt.legend()
        plt.title('Realized vs predicted q')
        filename = os.path.dirname(os.path.dirname(__file__)) +\
                   f'/figures/training/training_batch_{n}_realized_vs_predicted_q.png'
        plt.savefig(filename)

        # --------- Plotting detail of regressor
        for idx, variable in inverse_state_shape.items():
            plt.figure(figsize=(1000 / dpi, 600 / dpi), dpi=dpi)
            plt.plot(x_plot[:, idx], y_plot, '.', markersize=1, alpha=0.5, color='b')
            plt.plot(x_plot[:, idx], q_predicted, '.', markersize=1, alpha=0.5, color='r')
            xlim = [np.quantile(x_plot[:, idx], low_quant), np.quantile(x_plot[:, idx], high_quant)]
            ylim = [min(np.quantile(y_plot, low_quant), np.quantile(q_predicted, low_quant)),
                    max(np.quantile(y_plot, high_quant), np.quantile(q_predicted, high_quant))]
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel(variable)
            plt.ylabel('q')
            plt.title('Realized (blue) / predicted (red) q')
            filename = os.path.dirname(os.path.dirname(__file__)) + \
                       f'/figures/training/training_batch_{n}_{variable}.png'
            plt.savefig(filename)

        plt.figure(figsize=(1000 / dpi, 600 / dpi), dpi=dpi)
        plt.plot(x_plot[:, -1], y_plot, '.', markersize=1, alpha=0.5, color='b')
        plt.plot(x_plot[:, -1], q_predicted, '.', markersize=1, alpha=0.5, color='r')
        xlim = [np.quantile(x_plot[:, -1], low_quant), np.quantile(x_plot[:, -1], high_quant)]
        ylim = [min(np.quantile(y_plot, low_quant), np.quantile(q_predicted, low_quant)),
                max(np.quantile(y_plot, high_quant), np.quantile(q_predicted, high_quant))]
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('action')
        plt.ylabel('q')
        plt.title('Realized (blue) / predicted (red) q')
        filename = os.path.dirname(os.path.dirname(__file__)) + \
                   f'/figures/training/training_batch_{n}_action.png'
        plt.savefig(filename)

    def _set_supervised_regressor_parameters(self):

        max_iter = self._max_iter
        n_iter_no_change = self._n_iter_no_change
        alpha_ann = 0.0001
        early_stopping = self._early_stopping
        validation_fraction = 0.01
        activation = self._activation

        return alpha_ann, max_iter, n_iter_no_change, early_stopping, validation_fraction, activation

    def _prepare_data_for_supervised_regressor_fit(self, n):
        x_grid = []
        y_grid = []
        for j in self.state_action_grid_dict[n].keys():
            for t in range(len(self.state_action_grid_dict[n][j])):
                state = self.state_action_grid_dict[n][j][t][0]
                action = self.state_action_grid_dict[n][j][t][1]

                x = self.agent.extract_q_value_model_input_trading(state=state, action=action)
                q = self.q_grid_dict[n][j][t]

                x_grid.append(x)
                y_grid.append(q)

        x_array = np.array(x_grid).squeeze()
        y_array = np.array(y_grid).squeeze()

        if self._restrict_evaluation_grid:
            self._store_x_evaluation_grid(x_array)
        else:
            if n == 0:
                self._store_x_evaluation_grid(x_array)

        return x_array, y_array

    def _store_x_evaluation_grid(self, x_array):

        num_col = x_array.shape[1]
        x_evaluation_grid_mins = [np.quantile(x_array[:, j], 0.001) for j in range(num_col)]
        x_evaluation_grid_maxs = [np.quantile(x_array[:, j], 0.999) for j in range(num_col)]
        grids_lst = [np.linspace(x_evaluation_grid_mins[j], x_evaluation_grid_maxs[j]) for j in range(num_col)]
        self._x_evaluation_meshgrid = np.meshgrid(*grids_lst, indexing='ij')

    def _check_n(self, n: int):
        if n >= self.n_batches:
            raise NameError(f'Trying to extract simulations for batch n = {n + 1}, '
                            + f'but only {self.n_batches + 1} batches have been simulated.')

    def _check_j(self, j: int):
        if j >= self.j_episodes:
            raise NameError(f'Trying to simulate episode j = {j + 1}, '
                            + f'but only {self.j_episodes + 1} market paths have been simulated.')


