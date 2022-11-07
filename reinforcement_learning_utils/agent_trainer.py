import os

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from joblib import load, dump
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType, OptimizerType
from market_utils.market import instantiate_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import State, Action


# TODO: methods should be generalized, then specialized with a "trading" keyword in the name

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
                 ann_architecture: tuple = None,
                 early_stopping: bool = True,
                 max_iter: int = 200,
                 n_iter_no_change : int = 10,
                 activation: str = 'relu'):

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

        self.environment = Environment(market=self.market)
        self.agent = Agent(self.environment,
                           optimizerType=self._optimizerType,
                           average_across_models=self._average_across_models,
                           use_best_n_batch=self._use_best_n_batch)

        if train_benchmarking_GP_reward and not self.environment.observe_GP:
            self.environment.observe_GP = True
            self.environment.instantiate_market_benchmark_and_agent_GP()

        self.factor_in_state = self.environment.factor_in_state
        self.observe_GP = self.environment.observe_GP
        self.GP_action_in_state = self.environment.GP_action_in_state
        self.ttm_in_state = self.environment.ttm_in_state
        self.pnl_in_state = self.environment.pnl_in_state
        self.train_benchmarking_GP_reward = train_benchmarking_GP_reward

        self._plot_regressor = plot_regressor
        self._ann_architecture = ann_architecture
        self._early_stopping = early_stopping
        self._max_iter = max_iter
        self._n_iter_no_change = n_iter_no_change
        self._activation = activation

    def train(self, j_episodes: int, n_batches: int, t_: int, eps_start: float = 0.01, parallel_computing: bool = False,
              n_cores: int = None):

        self.j_episodes = j_episodes
        self.n_batches = n_batches
        self.t_ = t_

        if parallel_computing and n_cores is None:
            print('Number of cores to use for parallel computing not set. Setting it to maximum available.')
            n_cores = os.cpu_count()

        if parallel_computing and n_cores > os.cpu_count():
            print('Number of cores set exceeds those available on this machine. Setting it to maximum available.')
            n_cores = os.cpu_count()

        self._train_trading(eps_start, parallel_computing, n_cores)

    def _train_trading(self, eps_start: float, parallel_computing: bool, n_cores: int):

        self._generate_all_batches(eps_start, parallel_computing, n_cores)

    def _generate_all_batches(self, eps_start: float, parallel_computing: bool, n_cores: int):

        self.state_action_grid_dict = {}
        self.q_grid_dict = {}
        self.reward_RL = {}
        self.reward_GP = {}

        eps = eps_start

        for n in range(self.n_batches):
            self._generate_batch(n=n, eps=eps, parallel_computing=parallel_computing, n_cores=n_cores)

            eps = max(eps/3, 10**-5)

        n_vs_reward_RL = np.array([[n, reward_RL] for n, reward_RL in self.reward_RL.items()])
        self.best_n = int(n_vs_reward_RL[np.argmax(n_vs_reward_RL[:, 1]), 0])

        dump(self.best_n + 1, os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/best_n.joblib')
        print(f'Trained using N = {self.n_batches}; best reward obtained on batch n = {self.best_n + 1}')

    def _generate_batch(self, n: int, eps: float, parallel_computing: bool, n_cores: int):

        self._check_n(n)

        self.market.simulate_market_trading(n, self.j_episodes, self.t_)  # TODO: should go to dedicated method

        self.state_action_grid_dict[n] = {}
        self.q_grid_dict[n] = {}
        self.reward_RL[n] = 0.
        self.reward_GP[n] = 0.

        if parallel_computing:

            self._create_batch_parallel(eps, n, n_cores)

        else:

            self._create_batch_sequential(eps, n)

        self._fit_supervised_regressor(n)  # TODO: should go to dedicated method

        del self.market.simulations_trading[n]

        self.reward_RL[n] /= (self.j_episodes * self.t_)
        self.reward_GP[n] /= (self.j_episodes * self.t_)

        print(f'Average RL reward for batch {n + 1}: {self.reward_RL[n]}')
        print(f'Average GP reward for batch {n + 1}: {self.reward_GP[n]} \n')

    def _create_batch_sequential(self, eps, n):

        for j in tqdm(range(self.j_episodes), 'Creating episodes in batch %d of %d.' % (n + 1, self.n_batches)):
            state_action_grid, q_grid, reward_RL_j, reward_GP_j = self._generate_single_episode(j, n, eps)
            self._store_grids_in_dict(j, n, q_grid, state_action_grid)
            self.reward_RL[n] += reward_RL_j
            self.reward_GP[n] += reward_GP_j

    def _create_batch_parallel(self, eps, n, n_cores):

        print('Creating batch %d of %d.' % (n + 1, self.n_batches))
        generate_single_episode = partial(self._generate_single_episode, n=n, eps=eps)

        p = mp.Pool(n_cores)

        # TODO: define once and for all which of these three approaches is fastest
        # 1:
        episodes = list(tqdm(p.imap_unordered(func=generate_single_episode,
                                              iterable=range(self.j_episodes),
                                              chunksize=int(self.j_episodes / n_cores)),
                             total=self.j_episodes))

        # 2:
        # episodes = list(p.imap_unordered(func=generate_single_episode,
        #                                  iterable=range(self.j_episodes),
        #                                  chunksize=int(self.j_episodes/n_cores)))

        # 3:
        # episodes = p.map(generate_single_episode, range(self.j_episodes))

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

        for t in range(1, self.t_):

            # Observe reward_RL and state at time t
            reward_RL, next_state = self._get_reward_next_state_trading(state=state, action=action, n=n, j=j, t=t)
            reward_GP = self._get_reward_GP(j=j, n=n, state=state, action_GP=state.action_GP, t=t)

            if self.train_benchmarking_GP_reward:

                if reward_GP > reward_RL:  # if reward_GP > reward_RL, choose GP action

                    action = state.action_GP
                    reward_RL, next_state = self._get_reward_next_state_trading(state=state, action=action, n=n, j=j,
                                                                                t=t)

            reward_RL_j += reward_RL
            reward_GP_j += reward_GP

            # Choose action at time t
            next_action = self.agent.policy(state=next_state, eps=eps)

            # Observe next point on value function grid
            q = self._sarsa_updating_formula(next_state=next_state, next_action=next_action, reward=reward_RL)

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

        next_state, reward = self.environment.compute_reward_and_next_state(state=state, action=action, n=n, j=j, t=t,
                                                                            predict_pnl_for_reward=self._predict_pnl_for_reward)

        return reward, next_state

    def _sarsa_updating_formula(self, next_state: State, next_action: Action, reward: float):

        q = reward + self.environment.gamma * self.agent.q_value(state=next_state, action=next_action)

        return q

    def _fit_supervised_regressor(self, n: int):

        print('    Fitting supervised regressor %d of %d.' % (n + 1, self.n_batches))

        x_array, y_array = self._prepare_data_for_supervised_regressor_fit(n)

        model = self._set_and_fit_supervised_regressor_model(x_array, y_array, n)

        self.agent.update_q_value_models(q_value_model=model)

    def _set_and_fit_supervised_regressor_model(self, x_array, y_array, n):
        alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change, early_stopping, validation_fraction, activation = \
            self._set_supervised_regressor_parameters()
        model = self._fit_supervised_regressor_model(alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change,
                                                     early_stopping, validation_fraction, activation, x_array, y_array,
                                                     n)
        return model

    def _fit_supervised_regressor_model(self, alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change, early_stopping,
                                        validation_fraction, activation, x_array, y_array, n):
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             alpha=alpha_ann,
                             max_iter=max_iter,
                             n_iter_no_change=n_iter_no_change,
                             early_stopping=early_stopping,
                             validation_fraction=validation_fraction,
                             activation=activation,
                             verbose=1).fit(x_array, y_array)

        if self._plot_regressor:
            self._make_regressor_plots(model, n, x_array, y_array)

        return model

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
                   '/figures/training/training_batch_%d_realized_vs_predicted_q.png' % n
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
            filename = os.path.dirname(os.path.dirname(__file__)) +\
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
        filename = os.path.dirname(os.path.dirname(__file__)) +\
                   f'/figures/training/training_batch_{n}_action.png'
        plt.savefig(filename)

    def _set_supervised_regressor_parameters(self):

        if self._ann_architecture is None:
            self._ann_architecture = (int(5 * 10 ** (-4) * self.j_episodes * self.t_),)

        hidden_layer_sizes = self._ann_architecture
        max_iter = self._max_iter
        n_iter_no_change = self._n_iter_no_change
        alpha_ann = 0.0001
        early_stopping = self._early_stopping
        validation_fraction = 0.01
        activation = self._activation

        return alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change, early_stopping, validation_fraction, activation

    def _prepare_data_for_supervised_regressor_fit(self, n):
        x_grid = []
        y_grid = []
        for j in range(self.j_episodes):
            for t in range(self.t_ - 1):
                state = self.state_action_grid_dict[n][j][t][0]
                action = self.state_action_grid_dict[n][j][t][1]

                x = self.agent.extract_q_value_model_input_trading(state=state, action=action)
                q = self.q_grid_dict[n][j][t]

                x_grid.append(x)
                y_grid.append(q)

        x_array = np.array(x_grid).squeeze()
        y_array = np.array(y_grid).squeeze()
        return x_array, y_array

    def _check_n(self, n: int):
        if n >= self.n_batches:
            raise NameError('Trying to extract simulations for batch n = %d, '
                            + 'but only %d batches have been simulated.' % (n + 1, self.n_batches + 1))

    def _check_j(self, j: int):
        if j >= self.j_episodes:
            raise NameError('Trying to simulate episode j = %d, '
                            + 'but only %d market paths have been simulated.' % (j + 1, self.j_episodes + 1))


def read_trading_parameters_training(ticker):
    filename = os.path.dirname(os.path.dirname(__file__)) + \
               '/data/data_source/settings/settings.csv'
    df_trad_params = pd.read_csv(filename, index_col=0)

    shares_scale = float(load(os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/shares_scale.joblib'))

    j_episodes = int(df_trad_params.loc['j_episodes'][0])
    n_batches = int(df_trad_params.loc['n_batches'][0])
    t_ = int(df_trad_params.loc['t_'][0])

    if df_trad_params.loc['parallel_computing'][0] == 'Yes':
        parallel_computing = True
        n_cores = int(df_trad_params.loc['n_cores'][0])
        n_cores = min(n_cores, mp.cpu_count())
    elif df_trad_params.loc['parallel_computing'][0] == 'No':
        parallel_computing = False
        n_cores = None
    else:
        raise NameError('Invalid value for parameter parallel_computing in ' + ticker + '_trading_parameters.csv')

    return shares_scale, j_episodes, n_batches, t_, parallel_computing, n_cores
