import os

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from benchmark_agents.agents import AgentGP
from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType, FactorType
from market_utils.market import instantiate_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import State, Action


# TODO: methods should be generalized, then specialized with a "trading" keyword in the name

class AgentTrainer:

    def __init__(self, riskDriverDynamicsType: RiskDriverDynamicsType, factorDynamicsType: FactorDynamicsType,
                 ticker: str, riskDriverType: RiskDriverType, shares_scale: float = 1,
                 factorType: FactorType = FactorType.Observable,
                 compare_to_GP: bool = True,
                 train_using_GP_reward: bool = True,
                 plot_regressor: bool = True,
                 large_regressor: bool = True):

        if train_using_GP_reward and not compare_to_GP:
            print('Warning! You set train_using_GP_reward = True but compare_to_GP = False. Forcing compare_to_GP = True.')
            compare_to_GP = True

        self._compare_to_GP = compare_to_GP
        self._train_using_GP_reward = train_using_GP_reward
        self._plot_regressor = plot_regressor
        self._large_regressor = large_regressor

        self.market = instantiate_market(riskDriverDynamicsType=riskDriverDynamicsType,
                                         factorDynamicsType=factorDynamicsType,
                                         ticker=ticker, riskDriverType=riskDriverType, factorType=factorType)
        self.shares_scale = shares_scale
        self.environment = Environment(market=self.market)
        self.agent = Agent(self.environment)

        self.market_benchmark = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear,
                                                   factorDynamicsType=FactorDynamicsType.AR,
                                                   ticker=ticker,
                                                   riskDriverType=RiskDriverType.PnL,
                                                   factorType=FactorType.Observable)
        self.agent_GP = AgentGP(market=self.market_benchmark)

    def train(self, j_episodes: int, n_batches: int, t_: int, eps_start: float = 0.1, parallel_computing: bool = False,
              n_cores: int = None):

        self.j_episodes = j_episodes
        self.n_batches = n_batches
        self.t_ = t_

        if parallel_computing and n_cores is None:
            print('Number of cores to use for parallel computing not set. Setting it to maximum available.')
            n_cores = os.cpu_count()

        if parallel_computing and n_cores > os.cpu_count():
            print('Number of cores set is greater than those available on this machine. Setting it to maximum available.')
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

            eps = eps/3

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

        print(f'Reward for batch {n+1}: {self.reward_RL[n]}')
        print(f'GP reward for batch {n+1}: {self.reward_GP[n]} \n')

    def _create_batch_sequential(self, eps, n):
        for j in tqdm(range(self.j_episodes), 'Creating episodes in batch %d of %d.' % (n + 1, self.n_batches)):
            state_action_grid, q_grid, reward_RL_j, reward_GP_j =\
                self._generate_single_episode(j, n, eps)
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
                                              chunksize=int(self.j_episodes/n_cores)),
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

            action_GP, reward_GP = self._get_reward_action_GP(j, n, state, t)

            if self._train_using_GP_reward:

                if reward_GP > reward_RL: # if reward_GP > reward_RL, choose GP action

                    # TODO: think about this: if the agent always chooses the action that is at least optimal (in GP sense)
                    #  then it is likely that it will never learn by making mistakes

                    action = action_GP
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

    def _get_reward_action_GP(self, j, n, state, t):

        if self._compare_to_GP:
            # compute action_GP for state
            rescaled_trade_GP = self.agent_GP.policy(current_factor=state.current_factor,
                                                     current_rescaled_shares=state.current_rescaled_shares,
                                                     shares_scale=self.shares_scale,
                                                     price=state.current_price)
            action_GP = Action()
            action_GP.set_trading_attributes(rescaled_trade=rescaled_trade_GP,
                                             shares_scale=self.shares_scale)
            # compute reward_GP for state and action_GP
            reward_GP, _ = self._get_reward_next_state_trading(state=state, action=action_GP, n=n, j=j, t=t)

        else:
            action_GP, reward_GP = None, np.nan

        return action_GP, reward_GP

    def _get_reward_next_state_trading(self, state: State, action: Action, n: int, j: int, t: int):

        reward, next_state = self.environment.compute_reward_and_next_state(state=state, action=action, n=n, j=j, t=t)

        return reward, next_state

    def _sarsa_updating_formula(self, next_state: State, next_action: Action, reward: float):

        q = reward + self.environment.gamma * self.agent.q_value(state=next_state, action=next_action)

        return q

    def _fit_supervised_regressor(self, n: int):

        print('    Fitting supervised regressor %d of %d.' % (n+1, self.n_batches))

        x_array, y_array = self._prepare_data_for_supervised_regressor_fit(n)

        model = self._set_and_fit_supervised_regressor_model(x_array, y_array, n)

        self.agent.update_q_value_models(q_value_model=model)

    def _set_and_fit_supervised_regressor_model(self, x_array, y_array, n):
        alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change, early_stopping, validation_fraction =\
            self._set_supervised_regressor_parameters()
        model = self._fit_supervised_regressor_model(alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change,
                                                     early_stopping, validation_fraction, x_array, y_array, n)
        return model

    def _fit_supervised_regressor_model(self, alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change, early_stopping,
                                        validation_fraction, x_array, y_array, n):
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             alpha=alpha_ann,
                             max_iter=max_iter,
                             n_iter_no_change=n_iter_no_change,
                             early_stopping=early_stopping,
                             validation_fraction=validation_fraction,
                             activation='relu',
                             verbose=1).fit(x_array, y_array)

        if self._plot_regressor:

            self._make_regressor_plots(model, n, x_array, y_array)

        return model

    def _make_regressor_plots(self, model, n, x_array, y_array):

        low_quant = 0.001
        high_quant = 0.999
        j_plot = np.random.randint(low=0, high=self.j_episodes, size=min(self.j_episodes, 10**5))
        x_plot = x_array[j_plot, :]

        # TODO: generalize to non-observable factors
        q_predicted = model.predict(x_plot)
        current_factor_array = x_plot[:, 0]
        current_rescaled_shares_array = x_plot[:, 1]
        rescaled_trade_array = x_plot[:, 2]

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)

        ax1 = plt.subplot2grid((2, 2), (0, 0))
        xlim = [np.quantile(y_array, low_quant), np.quantile(y_array, 0.95)]
        ylim = [np.quantile(q_predicted, low_quant), np.quantile(q_predicted, 0.95)]
        plt.scatter(y_array, q_predicted, s=1)
        plt.plot(xlim, ylim, label='45° line', color='r')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('Realized q')
        plt.ylabel('Predicted q')
        plt.legend()
        plt.title('Realized vs predicted q')

        ax2 = plt.subplot2grid((2, 2), (0, 1))
        plt.plot(current_factor_array, y_array, '.', markersize=5, alpha=0.5, color='b')
        plt.plot(current_factor_array, q_predicted, '.', markersize=5, alpha=0.5, color='r')
        xlim = [np.quantile(current_factor_array, low_quant), np.quantile(current_factor_array, high_quant)]
        ylim = [min(np.quantile(y_array, low_quant), np.quantile(q_predicted, low_quant)),
                max(np.quantile(y_array, high_quant), np.quantile(q_predicted, high_quant))]
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('Current factor')
        plt.ylabel('q')
        plt.title('Realized (blue) / predicted (red) q')

        ax3 = plt.subplot2grid((2, 2), (1, 0))
        plt.plot(current_rescaled_shares_array, y_array, '.', markersize=5, alpha=0.5, color='b')
        plt.plot(current_rescaled_shares_array, q_predicted, '.', markersize=5, alpha=0.5, color='r')
        xlim = [np.quantile(current_rescaled_shares_array, low_quant),
                np.quantile(current_rescaled_shares_array, high_quant)]
        ylim = [min(np.quantile(y_array, low_quant), np.quantile(q_predicted, low_quant)),
                max(np.quantile(y_array, high_quant), np.quantile(q_predicted, high_quant))]
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('Current rescaled shares')
        plt.ylabel('q')
        plt.title('Realized (blue) / predicted (red) q')

        ax4 = plt.subplot2grid((2, 2), (1, 1))
        plt.plot(rescaled_trade_array, y_array, '.', markersize=5, alpha=0.5, color='b')
        plt.plot(rescaled_trade_array, q_predicted, '.', markersize=5, alpha=0.5, color='r')
        xlim = [np.quantile(rescaled_trade_array, low_quant), np.quantile(current_rescaled_shares_array, high_quant)]
        ylim = [min(np.quantile(y_array, low_quant), np.quantile(q_predicted, low_quant)),
                max(np.quantile(y_array, high_quant), np.quantile(q_predicted, high_quant))]
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('Rescaled trade')
        plt.ylabel('q')
        plt.title('Realized (blue) / predicted (red) q')

        plt.tight_layout()

        filename = os.path.dirname(os.path.dirname(__file__)) + '/figures/training/training_batch_%d.png' % n
        plt.savefig(filename)

    def _set_supervised_regressor_parameters(self):

        if self._large_regressor:

            # hidden_layer_sizes = (64, 32, 8)
            hidden_layer_sizes = (100,)

        else:

            hidden_layer_sizes = (32,)

        # max_iter = 10
        max_iter = 200

        # n_iter_no_change = 5
        n_iter_no_change = 10

        alpha_ann = 0.0001

        # early_stopping = False
        early_stopping = True

        validation_fraction = 0.1

        return alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change, early_stopping, validation_fraction

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

    filename = os.path.dirname(os.path.dirname(__file__)) +\
               '/data/data_source/settings/settings.csv'
    df_trad_params = pd.read_csv(filename, index_col=0)

    shares_scale = float(df_trad_params.loc['shares_scale'][0])
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
