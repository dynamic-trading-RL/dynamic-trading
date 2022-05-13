import numpy as np
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType, FactorType
from market_utils.market import instantiate_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import State, Action

# methods should be generalized, then specialized with a "trading" keyword in the name. E.g.
# def _genera

class AgentTrainer:

    def __init__(self, riskDriverDynamicsType: RiskDriverDynamicsType, factorDynamicsType: FactorDynamicsType,
                 ticker: str, riskDriverType: RiskDriverType, shares_scale: float = 1,
                 factorType: FactorType = FactorType.Observable):

        self.market = instantiate_market(riskDriverDynamicsType=riskDriverDynamicsType,
                                         factorDynamicsType=factorDynamicsType,
                                         ticker=ticker, riskDriverType=riskDriverType, factorType=factorType)
        self.shares_scale = shares_scale
        self.environment = Environment(market=self.market)
        self.agent = Agent(self.environment)

    def train(self, j_episodes: int, n_batches: int, t_: int, eps_start: float = 0.1):

        self.j_episodes = j_episodes
        self.n_batches = n_batches
        self.t_ = t_

        self._train_trading(eps_start)

    def _train_trading(self, eps_start: float):

        self.market.simulate_market_trading(self.n_batches, self.j_episodes, self.t_)
        self._generate_all_batches(eps_start)

    def _generate_all_batches(self, eps_start: float):

        self.state_action_grid_dict = {}
        self.q_grid_dict = {}

        eps = eps_start

        for n in range(self.n_batches):

            self._generate_batch(n=n, eps=eps)

            eps = eps/3

    def _generate_batch(self, n: int, eps: float):

        self._check_n(n)
        self.state_action_grid_dict[n] = {}
        self.q_grid_dict[n] = {}

        for j in tqdm(range(self.j_episodes), 'Creating episodes in batch %d of %d.' % (n+1, self.n_batches)):

            self._generate_single_episode(n=n, j=j, eps=eps)

        self._fit_supervised_regressor(n)

    def _generate_single_episode(self, n: int, j: int, eps: float):

        self._check_j(j)

        # Initialize grid for supervised regressor interpolation
        state_action_grid = []
        q_grid = []

        # Observe state at t = 0
        state = self.environment.instantiate_initial_state_trading(n=n, j=j, shares_scale=self.shares_scale)

        # Choose action at t = 0
        action = self.agent.policy(state=state, eps=eps)

        for t in range(1, self.t_):

            # Observe reward and state at time t
            reward, next_state = self._get_reward_next_state_trading(state=state, action=action, n=n, j=j, t=t)

            # Choose action at time t
            next_action = self.agent.policy(state=next_state, eps=eps)

            # Observe next point on value function grid
            q = self._sarsa_updating_formula(next_state=next_state, next_action=next_action, reward=reward)

            # Store point estimate
            state_action_grid.append([state, action])
            q_grid.append(q)

            # Update state and action
            state = next_state
            action = next_action

        # Store grid for supervised regressor interpolation
        self.state_action_grid_dict[n][j] = state_action_grid
        self.q_grid_dict[n][j] = q_grid

    def _get_reward_next_state_trading(self, state: State, action: Action, n: int, j: int, t: int):

        reward, next_state = self.environment.compute_reward_and_next_state(state=state, action=action, n=n, j=j, t=t)

        return reward, next_state

    def _sarsa_updating_formula(self, next_state: State, next_action: Action, reward: float):

        q = reward + self.environment.gamma * self.agent.q_value(state=next_state, action=next_action)

        return q

    def _fit_supervised_regressor(self, n: int):

        x_array, y_array = self._prepare_data_for_supervised_regressor_fit(n)

        model = self._set_and_fit_supervised_regressor_model(x_array, y_array)

        self.agent.update_q_value_models(q_value_model=model)

    def _set_and_fit_supervised_regressor_model(self, x_array, y_array):
        alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change = self._set_supervised_regressor_parameters()
        model = self._fit_supervised_regressor_model(alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change, x_array,
                                                     y_array)
        return model

    def _fit_supervised_regressor_model(self, alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change, x_array,
                                        y_array):
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             alpha=alpha_ann,
                             max_iter=max_iter,
                             n_iter_no_change=n_iter_no_change,
                             activation='relu').fit(x_array, y_array)
        return model

    def _set_supervised_regressor_parameters(self):
        hidden_layer_sizes = (64, 32, 8)
        max_iter = 10
        n_iter_no_change = 2
        alpha_ann = 0.0001
        return alpha_ann, hidden_layer_sizes, max_iter, n_iter_no_change

    def _prepare_data_for_supervised_regressor_fit(self, n):
        x_grid = []
        y_grid = []
        for j in range(self.j_episodes):
            for t in range(self.t_ - 1):

                state = self.state_action_grid_dict[n][j][t][0]
                action = self.state_action_grid_dict[n][j][t][1]
                q = self.q_grid_dict[n][j][t]

                x_grid.append(self.agent._extract_q_value_model_input_trading(state=state, action=action))
                y_grid.append(q)

        x_array = np.array(x_grid).squeeze()
        y_array = np.array(y_grid)
        return x_array, y_array

    def _check_n(self, n: int):
        if n >= self.n_batches:
            raise NameError('Trying to extract simulations for batch n = %d, '
                            + 'but only %d batches have been simulated.' % (n + 1, self.n_batches + 1))

    def _check_j(self, j: int):
        if j >= self.j_episodes:
            raise NameError('Trying to simulate episode j = %d, '
                            + 'but only %d market paths have been simulated.' % (j + 1, self.j_episodes + 1))


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    riskDriverDynamicsType = RiskDriverDynamicsType.Linear
    factorDynamicsType = FactorDynamicsType.AR
    ticker = 'WTI'
    riskDriverType = RiskDriverType.PnL
    factorType = FactorType.Observable
    j_episodes = 20
    n_batches = 5
    t_ = 50

    agentTrainer = AgentTrainer(riskDriverDynamicsType=riskDriverDynamicsType,
                                factorDynamicsType=factorDynamicsType,
                                ticker=ticker,
                                riskDriverType=riskDriverType,
                                factorType=factorType)

    agentTrainer.train(j_episodes=j_episodes, n_batches=n_batches, t_=t_)

    agentTrainer.agent.dump_q_value_models()

