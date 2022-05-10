from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType
from market_utils.market import instantiate_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import State, Action


class AgentTrainer:

    def __init__(self, riskDriverDynamicsType: RiskDriverDynamicsType, factorDynamicsType: FactorDynamicsType,
                 ticker: str, riskDriverType: RiskDriverType):

        self.agent = Agent()
        self.market = instantiate_market(riskDriverDynamicsType=riskDriverDynamicsType,
                                         factorDynamicsType=factorDynamicsType,
                                         ticker=ticker, riskDriverType=riskDriverType)
        self.environment = Environment(market=self.market)

    def simulate_market_for_training(self, j_episodes: int, n_batches: int, t_: int):

        pnl, factor = self.market.simulate_market_for_batches(j_episodes=j_episodes, n_batches=n_batches, t_=t_)

        self.j_episodes = j_episodes
        self.n_batches = n_batches
        self.t_ = t_
        self.pnl = pnl
        self.factor = factor

    def _generate_all_batches(self):

        self.state_action_grid_dict = {}
        self.q_grid_dict = {}

        for n in range(self.n_batches):

            self._generate_batch(n=n)

    def _generate_batch(self, n: int):

        self.check_n(n)

        for j in range(self.j_episodes):

            self._generate_single_episode(n=n, j=j)

    def _get_market_simulation(self, n: int, j: int):

        return self.pnl[:, n, j], self.factor[:, n, j]

    def _generate_single_episode(self, n: int, j: int):

        self._check_n(n)
        self._check_j(j)

        # Get market simulation
        pnl, factor = self._get_market_simulation(n=n, j=j)

        # Initialize grid for supervised regressor interpolation
        self.state_action_grid_dict[n] = {}
        self.q_grid_dict[n] = {}
        state_action_grid = []
        q_grid = []

        # Observe state at t = 0
        state = State()

        # Choose action at t = 0
        action = self.agent.policy(state)

        for t in range(self.t_):

            # Observe reward and state at time t
            reward, new_state = self.environment.compute_reward_and_new_state(state=state, action=action)

            # Choose action at time t
            new_action = self.agent.policy(state=new_state)

            # Observe new point on value function grid
            q = self._sarsa_updating_formula(state=state, action=action, new_state=new_state, new_action=new_action,
                                             reward=reward)

            # Store point estimate
            state_action_grid.append([state, action])
            q_grid.append(q)

            # Update state and action
            state = new_state
            action = new_action

        # Store grid for supervised regressor interpolation
        self.state_action_grid_dict[n][j] = state_action_grid
        self.q_grid_dict[n][j] = q_grid

    def _sarsa_updating_formula(self, state: State, action: Action, new_state: State, new_action: Action,
                                reward: float):
        q = 0.

        return q

    def _check_n(self, n: int):
        if n >= self.n_batches:
            raise NameError('Trying to extract simulations for batch n = %d, '
                            + 'but only %d batches have been simulated.' % (n + 1, self.n_batches + 1))

    def _check_j(self, j: int):
        if j >= self.j_episodes:
            raise NameError('Trying to simulate episode j = %d, '
                            + 'but only %d market paths have been simulated.' % (j + 1, self.j_episodes + 1))
