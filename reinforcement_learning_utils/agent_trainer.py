from enums import RiskDriverDynamicsType, FactorDynamicsType, RiskDriverType, FactorType
from market_utils.market import instantiate_market
from reinforcement_learning_utils.agent import Agent
from reinforcement_learning_utils.environment import Environment
from reinforcement_learning_utils.state_action_utils import State, Action

# methods should be generalized, then specialized with a "trading" keyword in the name. E.g.
# def _genera

class AgentTrainer:

    def __init__(self, riskDriverDynamicsType: RiskDriverDynamicsType, factorDynamicsType: FactorDynamicsType,
                 ticker: str, riskDriverType: RiskDriverType, factorType: FactorType = FactorType.Observable):

        self.market = instantiate_market(riskDriverDynamicsType=riskDriverDynamicsType,
                                         factorDynamicsType=factorDynamicsType,
                                         ticker=ticker, riskDriverType=riskDriverType, factorType=factorType)
        self.environment = Environment(market=self.market)
        self.agent = Agent(self.environment)

    def simulate_market_training(self, j_episodes: int, n_batches: int, t_: int):

        pnl, factor = self.market.simulate_market_for_batches(j_episodes=j_episodes, n_batches=n_batches, t_=t_)

        self.j_episodes = j_episodes
        self.n_batches = n_batches
        self.t_ = t_
        self.pnl = pnl
        self.factor = factor

    def train(self):

        self._train_trading()

    def _train_trading(self):
        pass

    def _generate_all_batches(self):

        self.state_action_grid_dict = {}
        self.q_grid_dict = {}

        for n in range(self.n_batches):

            self._generate_batch(n=n)

    def _generate_batch(self, n: int):

        self.check_n(n)

        for j in range(self.j_episodes):

            self._generate_single_episode(n=n, j=j)

    def _generate_single_episode(self, n: int, j: int):

        self._check_n(n)
        self._check_j(j)

        # Get market simulation
        pnl, factor = self._get_market_simulation_trading(n=n, j=j)

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
            reward, next_state = self.environment.compute_reward_and_next_state(state=state, action=action)

            # Choose action at time t
            next_action = self.agent.policy(state=next_state)

            # Observe next point on value function grid
            q = self._sarsa_updating_formula(state=state, action=action, next_state=next_state, next_action=next_action,
                                             reward=reward)

            # Store point estimate
            state_action_grid.append([state, action])
            q_grid.append(q)

            # Update state and action
            state = next_state
            action = next_action

        # Store grid for supervised regressor interpolation
        self.state_action_grid_dict[n][j] = state_action_grid
        self.q_grid_dict[n][j] = q_grid

    def _sarsa_updating_formula(self, state: State, action: Action, next_state: State, next_action: Action,
                                reward: float):
        q = 0.

        return q

    def _get_market_simulation_trading(self, n: int, j: int):

        return self.pnl[:, n, j], self.factor[:, n, j]

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

    riskDriverDynamicsType = RiskDriverDynamicsType.Linear
    factorDynamicsType = FactorDynamicsType.AR
    ticker = 'WTI'
    riskDriverType = RiskDriverType.PnL
    factorType = FactorType.Observable
    j_episodes = 15000
    n_batches = 3
    t_ = 200

    agentTrainer = AgentTrainer(riskDriverDynamicsType=riskDriverDynamicsType,
                                factorDynamicsType=factorDynamicsType,
                                ticker=ticker,
                                riskDriverType=riskDriverType,
                                factorType=factorType)

    agentTrainer.simulate_market_training(j_episodes=j_episodes, n_batches=n_batches, t_=t_)

    a = 1
