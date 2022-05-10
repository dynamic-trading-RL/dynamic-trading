from reinforcement_learning_utils.state_action_utils import Action, State
from market_utils.market import Market


class Environment:

    def __init__(self, market: Market):

        self.market = market

    def compute_reward_and_new_state(self, state: State, action: Action):

        reward = self._compute_reward(state=state, action=action)
        new_state = self._compute_new_state(state=state, action=action)

        return reward, new_state

    def _compute_reward(self, state: State, action: Action):

        reward = 0.

        return reward

    def _compute_new_state(self, state: State, action: Action):

        new_state = State()

        return new_state
