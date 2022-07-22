import pandas as pd
import os

from benchmark_agents.agents import AgentGP
from enums import RiskDriverDynamicsType , FactorDynamicsType , RiskDriverType , FactorType
from reinforcement_learning_utils.state_action_utils import Action , State
from market_utils.market import Market , instantiate_market


class Environment :

    def __init__(self , market: Market) :

        self.market = market
        self._set_attributes()

    def compute_reward_and_next_state(self , state: State , action: Action , n: int , j: int , t: int) :

        reward = self._compute_reward(state=state , action=action)
        next_state = self._compute_next_state(state=state , action=action , n=n , j=j , t=t)

        return reward , next_state

    def instantiate_initial_state_trading(self , n: int , j: int , shares_scale: float = 1) :

        current_rescaled_shares = 0.
        current_other_observable = 0.

        pnl , factor , price = self._get_market_simulation_trading(n=n , j=j , t=0)

        if self.observe_GP :
            rescaled_trade_GP = self.agent_GP.policy(current_factor=factor ,
                                                     current_rescaled_shares=current_rescaled_shares ,
                                                     shares_scale=shares_scale ,
                                                     price=price)
            action_GP = Action()
            action_GP.set_trading_attributes(rescaled_trade=rescaled_trade_GP ,
                                             shares_scale=shares_scale)
        else :
            action_GP = None

        state = State()
        state.set_trading_attributes(current_factor=factor ,
                                     current_rescaled_shares=current_rescaled_shares ,
                                     current_other_observable=current_other_observable ,
                                     shares_scale=shares_scale ,
                                     current_price=price ,
                                     action_GP=action_GP)

        return state

    def _compute_reward(self , state: State , action: Action) :

        reward = self._compute_trading_reward(state , action)

        return reward

    def _compute_next_state(self , state: State , action: Action , n: int , j: int , t: int) :

        next_state = self._compute_trading_next_state(state , action , n , j , t)

        return next_state

    def _compute_trading_reward(self , state , action) :

        current_shares = state.current_shares
        current_factor = state.current_factor
        current_price = state.current_price

        pnl = self.market.next_step_pnl(factor=current_factor , price=current_price)
        sig2 = self.market.next_step_sig2(factor=current_factor , price=current_price)

        cost = self.compute_trading_cost(action , sig2)

        reward = self.gamma * (current_shares * pnl - 0.5 * self.kappa * current_shares * sig2 * current_shares) - cost

        return reward

    def _compute_trading_next_state(self , state: State , action: Action , n: int , j: int , t: int) :

        current_rescaled_shares = state.current_rescaled_shares
        shares_scale = state.shares_scale

        next_rescaled_shares = current_rescaled_shares + action.rescaled_trade

        _ , factor , price = self._get_market_simulation_trading(n=n , j=j , t=t)
        next_factor = factor
        next_other_observable = 0.
        next_price = price

        if self.observe_GP :
            next_rescaled_shares_GP = self.agent_GP.policy(current_factor=factor ,
                                                           current_rescaled_shares=current_rescaled_shares ,
                                                           shares_scale=shares_scale ,
                                                           price=price)
            next_action_GP = Action()
            next_action_GP.set_trading_attributes(rescaled_trade=next_rescaled_shares_GP ,
                                                  shares_scale=shares_scale)
        else :
            next_action_GP = None

        next_state = State()
        next_state.set_trading_attributes(current_factor=next_factor ,
                                          current_rescaled_shares=next_rescaled_shares ,
                                          current_other_observable=next_other_observable ,
                                          shares_scale=shares_scale ,
                                          current_price=next_price ,
                                          action_GP=next_action_GP)

        return next_state

    def compute_trading_cost(self , action , sig2) :

        trade = action.trade

        return 0.5 * trade * self.lam * sig2 * trade

    def compute_trading_risk(self , state , sig2) :

        current_shares = state.current_shares

        return 0.5 * current_shares * self.kappa * sig2 * current_shares

    def _get_market_simulation_trading(self , n: int , j: int , t: int) :

        return (self.market.simulations_trading[n]['pnl'][j , t] ,
                self.market.simulations_trading[n]['factor'][j , t] ,
                self.market.simulations_trading[n]['price'][j , t])

    def _set_attributes(self) :

        self._set_trading_attributes()

    def _set_trading_attributes(self) :

        self.ticker = self.market.ticker
        filename = os.path.dirname(os.path.dirname(__file__)) + \
                   '/data/data_source/settings/settings.csv'
        df_trad_params = pd.read_csv(filename , index_col=0)

        lam = float(df_trad_params.loc['lam'][0])
        self.lam = lam

        if str(df_trad_params.loc['GP_action_in_state'][0]) == 'Yes' :
            GP_action_in_state = True
            observe_GP = True
        elif str(df_trad_params.loc['GP_action_in_state'][0]) == 'No' :
            GP_action_in_state = False
            observe_GP = False
        else :
            raise NameError('GP_action_in_state in settings file must be either Yes or No')
        self.GP_action_in_state = GP_action_in_state
        self.observe_GP = observe_GP

        if self.observe_GP :
            self.instantiate_market_benchmark_and_agent_GP()

        self.factorType = self.market.factorType

    def instantiate_market_benchmark_and_agent_GP(self):
        self.market_benchmark = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear ,
                                                   factorDynamicsType=FactorDynamicsType.AR ,
                                                   ticker=self.ticker ,
                                                   riskDriverType=RiskDriverType.PnL ,
                                                   factorType=FactorType.Observable)
        self.agent_GP = AgentGP(market=self.market_benchmark)

    def _get_trading_parameters_from_agent(self , gamma: float , kappa: float) :

        self.gamma = gamma
        self.kappa = kappa
