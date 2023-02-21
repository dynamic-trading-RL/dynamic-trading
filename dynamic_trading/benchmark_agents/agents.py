import os
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dynamic_trading.enums.enums import RiskDriverDynamicsType, RiskDriverType, FactorDynamicsType, StrategyType
from dynamic_trading.market_utils.market import Market, instantiate_market


class AgentBenchmark:
    """
    Base class for benchmark agent.

    Attributes
    ----------
    gamma : float
        The cumulative reward discount factor.
    kappa : float
        The agent's risk aversion.
    lam : float
        The cost friction parameter.
    market : Market
        The market on which the agent is operating.
    strategyType: StrategyType
        The strategy being performed by the agent. Refer to :obj:`StrategyType` for more details.

    """

    def __init__(self, market: Market):
        """
        Class constructor.

        Parameters
        ----------
        market : Market
            The market on which the agent is operating.

        """

        self._check_input(market)

        self.market = market
        self._set_attributes()
        self._set_lam()

    @staticmethod
    def _check_input(market: Market) -> None:

        if market.marketDynamics.riskDriverDynamics.riskDriverDynamicsType != RiskDriverDynamicsType.Linear:
            raise NameError('riskDriverDynamicsType for benchmark agent should be Linear')

        if market.marketDynamics.factorDynamics.factorDynamicsType != FactorDynamicsType.AR:
            raise NameError('factorDynamicsType for benchmark agent should be AR')

        if market.riskDriverType != RiskDriverType.PnL:
            raise NameError('riskDriverType for benchmark agent should be PnL')

    def compute_trading_cost(self, trade: float, factor: float, price: float) -> float:
        """
        Computes the trading cost associated to a specific trade.

        Parameters
        ----------
        trade : float
            Trade implemented by the agent.
        factor: float
            Observation of the factor, used to predict the price change variance sig2.
        price : float
            Price of the security.

        Returns
        -------
        trading_cost : float
            The trading cost.

        """

        sig2 = self.market.next_step_sig2(factor=factor, price=price)
        trading_cost = 0.5 * trade * self.lam * sig2 * trade

        return trading_cost

    def compute_trading_risk(self, factor: float, price: float, rescaled_shares: float, shares_scale: float) -> float:
        """
        Computes the risk associated to a specific trade.

        Parameters
        ----------
        factor: float
            Observation of the factor, used to predict the price change variance sig2.
        price : float
            Price of the security.
        rescaled_shares : float
            Shares, rescaled by :obj:`shares_scale`.
        shares_scale : float
            Factor for rescaling the shares.

        Returns
        -------
        trading_risk : float
            The trading risk.

        """

        sig2 = self.market.next_step_sig2(factor=factor, price=price)
        shares = rescaled_shares * shares_scale
        trading_risk = 0.5 * shares * self.kappa * sig2 * shares

        return trading_risk

    def _set_attributes(self):

        gamma, kappa, strategyType = self._read_trading_parameters()

        self.gamma = gamma
        self.kappa = kappa
        self.strategyType = strategyType

    def _read_trading_parameters(self):

        df_trad_params = self._get_df_trad_params()
        df_lam_kappa = self._get_df_lam_kappa()

        gamma = float(df_trad_params.loc['gamma'][0])
        kappa = float(df_lam_kappa.loc['kappa'])
        strategyType = StrategyType(df_trad_params.loc['strategyType'][0])

        return gamma, kappa, strategyType

    @staticmethod
    def _get_df_trad_params():

        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        filename += '/resources/data/data_source/settings.csv'
        df_trad_params = pd.read_csv(filename, index_col=0)
        return df_trad_params

    def _get_df_lam_kappa(self):

        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) +\
                   '/resources/data/data_source/market_data/commodities-summary-statistics.xlsx'
        df_lam_kappa = pd.read_excel(filename, index_col=0, sheet_name='Simplified contract multiplier')
        df_lam_kappa = df_lam_kappa.loc[self.market.ticker]
        return df_lam_kappa

    def _get_next_step_pnl_and_sig2(self, factor: float, price: float) -> Tuple[float, float]:

        pnl = self.market.next_step_pnl(factor=factor, price=price)
        sig2 = self.market.next_step_sig2(factor=factor, price=price)
        return pnl, sig2

    def _get_current_shares_pnl_and_sig2(self, factor: float, rescaled_shares: float,
                                         price: float, shares_scale: float) -> Tuple[float, float, float]:

        shares = rescaled_shares * shares_scale
        pnl, sig2 = self._get_next_step_pnl_and_sig2(factor, price)
        return shares, pnl, sig2

    def _set_lam(self):

        lam = self._read_lam()
        self.lam = lam

    def _read_lam(self) -> float:

        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) +\
                   '/resources/data/data_source/market_data/commodities-summary-statistics.xlsx'
        df_lam_kappa = pd.read_excel(filename, index_col=0, sheet_name='Simplified contract multiplier')
        df_lam_kappa = df_lam_kappa.loc[self.market.ticker]  # TODO: should it be self.environment.ticker?

        lam = float(df_lam_kappa.loc['lam'])

        return lam

    def _update_trade_by_strategyType(self, shares: float, trade: float) -> float:

        if self.strategyType == StrategyType.Unconstrained:
            pass
        elif self.strategyType == StrategyType.LongOnly:
            if shares + trade < 0:
                trade = - shares
        else:
            raise NameError(f'Invalid strategyType = {self.strategyType.value}')
        return trade


class AgentMarkowitz(AgentBenchmark):
    """
    Base class for Markowitz agent agent.

    Attributes
    ----------
    use_quadratic_cost_in_markowitz : bool
        If ``True``, it uses the static Markowitz solution taking into account transaction costs.

    """

    def __init__(self, market: Market):
        """
        Class constructor.

        Parameters
        ----------
        market : Market
            The market on which the agent is operating.

        """

        super().__init__(market)
        self._set_specific_markowitz_attributes()

    def policy(self, factor: float, rescaled_shares: float, shares_scale: float = 1,
               price: float = None) -> float:
        """
        Implements the Markowitz policy.

        Parameters
        ----------
        factor : float
            Current factor observation.
        rescaled_shares : float
            Current rescaled shares.
        shares_scale : float
            Factor for rescaling the shares.
        price : float
            Current price observation.

        Returns
        -------
        rescaled_trade: float
            The trade, rescaled by the factor `shares_scale`.

        """

        shares, pnl, sig2 = self._get_current_shares_pnl_and_sig2(factor, rescaled_shares, price, shares_scale)

        trade = self._get_markowitz_trade(shares, pnl, sig2)

        rescaled_trade = trade / shares_scale

        return rescaled_trade

    def _get_markowitz_trade(self, shares: float, pnl: float, sig2: float):

        if self.use_quadratic_cost_in_markowitz:
            trade = ((self.kappa * self.gamma + self.lam) * sig2) ** (-1) * (self.gamma * pnl + self.lam*sig2*shares)\
                    - shares
        else:
            trade = (self.kappa * sig2) ** (-1) * pnl - shares

        trade = self._update_trade_by_strategyType(shares, trade)

        return trade

    def _set_specific_markowitz_attributes(self):

        df_trad_params = self._get_df_trad_params()

        if str(df_trad_params.loc['use_quadratic_cost_in_markowitz'][0]) == 'Yes':
            use_quadratic_cost_in_markowitz = True
        elif str(df_trad_params.loc['use_quadratic_cost_in_markowitz'][0]) == 'No':
            use_quadratic_cost_in_markowitz = False
        else:
            raise NameError('Invalid value for parameter use_quadratic_cost_in_markowitz in settings.csv')

        self.use_quadratic_cost_in_markowitz = use_quadratic_cost_in_markowitz


class AgentGP(AgentBenchmark):
    """
    Base class for GP agent.

    """

    def __init__(self, market: Market):
        """
        Class constructor.

        Parameters
        ----------
        market : Market
            The market on which the agent is operating.

        """

        super().__init__(market)

    def policy(self, factor: float, rescaled_shares: float, shares_scale: float = 1,
               price: float = None) -> float:
        """
        Implements the GP policy.

        Parameters
        ----------
        factor : float
            Current factor observation.
        rescaled_shares : float
            Current rescaled shares.
        shares_scale : float
            Factor for rescaling the shares.
        price : float
            Current price observation.

        Returns
        -------
        rescaled_trade: float
            The trade, rescaled by the factor `shares_scale`.
        """

        shares, pnl, sig2 = self._get_current_shares_pnl_and_sig2(factor, rescaled_shares, price, shares_scale)

        trade = self._get_gp_trade(shares, pnl, sig2)

        rescaled_trade = trade / shares_scale

        return rescaled_trade

    def _get_gp_trade(self, shares: float, pnl: float, sig2: float) -> float:

        a = self._get_a()
        gp_rescaling = self._get_gp_rescaling(a)
        aim_ptf = (self.kappa * sig2) ** (-1) * gp_rescaling * pnl
        trade = (1 - a / self.lam) * shares + a / self.lam * aim_ptf - shares

        trade = self._update_trade_by_strategyType(shares, trade)

        return trade

    def _get_a(self) -> float:

        a = (-(self.kappa * self.gamma + self.lam * (1 - self.gamma)) +
             np.sqrt((self.kappa * self.gamma + self.lam * (1 - self.gamma)) ** 2 +
                     4 * self.kappa * self.lam * self.gamma ** 2)) / (2 * self.gamma)
        return a

    def _get_gp_rescaling(self, a: float) -> float:

        if self.market.riskDriverType != RiskDriverType.PnL:
            print('Trying to use GP with a model not on PnL. ',
                  f'The model is actually on {self.market.riskDriverType.value}')

        Phi = self._read_Phi()

        gp_rescaling = 1 / (1 + Phi * a / self.kappa)

        return gp_rescaling

    def _read_Phi(self) -> float:

        ticker = self.market.ticker
        riskDriverType = self.market.riskDriverType
        filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + \
            '/resources/data/financial_time_series_data/financial_time_series_calibrations/' + \
            ticker + '-riskDriverType-' + riskDriverType.value + '-factor-calibrations.xlsx'
        df_factor_params = pd.read_excel(filename, sheet_name='AR', index_col=0)
        Phi = 1 - df_factor_params.loc['B'][0]
        return Phi


if __name__ == '__main__':

    # Instantiate market
    market = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear,
                                factorDynamicsType=FactorDynamicsType.AR,
                                ticker='WTI',
                                riskDriverType=RiskDriverType.PnL)

    # Instantiate agents
    agentMarkowitz = AgentMarkowitz(market=market)
    agentGP = AgentGP(market=market)

    # Plot policies
    rescaled_shares = 1.
    factor_array = np.linspace(-0.2, 0.2, num=5)
    markowitz_action_list = []
    gp_action_list = []
    for factor in factor_array:
        markowitz_action_list.append(agentMarkowitz.policy(factor=factor,
                                                           rescaled_shares=rescaled_shares))
        gp_action_list.append(agentGP.policy(factor=factor,
                                             rescaled_shares=rescaled_shares))

    dpi = plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
    plt.plot(factor_array, markowitz_action_list, label='Markowitz')
    plt.plot(factor_array, gp_action_list, label='GP')
    plt.legend()
    plt.grid()
    plt.show()
