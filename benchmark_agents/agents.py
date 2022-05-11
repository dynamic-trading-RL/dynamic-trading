import pandas as pd

from enums import RiskDriverDynamicsType, RiskDriverType, FactorDynamicsType
from market_utils.market import Market, instantiate_market

class AgentBenchmark:

    def __init__(self, market: Market):

        self.market = market
        self._set_attributes()

    def _set_attributes(self):

        gamma, kappa, lam = self._read_trading_parameters()
        sig2, mu_factor, B_factor = self._read_dynamics_parameters()

        self.gamma = gamma
        self.kappa = kappa
        self.lam = lam
        self.sig2 = sig2
        self.mu_factor = mu_factor
        self.B_factor = B_factor

    def _read_dynamics_parameters(self):

        ticker = self.market.marketDynamics.riskDriverDynamics.tic

        filename = '../data/data_source/trading-parameters.csv'

        df_trad_params = pd.read_csv(filename, index_col=0)

        gamma = df_trad_params.loc['gamma'].iloc[0]
        kappa = df_trad_params.loc['kappa'].iloc[0]
        lam = df_trad_params.loc['lam'].iloc[0]

        return gamma, kappa, lam

        return sig2, mu_factor, B_factor

    def _read_trading_parameters(self):

        filename = '../data/data_source/trading-parameters.csv'

        df_trad_params = pd.read_csv(filename, index_col=0)

        gamma = df_trad_params.loc['gamma'].iloc[0]
        kappa = df_trad_params.loc['kappa'].iloc[0]
        lam = df_trad_params.loc['lam'].iloc[0]

        return gamma, kappa, lam


class AgentMarkowitz(AgentBenchmark):

    def __init__(self, market: Market):

        super().__init__(market)

    def policy(self, current_factor: float, current_rescaled_shares: float, shares_scale: float = 1):

        current_shares = current_rescaled_shares * shares_scale

        trade = (self.kappa * self.sig2)**(-1) * self.B * current_factor - current_shares

        rescaled_trade = trade / shares_scale

        return rescaled_trade


if __name__ == '__main__':

    market = instantiate_market(riskDriverDynamicsType=RiskDriverDynamicsType.Linear,
                                factorDynamicsType=FactorDynamicsType.AR,
                                ticker='WTI',
                                riskDriverType=RiskDriverType.PnL)

    agentMarkowitz = AgentMarkowitz(market=market)
