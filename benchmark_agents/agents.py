import pandas as pd

from enums import RiskDriverDynamicsType, RiskDriverType, FactorDynamicsType
from market_utils.market import Market, instantiate_market

class AgentBenchmark:

    def __init__(self, market: Market):

        self.market = market
        self._set_attributes()

    def _set_attributes(self):

        gamma, kappa, lam = self._read_trading_parameters()
        sig2, mu, B = self._read_dynamics_parameters()

        self.gamma = gamma
        self.kappa = kappa
        self.lam = lam
        self.sig2 = sig2
        self.mu = mu
        self.B = B

    def _read_dynamics_parameters(self):

        ticker = self.market.ticker
        riskDriverType = self.market.riskDriverType

        sig2 = self._get_sig2_from_file(riskDriverType, ticker)
        B, mu = self._get_mu_B_from_file(riskDriverType, ticker)

        return sig2, mu, B

    def _get_sig2_from_file(self, riskDriverType, ticker):
        filename = '../data/data_tmp/' + ticker + '-riskDriverType-' + \
                   riskDriverType.value + '-risk-driver-calibrations.xlsx'
        df_risk_driver_calibrations = pd.read_excel(filename, sheet_name='Linear', index_col=0)
        sig2 = df_risk_driver_calibrations.loc['sig2'][0]
        return sig2

    def _get_mu_B_from_file(self, riskDriverType, ticker):
        filename = '../data/data_tmp/' + ticker + '-riskDriverType-' + \
                   riskDriverType.value + '-factor-calibrations.xlsx'
        df_factor_calibrations = pd.read_excel(filename, sheet_name='AR', index_col=0)
        mu = df_factor_calibrations.loc['mu'][0]
        B = df_factor_calibrations.loc['B'][0]
        return B, mu

    def _read_trading_parameters(self):

        ticker = self.market.ticker

        filename = '../data/data_source/trading_data/' + ticker + '-trading-parameters.csv'
        df_trad_params = pd.read_csv(filename, index_col=0)

        gamma = df_trad_params.loc['gamma'][0]
        kappa = df_trad_params.loc['kappa'][0]
        lam = df_trad_params.loc['lam'][0]

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
