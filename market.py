import numpy as np
import matplotlib.pyplot as plt
from dynamics import MarketDynamics, AssetDynamics, FactorDynamics
from enums import AssetDynamicsType, FactorDynamicsType


class Market:

    def __init__(self, marketDynamics: MarketDynamics, start_price: float):

        self.marketDynamics = marketDynamics
        self.use_pnl = self.marketDynamics.use_pnl
        self._setMarketId()
        self.start_price = start_price
        self.simulations = {}

    def _setMarketId(self):

        assetDynamicsType = self.marketDynamics.assetDynamics.assetDynamicsType
        factorDynamicsType = self.marketDynamics.factorDynamics.factorDynamicsType
        self.marketId = assetDynamicsType.value + '-' + factorDynamicsType.value

    def next_step(self, f: float):

        assetDynamicsType = self.marketDynamics.assetDynamics.assetDynamicsType
        parameters = self.marketDynamics.assetDynamics.parameters

        if assetDynamicsType == AssetDynamicsType.Linear:

            return parameters['mu'] + parameters['B']*f

        elif assetDynamicsType == AssetDynamicsType.NonLinear:

            if f < parameters['c']:

                return parameters['mu_0'] + parameters['B_0']*f

            else:

                return parameters['mu_1'] + parameters['B_1']*f

    def simulate(self, j_, t_, delta_stationary: int = 50):

        np.random.seed(789)
        self._simulate_factor(j_, t_, delta_stationary)
        self._simulate_return_and_pnl_and_price()

    def simulate_batches(self, j_episodes, n_batches, t_):

        self.simulate(j_=j_episodes * n_batches, t_=t_)

        price = self.simulations['price'].reshape((j_episodes, n_batches, t_))
        pnl = self.simulations['pnl'].reshape((j_episodes, n_batches, t_))
        ret = self.simulations['return'].reshape((j_episodes, n_batches, t_))
        f = self.simulations['f'].reshape((j_episodes, n_batches, t_))

        return price, pnl, ret, f

    def _simulate_factor(self, j_, t_, delta_stationary):

        factorDynamicsType = self.marketDynamics.factorDynamics.factorDynamicsType
        parameters = self.marketDynamics.factorDynamics.parameters

        f, norm, t_stationary = self._initialize_factor_simulations(delta_stationary, j_, t_)

        if factorDynamicsType == FactorDynamicsType.AR:

            self._simulate_factor_ar(f, norm, parameters, t_stationary)

        elif factorDynamicsType == FactorDynamicsType.SETAR:

            self._simulate_factor_setar(f, norm, parameters, t_stationary)

        elif factorDynamicsType == FactorDynamicsType.GARCH:

            self._simulate_factor_garch(f, j_, norm, parameters, t_stationary)

        elif factorDynamicsType == FactorDynamicsType.TARCH:

            self._simulate_factor_tarch(f, j_, norm, parameters, t_stationary)

        elif factorDynamicsType == FactorDynamicsType.AR_TARCH:

            self._simulate_factor_ar_tarch(f, j_, norm, parameters, t_stationary)

        else:
            raise NameError('Invalid factorDynamicsType')

        self.simulations['f'] = f[:, -t_:]

    def _simulate_return_and_pnl_and_price(self):
        if self.use_pnl:
            self._simulate_pnl_then_return_then_price()
        else:
            self._simulate_return_then_pnl_then_price()

    def _simulate_pnl_then_return_then_price(self):

        pnl = self._simulate_asset_impl()

        self.simulations['pnl'] = pnl

        self.simulations['return'] = \
            self.simulations['pnl'] / (self.start_price + np.cumsum(self.simulations['pnl'], axis=1))

        self._get_price_from_pnl()

    def _simulate_return_then_pnl_then_price(self):

        ret = self._simulate_asset_impl()

        self.simulations['return'] = ret

        self.simulations['pnl'] = \
            self.start_price * np.cumprod(1 + self.simulations['return'], axis=1) / \
            (1 + self.simulations['return']) * self.simulations['return']

        self._get_price_from_pnl()

    def _simulate_asset_impl(self):
        assetDynamicsType, parameters = self.marketDynamics.get_assetDynamicsType_and_parameters()
        pnl = self._simulate_asset(assetDynamicsType, parameters)
        return pnl

    def _simulate_asset(self, assetDynamicsType, parameters):
        f, norm, asset, t_ = self._initialize_asset_simulations()
        if assetDynamicsType == AssetDynamicsType.Linear:

            self._simulate_asset_linear(asset, f, norm, parameters)

        elif assetDynamicsType == AssetDynamicsType.NonLinear:

            self._simulate_asset_non_linear(asset, f, norm, parameters, t_)

        else:
            raise NameError('Invalid assetDynamicsType')
        return asset

    def _simulate_asset_linear(self, asset, f, norm, parameters):
        sig2 = parameters['sig2']
        asset[:, 1:] = self.next_step(f[:, :-1]) + np.sqrt(sig2) * norm[:, 1:]

    def _simulate_asset_non_linear(self, asset, f, norm, parameters, t_):
        c = parameters['c']
        sig2_0 = parameters['sig2_0']
        sig2_1 = parameters['sig2_1']
        for t in range(1, t_):
            ind_0, ind_1 = self._get_threshold_indexes(c, f, t)
            self._get_next_step_asset(asset, f, ind_0, norm, sig2_0, t)
            self._get_next_step_asset(asset, f, ind_1, norm, sig2_1, t)

    def get_Sigma(self):

        assetDynamicsType = \
            self.marketDynamics.assetDynamics.assetDynamicsType

        if assetDynamicsType == AssetDynamicsType.Linear:

            return self.marketDynamics.assetDynamics.parameters['sig2']

        elif assetDynamicsType == AssetDynamicsType.NonLinear:

            # ??? should become weighted average
            # ??? in the episode generation, it should get either sig2_0/1 and not
            # the weighted average
            return 0.5 * (self.marketDynamics.assetDynamics.parameters['sig2_0']
                          + self.marketDynamics.assetDynamics.parameters['sig2_0'])

    def _simulate_factor_ar(self, f, norm, parameters, t_stationary):
        B, mu, sig2 = self._get_linear_params(parameters)
        ind = range(f.shape[0])
        for t in range(1, t_stationary):
            self._next_step_factor_linear(B, f, ind, mu, norm, sig2, t)

    def _simulate_factor_setar(self, f, norm, parameters, t_stationary):
        B_0, B_1, c, mu_0, mu_1, sig2_0, sig2_1 = self._get_threshold_params(parameters)
        for t in range(1, t_stationary):
            ind_0, ind_1 = self._get_threshold_indexes(c, f, t)

            self._next_step_factor_linear(B_0, f, ind_0, mu_0, norm, sig2_0, t)
            self._next_step_factor_linear(B_1, f, ind_1, mu_1, norm, sig2_1, t)

    def _simulate_factor_garch(self, f, j_, norm, parameters, t_stationary):
        alpha, beta, mu, omega = self._get_garch_parameters(parameters)
        epsi, sig = self._initialize_arch_simulations(j_, omega, t_stationary)
        for t in range(1, t_stationary):
            sig2 = self._get_next_step_sig2_arch(alpha, beta, epsi, omega, sig, t)

            self._get_next_step_arch(epsi, f, mu, norm, sig, sig2, t)
        self.simulations['sig'] = sig

    def _simulate_factor_tarch(self, f, j_, norm, parameters, t_stationary):
        alpha, beta, c, gamma, mu, omega = self._get_tarch_parameters(parameters)
        epsi, sig = self._initialize_arch_simulations(j_, omega, t_stationary)
        for t in range(1, t_stationary):
            sig2 = self._get_next_step_sig2_tarch(alpha, beta, c, epsi, gamma, omega, sig, t)

            self._get_next_step_arch(epsi, f, mu, norm, sig, sig2, t)
        self.simulations['sig'] = sig

    def _simulate_factor_ar_tarch(self, f, j_, norm, parameters, t_stationary):
        B, alpha, beta, c, gamma, mu, omega = self._get_ar_tarch_parameters(parameters)
        epsi, sig = self._initialize_arch_simulations(j_, omega, t_stationary)
        for t in range(1, t_stationary):
            sig2 = self._get_next_step_sig2_tarch(alpha, beta, c, epsi, gamma, omega, sig, t)

            self._get_next_step_ar_arch(B, epsi, f, mu, norm, sig, sig2, t)
        self.simulations['sig'] = sig

    def _initialize_asset_simulations(self):
        f = self.simulations['f']
        j_, t_ = f.shape
        asset = np.zeros((j_, t_))
        norm = np.random.randn(j_, t_)
        return f, norm, asset, t_

    def _initialize_factor_simulations(self, delta_stationary, j_, t_):
        t_stationary = t_ + delta_stationary
        f = np.zeros((j_, t_stationary))
        norm = np.random.randn(j_, t_stationary)
        return f, norm, t_stationary

    def _initialize_arch_simulations(self, j_, omega, t_stationary):
        sig = np.zeros((j_, t_stationary))
        sig[:, 0] = np.sqrt(omega)
        epsi = np.zeros((j_, t_stationary))
        return epsi, sig

    def _get_next_step_sig2_arch(self, alpha, beta, epsi, omega, sig, t):
        sig2 = omega + alpha * epsi[:, t - 1] ** 2 + beta * sig[:, t - 1] ** 2
        return sig2

    def _get_next_step_sig2_tarch(self, alpha, beta, c, epsi, gamma, omega, sig, t):
        sig2 = self._get_next_step_sig2_arch(alpha, beta, epsi, omega, sig, t)
        sig2[epsi[:, t - 1] < c] += gamma * epsi[epsi[:, t - 1] < c, t - 1]
        return sig2

    def _get_next_step_arch(self, epsi, f, mu, norm, sig, sig2, t):
        sig[:, t] = np.sqrt(sig2)
        epsi[:, t] = sig[:, t] * norm[:, t]
        f[:, t] = mu + f[:, t - 1] + epsi[:, t]

    def _get_next_step_ar_arch(self, B, epsi, f, mu, norm, sig, sig2, t):
        self._get_next_step_arch(epsi, f, mu, norm, sig, sig2, t)
        f[:, t] += (B - 1) * f[:, t - 1]

    def _get_ar_tarch_parameters(self, parameters):
        alpha, beta, c, gamma, mu, omega = self._get_tarch_parameters(parameters)
        B = parameters['B']
        return B, alpha, beta, c, gamma, mu, omega

    def _get_tarch_parameters(self, parameters):
        alpha, beta, mu, omega = self._get_garch_parameters(parameters)
        gamma = parameters['gamma']
        c = parameters['c']
        return alpha, beta, c, gamma, mu, omega

    def _get_garch_parameters(self, parameters):
        mu = parameters['mu']
        omega = parameters['omega']
        alpha = parameters['alpha']
        beta = parameters['beta']
        return alpha, beta, mu, omega

    def _get_threshold_indexes(self, c, f, t):
        ind_0 = f[:, t - 1] < c
        ind_1 = f[:, t - 1] >= c
        return ind_0, ind_1

    def _get_threshold_params(self, parameters):
        c = parameters['c']
        mu_0 = parameters['mu_0']
        B_0 = parameters['B_0']
        sig2_0 = parameters['sig2_0']
        mu_1 = parameters['mu_1']
        B_1 = parameters['B_1']
        sig2_1 = parameters['sig2_1']
        return B_0, B_1, c, mu_0, mu_1, sig2_0, sig2_1

    def _get_linear_params(self, parameters):
        mu = parameters['mu']
        B = parameters['B']
        sig2 = parameters['sig2']
        return B, mu, sig2

    def _get_price_from_pnl(self):
        self.simulations['price'] = self.start_price + np.cumsum(self.simulations['pnl'], axis=1)

    def _get_next_step_asset(self, asset, f, ind_0, norm, sig2_0, t):
        asset[ind_0, t] = np.vectorize(self.next_step, otypes=[float])(f[ind_0, t - 1]) + np.sqrt(sig2_0) * norm[
            ind_0, t]

    def _next_step_factor_linear(self, B, f, ind, mu, norm, sig2, t):
        f[ind, t] = mu + B * f[ind, t - 1] + np.sqrt(sig2) * norm[ind, t]


class AllMarkets:

    def __init__(self):

        self._allMarkets_dict = {}

    def fill_allMarkets_dict(self, d):

        for key, item in d.items():

            self._allMarkets_dict[key] = item


def instantiate_market(assetDynamicsType: AssetDynamicsType,
                       factorDynamicsType: FactorDynamicsType,
                       ticker: str):

    # Instantiate dynamics
    assetDynamics = AssetDynamics(assetDynamicsType)
    factorDynamics = FactorDynamics(factorDynamicsType)

    # Read calibrated parameters
    assetDynamics.read_asset_parameters(ticker)
    factorDynamics.read_factor_parameters(ticker)

    start_price = assetDynamics.read_asset_start_price(ticker)

    # Set dynamics
    marketDynamics = MarketDynamics(assetDynamics=assetDynamics,
                                    factorDynamics=factorDynamics)
    market = Market(marketDynamics, start_price)

    return market


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':

    j_ = 100
    t_ = 100

    market = instantiate_market(assetDynamicsType=AssetDynamicsType.NonLinear,
                                factorDynamicsType=FactorDynamicsType.AR_TARCH,
                                ticker='WTI')

    market.simulate(j_=j_, t_=t_)

    fig = plt.figure()

    ax_price = plt.subplot2grid(shape=(4, 1), loc=(0, 0))
    ax_price.set_title('Price')
    ax_pnl = plt.subplot2grid(shape=(4, 1), loc=(1, 0))
    ax_pnl.set_title('PnL')
    ax_return = plt.subplot2grid(shape=(4, 1), loc=(2, 0))
    ax_return.set_title('Return')
    ax_factor = plt.subplot2grid(shape=(4, 1), loc=(3, 0))
    ax_factor.set_title('Factor')

    for j in range(min(j_, 50)):
        ax_price.plot(market.simulations['price'][j, :])
        ax_pnl.plot(market.simulations['pnl'][j, :])
        ax_return.plot(market.simulations['return'][j, :])
        ax_factor.plot(market.simulations['f'][j, :])

    plt.tight_layout()
    plt.show()
