import numpy as np

from dynamics import MarketDynamics, AssetDynamics, FactorDynamics
from enums import AssetDynamicsType, FactorDynamicsType


class Market:

    def __init__(self, marketDynamics: MarketDynamics,
                 start_price: float,
                 return_is_pnl: bool = False):

        self._marketDynamics = marketDynamics
        self._start_price = start_price
        self._marketId = self._setMarketId()
        self._return_is_pnl = return_is_pnl
        self._simulations = {}

    def _setMarketId(self):

        assetDynamicsType =\
            self._marketDynamics._assetDynamics._assetDynamicsType

        factorDynamicsType =\
            self._marketDynamics._factorDynamics._factorDynamicsType

        self._marketId =\
            assetDynamicsType.value + '-' + factorDynamicsType.value

    def next_step(self, f):

        assetDynamicsType =\
            self._marketDynamics._assetDynamics._assetDynamicsType
        parameters = self._marketDynamics._assetDynamics._parameters

        if assetDynamicsType == AssetDynamicsType.Linear:

            return parameters['mu'] + parameters['B']*f

        elif assetDynamicsType == AssetDynamicsType.NonLinear:

            if f < parameters['c']:

                return parameters['mu_0'] + parameters['B_0']*f

            else:

                return parameters['mu_1'] + parameters['B_1']*f

    def simulate(self, j_, t_):

        np.random.seed(789)
        self._simulate_factor(j_, t_)
        self._simulate_return()
        self._simulate_price()
        self._simulate_pnl()

    def simulate_market_and_extract_simulations(self, j_episodes, n_batches, t_):

        self.simulate(j_=j_episodes * n_batches, t_=t_)

        price = self._simulations['price'].reshape((j_episodes, n_batches, t_))
        pnl = self._simulations['pnl'].reshape((j_episodes, n_batches, t_))
        f = self._simulations['f'].reshape((j_episodes, n_batches, t_))

        return price, pnl, f

    def get_Sigma(self):

        assetDynamicsType = \
            self._marketDynamics._assetDynamics._assetDynamicsType

        if assetDynamicsType == AssetDynamicsType.Linear:

            Sigma = self._marketDynamics._assetDynamics._parameters['sig2']

        elif assetDynamicsType == AssetDynamicsType.NonLinear:

            # ??? should become weighted average
            # ??? in the episode generation, it should get either sig2_0/1 and not
            # the weighted average
            Sigma = \
                0.5 * (self._marketDynamics._assetDynamics._parameters['sig2_0'] +
                       self._marketDynamics._assetDynamics._parameters['sig2_0'])

        return Sigma

    def _simulate_factor(self, j_, t_):

        factorDynamicsType =\
            self._marketDynamics._factorDynamics._factorDynamicsType
        parameters = self._marketDynamics._factorDynamics._parameters

        t_stationary = t_ + 50

        f = np.zeros((j_, t_stationary))
        norm = np.random.randn(j_, t_stationary)

        if factorDynamicsType == FactorDynamicsType.AR:

            mu = parameters['mu']
            B = parameters['B']
            sig2 = parameters['sig2']

            for t in range(1, t_stationary):

                f[:, t] = B*f[:, t-1] + mu + np.sqrt(sig2)*norm[:, t]

        elif factorDynamicsType == FactorDynamicsType.SETAR:

            c = parameters['c']
            mu_0 = parameters['mu_0']
            B_0 = parameters['B_0']
            sig2_0 = parameters['sig2_0']
            mu_1 = parameters['mu_1']
            B_1 = parameters['B_1']
            sig2_1 = parameters['sig2_1']

            for t in range(1, t_stationary):

                ind_0 = f[:, t-1] < c
                ind_1 = f[:, t-1] >= c

                f[ind_0, t] =\
                    B_0*f[ind_0, t-1] + mu_0 + np.sqrt(sig2_0)*norm[ind_0, t]
                f[ind_1, t] =\
                    B_1*f[ind_1, t-1] + mu_1 + np.sqrt(sig2_1)*norm[ind_1, t]

        elif factorDynamicsType == FactorDynamicsType.GARCH:

            mu = parameters['mu']
            omega = parameters['omega']
            alpha = parameters['alpha']
            beta = parameters['beta']

            sig = np.zeros((j_, t_stationary))
            sig[:, 0] = np.sqrt(omega)

            epsi = np.zeros((j_, t_stationary))

            for t in range(1, t_stationary):

                sig[:, t] = np.sqrt(omega
                                    + alpha*epsi[:, t-1]**2
                                    + beta*sig[:, t-1]**2)
                epsi[:, t] = sig[:, t]*norm[:, t]
                f[:, t] = f[:, t-1] + mu + epsi[:, t]

            self._simulations['sig'] = sig

        elif factorDynamicsType == FactorDynamicsType.TARCH:

            mu = parameters['mu']
            omega = parameters['omega']
            alpha = parameters['alpha']
            beta = parameters['beta']
            gamma = parameters['gamma']
            c = parameters['c']

            sig = np.zeros((j_, t_stationary))
            sig[:, 0] = np.sqrt(omega)

            epsi = np.zeros((j_, t_stationary))

            for t in range(1, t_stationary):

                sig2 = omega + alpha*epsi[:, t-1]**2 + beta*sig[:, t-1]**2
                sig2[epsi[:, t-1] < c] += gamma*epsi[epsi[:, t-1] < c, t-1]
                sig[:, t] = np.sqrt(sig2)
                epsi[:, t] = sig[:, t]*norm[:, t]
                f[:, t] = f[:, t-1] + mu + epsi[:, t]

            self._simulations['sig'] = sig

        elif factorDynamicsType == FactorDynamicsType.AR_TARCH:

            mu = parameters['mu']
            B = parameters['B']
            omega = parameters['omega']
            alpha = parameters['alpha']
            beta = parameters['beta']
            gamma = parameters['gamma']
            c = parameters['c']

            sig = np.zeros((j_, t_stationary))
            sig[:, 0] = np.sqrt(omega)

            epsi = np.zeros((j_, t_stationary))

            for t in range(1, t_stationary):

                sig2 = omega + alpha*epsi[:, t-1]**2 + beta*sig[:, t-1]**2
                sig2[epsi[:, t-1] < c] += gamma*epsi[epsi[:, t-1] < c, t-1]
                sig[:, t] = np.sqrt(sig2)
                epsi[:, t] = sig[:, t]*norm[:, t]
                f[:, t] = B*f[:, t-1] + mu + epsi[:, t]

            self._simulations['sig'] = sig

        else:
            raise NameError('Invalid factorDynamicsType')

        self._simulations['f'] = f[:, -t_:]

    def _simulate_return(self):

        assetDynamicsType =\
            self._marketDynamics._assetDynamics._assetDynamicsType
        parameters = self._marketDynamics._assetDynamics._parameters

        f = self._simulations['f']
        j_, t_ = f.shape
        r = np.zeros((j_, t_))
        norm = np.random.randn(j_, t_)

        if assetDynamicsType == AssetDynamicsType.Linear:

            r[:, 1:] = self.next_step(f[:, :-1]) +\
                np.sqrt(parameters['sig2'])*norm[:, 1:]

        elif assetDynamicsType == AssetDynamicsType.NonLinear:

            c = parameters['c']
            sig2_0 = parameters['sig2_0']
            sig2_1 = parameters['sig2_1']

            for t in range(1, t_):

                ind_0 = f[:, t-1] < c
                ind_1 = f[:, t-1] >= c

                r[ind_0, t] = np.vectorize(self.next_step,
                                           otypes=[float])(f[ind_0, t-1])\
                    + np.sqrt(sig2_0)*norm[ind_0, t]

                r[ind_1, t] = np.vectorize(self.next_step,
                                           otypes=[float])(f[ind_1, t-1])\
                    + np.sqrt(sig2_1)*norm[ind_1, t]

        else:
            raise NameError('Invalid assetDynamicsType')

        self._simulations['r'] = r

    def _simulate_price(self):

        r = self._simulations['r']
        j_, t_ = r.shape
        price = np.zeros((j_, t_))

        price[:, 0] = self._start_price

        for t in range(1, t_):

            if self._return_is_pnl:

                price[:, t] = price[:, t-1] + r[:, t]

            else:

                price[:, t] = price[:, t-1]*(1 + r[:, t])

        self._simulations['price'] = price

    def _simulate_pnl(self):

        j_, _ = self._simulations['price'].shape

        self._simulations['pnl'] =\
            np.c_[np.zeros(j_), np.diff(self._simulations['price'], axis=1)]

        if self._return_is_pnl:

            must_be_zero = np.max(np.abs(self._simulations['pnl'] -
                                         self._simulations['r']))

            if must_be_zero > 10**-10:

                raise NameError('Return must be equal to PnL')


def instantiate_market(assetDynamicsType, factorDynamicsType, start_price,
                       return_is_pnl):

    # Instantiate dynamics
    assetDynamics = AssetDynamics(assetDynamicsType)
    factorDynamics = FactorDynamics(factorDynamicsType)

    # Read calibrated parameters
    return_parameters = assetDynamics.read_asset_parameters()
    factor_parameters = factorDynamics.read_factor_parameters()

    # Set dynamics
    assetDynamics.set_parameters(return_parameters)
    factorDynamics.set_parameters(factor_parameters)
    marketDynamics = MarketDynamics(assetDynamics=assetDynamics,
                                    factorDynamics=factorDynamics)
    market = Market(marketDynamics, start_price, return_is_pnl=return_is_pnl)

    return market





