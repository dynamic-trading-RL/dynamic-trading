# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:16:57 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from enum import Enum
from scipy.optimize import (dual_annealing, shgo, differential_evolution,
                            brute, minimize)


# -----------------------------------------------------------------------------
# enum: ReturnDynamicsType
# -----------------------------------------------------------------------------

class ReturnDynamicsType(Enum):

    Linear = 'Linear'
    NonLinear = 'NonLinear'


# -----------------------------------------------------------------------------
# enum: FactorDynamicsType
# -----------------------------------------------------------------------------

class FactorDynamicsType(Enum):

    AR = 'AR'
    SETAR = 'SETAR'
    GARCH = 'GARCH'
    TARCH = 'TARCH'
    AR_TARCH = 'AR_TARCH'


# -----------------------------------------------------------------------------
# enum: FactorType
# -----------------------------------------------------------------------------

class FactorType(Enum):

    Observable = 'Observable'
    Latent = 'Latent'


# -----------------------------------------------------------------------------
# class: Dynamics
# -----------------------------------------------------------------------------

class Dynamics:

    def __init__(self):

        self._parameters = {}


# -----------------------------------------------------------------------------
# class: ReturnDynamics
# -----------------------------------------------------------------------------

class ReturnDynamics(Dynamics):

    def __init__(self, returnDynamicsType: ReturnDynamicsType):

        super().__init__()
        self._returnDynamicsType = returnDynamicsType

    def set_parameters(self, param_dict):

        if self._returnDynamicsType == ReturnDynamicsType.Linear:

            set_linear_parameters(self._parameters, param_dict)

        elif self._returnDynamicsType == ReturnDynamicsType.NonLinear:

            set_threshold_parameters(self._parameters, param_dict)

        else:
            raise NameError('Invalid return dynamics')


# -----------------------------------------------------------------------------
# class: FactorDynamics
# -----------------------------------------------------------------------------

class FactorDynamics(Dynamics):

    def __init__(self, factorDynamicsType: FactorDynamicsType):

        super().__init__()
        self._factorDynamicsType = factorDynamicsType

    def set_parameters(self, param_dict):

        if self._factorDynamicsType == FactorDynamicsType.AR:

            set_linear_parameters(self._parameters, param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.SETAR:

            set_threshold_parameters(self._parameters, param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.GARCH:

            set_garch_parameters(self._parameters, param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.TARCH:

            set_tarch_parameters(self._parameters, param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.AR_TARCH:

            set_artarch_parameters(self._parameters, param_dict)

        else:
            raise NameError('Invalid factor dynamics')


# -----------------------------------------------------------------------------
# class: MarketDynamics
# -----------------------------------------------------------------------------

class MarketDynamics:

    def __init__(self,
                 returnDynamics: ReturnDynamics,
                 factorDynamics: FactorDynamics):

        self._returnDynamics = returnDynamics
        self._factorDynamics = factorDynamics


# -----------------------------------------------------------------------------
# class: Market
# -----------------------------------------------------------------------------

class Market:

    def __init__(self, marketDynamics: MarketDynamics,
                 startPrice: float):

        self._marketDynamics = marketDynamics
        self._startPrice = startPrice
        self._marketId = self._setMarketId()
        self._simulations = {}

    def _setMarketId(self):

        returnDynamicsType =\
            self._marketDynamics._returnDynamics._returnDynamicsType

        factorDynamicsType =\
            self._marketDynamics._factorDynamics._factorDynamicsType

        self._marketId =\
            returnDynamicsType.value + '-' + factorDynamicsType.value

    def simulate(self, j_, t_):

        np.random.seed(789)
        self._simulate_factor(j_, t_)
        self._simulate_return()
        self._simulate_price()
        self._simulate_pnl()

    def _simulate_factor(self, j_, t_):

        factorDynamicsType =\
            self._marketDynamics._factorDynamics._factorDynamicsType
        parameters = self._marketDynamics._factorDynamics._parameters

        f = np.zeros((j_, t_))
        norm = np.random.randn(j_, t_)

        if factorDynamicsType == FactorDynamicsType.AR:

            mu = parameters['mu']
            B = parameters['B']
            sig2 = parameters['sig2']

            for t in range(1, t_):

                f[:, t] = B*f[:, t-1] + mu + np.sqrt(sig2)*norm[:, t]

        elif factorDynamicsType == FactorDynamicsType.SETAR:

            c = parameters['c']
            mu_0 = parameters['mu_0']
            B_0 = parameters['B_0']
            sig2_0 = parameters['sig2_0']
            mu_1 = parameters['mu_1']
            B_1 = parameters['B_1']
            sig2_1 = parameters['sig2_1']

            for t in range(1, t_):

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

            sig = np.zeros((j_, t_))
            sig[:, 0] = np.sqrt(omega)

            epsi = np.zeros((j_, t_))

            for t in range(1, t_):

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

            sig = np.zeros((j_, t_))
            sig[:, 0] = np.sqrt(omega)

            epsi = np.zeros((j_, t_))

            for t in range(1, t_):

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

            sig = np.zeros((j_, t_))
            sig[:, 0] = np.sqrt(omega)

            epsi = np.zeros((j_, t_))

            for t in range(1, t_):

                sig2 = omega + alpha*epsi[:, t-1]**2 + beta*sig[:, t-1]**2
                sig2[epsi[:, t-1] < c] += gamma*epsi[epsi[:, t-1] < c, t-1]
                sig[:, t] = np.sqrt(sig2)
                epsi[:, t] = sig[:, t]*norm[:, t]
                f[:, t] = B*f[:, t-1] + mu + epsi[:, t]

            self._simulations['sig'] = sig

        else:
            raise NameError('Invalid factorDynamicsType')

        self._simulations['f'] = f

    def _simulate_return(self):

        returnDynamicsType =\
            self._marketDynamics._returnDynamics._returnDynamicsType
        parameters = self._marketDynamics._returnDynamics._parameters

        f = self._simulations['f']
        j_, t_ = f.shape
        r = np.zeros((j_, t_))
        norm = np.random.randn(j_, t_)

        if returnDynamicsType == ReturnDynamicsType.Linear:

            mu = parameters['mu']
            B = parameters['B']
            sig2 = parameters['sig2']

            r[:, 1:] = mu + B*f[:, :-1] + np.sqrt(sig2)*norm[:, 1:]

        elif returnDynamicsType == ReturnDynamicsType.NonLinear:

            c = parameters['c']
            mu_0 = parameters['mu_0']
            B_0 = parameters['B_0']
            sig2_0 = parameters['sig2_0']
            mu_1 = parameters['mu_1']
            B_1 = parameters['B_1']
            sig2_1 = parameters['sig2_1']

            for t in range(1, t_):

                ind_0 = f[:, t-1] < c
                ind_1 = f[:, t-1] >= c

                r[ind_0, t] =\
                    mu_0 + B_0*f[ind_0, t-1] + np.sqrt(sig2_0)*norm[ind_0, t]

                r[ind_1, t] =\
                    mu_1 + B_1*f[ind_1, t-1] + np.sqrt(sig2_1)*norm[ind_1, t]

        else:
            raise NameError('Invalid returnDynamicsType')

        self._simulations['r'] = r

    def _simulate_price(self):

        r = self._simulations['r']
        j_, t_ = r.shape
        price = np.zeros((j_, t_))

        price[:, 0] = self._startPrice

        for t in range(1, t_):

            price[:, t] = price[:, t-1]*(1 + r[:, t])

        self._simulations['price'] = price

    def _simulate_pnl(self):

        j_, _ = self._simulations['price'].shape

        self._simulations['pnl'] =\
            np.c_[np.zeros(j_), np.diff(self._simulations['price'], axis=1)]


# -----------------------------------------------------------------------------
# class: AllMarkets
# -----------------------------------------------------------------------------

class AllMarkets:

    def __init__(self):

        self._allMarketsDict = {}

    def fill_allMarketsDict(self, d):

        for key, item in d.items():

            self._allMarketsDict[key] = item


# -----------------------------------------------------------------------------
# class: Optimizers
# -----------------------------------------------------------------------------

class Optimizers:

    def __init__(self):
        self._shgo = 0
        self._dual_annealing = 0
        self._differential_evolution = 0
        self._brute = 0
        self._local = 0

    def __repr__(self):

        return 'Used optimizers:\n' +\
            '  shgo: ' + str(self._shgo) + '\n' +\
            '  dual_annealing: ' + str(self._dual_annealing) + '\n' +\
            '  differential_evolution: ' + str(self._differential_evolution) + '\n' +\
            '  brute: ' + str(self._brute) + '\n' +\
            '  local: ' + str(self._local)

    def __str__(self):

        return 'Used optimizers:\n' +\
            '  shgo: ' + str(self._shgo) + '\n' +\
            '  dual_annealing: ' + str(self._dual_annealing) + '\n' +\
            '  differential_evolution: ' + str(self._differential_evolution) + '\n' +\
            '  brute: ' + str(self._brute) + '\n' +\
            '  local: ' + str(self._local)


# -----------------------------------------------------------------------------
# set_linear_parameters
# -----------------------------------------------------------------------------

def set_linear_parameters(parameters, param_dict):

    parameters['mu'] = param_dict['mu']
    parameters['B'] = param_dict['B']
    parameters['sig2'] = param_dict['sig2']


# -----------------------------------------------------------------------------
# set_threshold_parameters
# -----------------------------------------------------------------------------

def set_threshold_parameters(parameters, param_dict):

    parameters['c'] = param_dict['c']
    parameters['mu_0'] = param_dict['mu_0']
    parameters['B_0'] = param_dict['B_0']
    parameters['sig2_0'] = param_dict['sig2_0']
    parameters['mu_1'] = param_dict['mu_1']
    parameters['B_1'] = param_dict['B_1']
    parameters['sig2_1'] = param_dict['sig2_1']


# -----------------------------------------------------------------------------
# set_garch_parameters
# -----------------------------------------------------------------------------

def set_garch_parameters(parameters, param_dict):

    parameters['mu'] = param_dict['mu']
    parameters['omega'] = param_dict['omega']
    parameters['alpha'] = param_dict['alpha']
    parameters['beta'] = param_dict['beta']


# -----------------------------------------------------------------------------
# set_tarch_parameters
# -----------------------------------------------------------------------------

def set_tarch_parameters(parameters, param_dict):

    set_garch_parameters(parameters, param_dict)
    parameters['gamma'] = param_dict['gamma']
    parameters['c'] = param_dict['c']


# -----------------------------------------------------------------------------
# set_artarch_parameters
# -----------------------------------------------------------------------------

def set_artarch_parameters(parameters, param_dict):

    set_tarch_parameters(parameters, param_dict)
    parameters['B'] = param_dict['B']


# -----------------------------------------------------------------------------
# read_return_parameters
# -----------------------------------------------------------------------------

def read_return_parameters(returnDynamicsType):

    if returnDynamicsType == ReturnDynamicsType.Linear:

        params = pd.read_excel('data/return_calibrations.xlsx',
                               sheet_name='linear',
                               index_col=0)

    elif returnDynamicsType == ReturnDynamicsType.NonLinear:

        params = pd.read_excel('data/return_calibrations.xlsx',
                               sheet_name='non-linear',
                               index_col=0)

    else:
        raise NameError('Invalid returnDynamicsType')

    return params['param'].to_dict()


# -----------------------------------------------------------------------------
# read_factor_parameters
# -----------------------------------------------------------------------------

def read_factor_parameters(factorDynamicsType):

    if factorDynamicsType == FactorDynamicsType.AR:

        params = pd.read_excel('data/factor_calibrations.xlsx',
                               sheet_name='AR',
                               index_col=0)

    elif factorDynamicsType == FactorDynamicsType.SETAR:

        params = pd.read_excel('data/factor_calibrations.xlsx',
                               sheet_name='SETAR',
                               index_col=0)

    elif factorDynamicsType == FactorDynamicsType.GARCH:

        params = pd.read_excel('data/factor_calibrations.xlsx',
                               sheet_name='GARCH',
                               index_col=0)

    elif factorDynamicsType == FactorDynamicsType.TARCH:

        params = pd.read_excel('data/factor_calibrations.xlsx',
                               sheet_name='TARCH',
                               index_col=0)

    elif factorDynamicsType == FactorDynamicsType.AR_TARCH:

        params = pd.read_excel('data/factor_calibrations.xlsx',
                               sheet_name='AR-TARCH',
                               index_col=0)

    else:
        raise NameError('Invalid factorDynamicsType')

    return params['param'].to_dict()


# -----------------------------------------------------------------------------
# instantiate_market
# -----------------------------------------------------------------------------

def instantiate_market(returnDynamicsType, factorDynamicsType, startPrice):

    # Instantiate dynamics
    returnDynamics = ReturnDynamics(returnDynamicsType)
    factorDynamics = FactorDynamics(factorDynamicsType)

    # Read calibrated parameters
    return_parameters = read_return_parameters(returnDynamicsType)
    factor_parameters = read_factor_parameters(factorDynamicsType)

    # Set dynamics
    returnDynamics.set_parameters(return_parameters)
    factorDynamics.set_parameters(factor_parameters)
    marketDynamics = MarketDynamics(returnDynamics=returnDynamics,
                                    factorDynamics=factorDynamics)
    market = Market(marketDynamics, startPrice)

    return market


# -----------------------------------------------------------------------------
# simulate_market
# -----------------------------------------------------------------------------

def simulate_market(market, j_episodes, n_batches, t_):

    market.simulate(j_=j_episodes*n_batches, t_=t_)

    price = market._simulations['price'].reshape((j_episodes, n_batches, t_))
    pnl = market._simulations['pnl'].reshape((j_episodes, n_batches, t_))
    f = market._simulations['f'].reshape((j_episodes, n_batches, t_))

    return price, pnl, f


# -----------------------------------------------------------------------------
# get_Sigma
# -----------------------------------------------------------------------------

def get_Sigma(market):

    returnDynamicsType =\
        market._marketDynamics._returnDynamics._returnDynamicsType

    if returnDynamicsType == ReturnDynamicsType.Linear:

        Sigma_r = market._marketDynamics._returnDynamics._parameters['sig2']

    elif returnDynamicsType == ReturnDynamicsType.NonLinear:

        # ??? should become weighted average
        Sigma_r =\
            0.5*(market._marketDynamics._returnDynamics._parameters['sig2_0'] +
                 market._marketDynamics._returnDynamics._parameters['sig2_0'])

    return Sigma_r


# -----------------------------------------------------------------------------
# generate_episode
# -----------------------------------------------------------------------------

def generate_episode(
                     # dummy for parallel computing
                     j,
                     # market simulations
                     price, pnl, f, factorType,
                     # reward/cost parameters
                     rho, gamma, Sigma_r, Lambda_r,
                     # RL parameters
                     eps, q_value, alpha,
                     optimizers, optimizer,
                     b,
                     bound=400):
    """
    Given a market simulation, this function generates an episode for the
    reinforcement learning agent training
    """

    reward_total = 0
    cost_total = 0

    t_ = f.shape[1]

    if factorType == FactorType.Observable:
        x_episode = np.zeros((t_-1, 3))
    elif factorType == FactorType.Latent:
        x_episode = np.zeros((t_-1, 2))
    else:
        raise NameError('Invalid factorType: ' + factorType.value)

    y_episode = np.zeros(t_-1)

    # Observe state
    if factorType == FactorType.Observable:
        state = [0., f[j, 0]]
    elif factorType == FactorType.Latent:
        state = [0.]
    else:
        raise NameError('Invalid factorType: ' + factorType.value)

    # Choose action

    lb = -bound
    ub = bound
    if np.random.rand() < eps:
        action = lb + (ub - lb)*np.random.rand()
    else:
        action = maxAction(q_value, state, [lb, ub], b,
                           optimizers, optimizer)

    for t in range(1, t_):

        # Observe s'
        if factorType == FactorType.Observable:
            state_ = [state[0] + action, f[j, t]]
        elif factorType == FactorType.Latent:
            state_ = [state[0] + action]
        else:
            raise NameError('Invalid factorType: ' + factorType.value)

        # Choose a' from s' using policy derived from q_value
        lb = -bound
        ub = bound
        if np.random.rand() < eps:
            action_ = lb + (ub - lb)*np.random.rand()
        else:
            action_ = maxAction(q_value, state_, [lb, ub], b,
                                optimizers, optimizer)

        # Observe reward
        x_tm1 = state_[0]
        r_t = pnl[j, t]
        a_tm1 = action

        Sigma = price[j, t-1]*Sigma_r
        Lambda = price[j, t-1]*Lambda_r

        cost_tm1 = cost(x_tm1, a_tm1, rho, gamma, Sigma, Lambda)
        reward_t = reward(x_tm1, r_t, cost_tm1, rho)

        cost_total += cost_tm1
        reward_total += reward_t

        # Update value function
        y = q_value(state, action) +\
            alpha*(reward_t +
                   (1 - rho)*q_value(state_, action_) -
                   q_value(state, action))

        # Update fitting pairs
        x_episode[t-1] = np.r_[state, action]
        y_episode[t-1] = y

        # Update state and action
        state = state_
        action = action_

    return x_episode, y_episode, j, reward_total, cost_total


# -----------------------------------------------------------------------------
# reward
# -----------------------------------------------------------------------------

def reward(x_tm1, r_t, cost_tm1, rho):

    return (1 - rho)*x_tm1*r_t - cost_tm1


# -----------------------------------------------------------------------------
# cost
# -----------------------------------------------------------------------------

def cost(x_tm1, a_tm1, rho, gamma, Sigma, Lambda):

    return 0.5*((1 - rho)*gamma*x_tm1*Sigma*x_tm1 + a_tm1*Lambda*a_tm1)


# -----------------------------------------------------------------------------
# maxAction
# -----------------------------------------------------------------------------

def maxAction(q_value, state, bounds, b, optimizers, optimizer=None):
    """
    This function determines the q-greedy action for a given
    q-value function and state
    """

    if b == 0:

        return -bounds[0] + (bounds[1] - bounds[0])*np.random.rand()

    else:

        bounds = [tuple(bounds)]

        # function
        def fun(a): return -q_value(state, a)

        if optimizer == 'best':
            n = np.array([optimizers._shgo, optimizers._dual_annealing,
                          optimizers._differential_evolution,
                          optimizers._brute, optimizers._local])
            i = np.argmax(n)
            if i == 0:
                optimizer = 'shgo'
            elif i == 1:
                optimizer = 'dual_annealing'
            elif i == 2:
                optimizer = 'differential_evolution'
            elif i == 3:
                optimizer = 'brute'
            elif i == 4:
                optimizer = 'local'
            else:
                print('Wrong optimizer')

        if optimizer is None:

            # optimizations
            res1 = shgo(fun, bounds)
            res2 = dual_annealing(fun, bounds)
            res3 = differential_evolution(fun, bounds)
            res4 = brute(fun, ranges=bounds,
                         Ns=max(100, int(bounds[0][1]-bounds[0][0]+1)),
                         finish=None,
                         full_output=True)
            res5 = minimize(fun, x0=np.array([0]), bounds=bounds)

            res = [res1, res2, res3, res4[0], res5]
            res_fun = np.array([res1.fun, res2.fun, res3.fun, res4[1],
                                res5.fun])

            i = np.argmax(res_fun)

            if i == 0:
                optimizers._shgo += 1
            elif i == 1:
                optimizers._dual_annealing += 1
            elif i == 2:
                optimizers._differential_evolution += 1
            elif i == 3:
                optimizers._brute += 1
            elif i == 4:
                optimizers._local += 1
            else:
                print('Wrong optimizer')

            res = res[i]

        elif optimizer == 'shgo':
            optimizers._shgo += 1
            res = shgo(fun, bounds=bounds)

        elif optimizer == 'dual_annealing':
            optimizers._dual_annealing += 1
            res = dual_annealing(fun, bounds)

        elif optimizer == 'differential_evolution':
            optimizers._differential_evolution += 1
            res = differential_evolution(fun, bounds)

        elif optimizer == 'brute':

            optimizers._brute += 1
            x_brute = brute(fun, ranges=bounds,
                            Ns=max(100, int(bounds[0][1]-bounds[0][0]+1)),
                            finish=None)

            return x_brute

        elif optimizer == 'local':
            optimizers._local += 1
            res = minimize(fun, x0=np.array([0]), bounds=bounds)

        else:
            raise NameError('Wrong optimizer: ' + optimizer)

        return res.x[0]


# -----------------------------------------------------------------------------
# set_regressor_parameters
# -----------------------------------------------------------------------------

def set_regressor_parameters(sup_model):

    if sup_model == 'ann_fast':
        hidden_layer_sizes = (64, 32, 8)
        max_iter = 10
        n_iter_no_change = 2
        alpha_ann = 0.0001

        return hidden_layer_sizes, max_iter, n_iter_no_change, alpha_ann

    elif sup_model == 'ann_deep':
        hidden_layer_sizes = (70, 50, 30, 10)
        max_iter = 200
        n_iter_no_change = 10
        alpha_ann = 0.001

        return hidden_layer_sizes, max_iter, n_iter_no_change, alpha_ann

    elif sup_model == 'random_forest':

        return None


# -----------------------------------------------------------------------------
# q_hat
# -----------------------------------------------------------------------------

def q_hat(state, action,
          qb_list,
          flag_qaverage=True, n_models=None):
    """
    This function evaluates the estimated q-value function in a given state and
    action pair. The other parameters are given to include the cases of
    model averaging and data rescaling.
    """
    res = 0.
    is_simulation = (np.ndim(state) > 1)

    if flag_qaverage:

        if n_models is None or n_models > len(qb_list):
            n_models = len(qb_list)
        for b in range(1, n_models+1):
            qb = qb_list[-b]
            if is_simulation:
                res = 0.5*(res + qb.predict(np.c_[state, action]))
            else:
                res = 0.5*(res + qb.predict(np.r_[state,
                                                  action].reshape(1, -1)))
        return res
    else:
        qb = qb_list[-1]
        if is_simulation:
            res = res + qb.predict(np.c_[state, action])
        else:
            res = res + qb.predict(np.r_[state, action].reshape(1, -1))
        return res


# -----------------------------------------------------------------------------
# compute_markovitz
# -----------------------------------------------------------------------------

def compute_markovitz(f, gamma, B, Sigma, price, mu_r):

    if f.ndim == 1:
        t_ = f.shape[0]
        j_ = 1
        f = f.reshape((j_, t_))
        price = price.reshape((j_, t_))
    elif f.ndim == 2:
        j_, t_ = f.shape

    Markovitz = np.zeros((j_, t_))
    for t in range(t_):

        resc_f = price[:, t]*(f[:, t] + mu_r/B)
        resc_Sigma = price[:, t]**2 * Sigma

        Markovitz[:, t] = (gamma*resc_Sigma)**(-1)*B*resc_f

    return Markovitz.squeeze()


# -----------------------------------------------------------------------------
# compute_GP
# -----------------------------------------------------------------------------

def compute_GP(f, gamma, lam, rho, B, Sigma, Phi, price, mu_r):

    if f.ndim == 1:
        t_ = f.shape[0]
        j_ = 1
        f = f.reshape((j_, t_))
    elif f.ndim == 2:
        j_, t_ = f.shape

    x = np.zeros((j_, t_))
    for t in range(t_):

        resc_f = price[:, t]*(f[:, t] + mu_r/B)
        resc_Sigma = price[:, t]**2 * Sigma

        a = (-(gamma*(1 - rho) + lam*rho) +
             np.sqrt((gamma*(1-rho) + lam*rho)**2 +
                     4*gamma*lam*(1-rho)**2)) / (2*(1-rho))

        aim_t = (gamma*resc_Sigma)**(-1) * (B/(1+Phi*a/gamma))*resc_f

        if t == 0:
            x[:, t] = a/lam * aim_t

        else:
            x[:, t] = (1 - a/lam)*x[:, t-1] + a/lam * aim_t

    return x.squeeze()


# -----------------------------------------------------------------------------
# compute_rl
# -----------------------------------------------------------------------------

def compute_rl(j, f, q_value, factorType, optimizers, optimizer=None,
               bound=100):

    if f.ndim == 1:
        t_ = f.shape[0]
    else:
        t_ = f.shape[1]

    shares = np.zeros(t_)

    for t in range(t_):

        if t == 0:

            if factorType == FactorType.Observable:
                state = np.array([0, f[t]])
            elif factorType == FactorType.Latent:
                state = np.array([0])
            else:
                raise NameError('Invalid factorType: ' + factorType.value)

            action = maxAction(q_value, state, [-bound, bound], 1, optimizers,
                               optimizer=optimizer)
            shares[t] = state[0] + action
        else:

            if factorType == FactorType.Observable:
                state = np.array([shares[t-1], f[t]])
            elif factorType == FactorType.Latent:
                state = np.array([shares[t-1]])
            else:
                raise NameError('Invalid factorType: ' + factorType.value)

            bounds = [-bound, bound]
            action = maxAction(q_value, state, bounds, 1, optimizers,
                               optimizer=optimizer)
            shares[t] = state[0] + action

    return shares


# -----------------------------------------------------------------------------
# compute_wealth
# -----------------------------------------------------------------------------

def compute_wealth(pnl, strat, gamma, Lambda, rho, Sigma, price):

    if pnl.ndim == 1:
        t_ = pnl.shape[0]
        j_ = 1
        pnl = pnl.reshape((j_, t_))
        strat = strat.reshape((j_, t_))
    elif pnl.ndim == 2:
        j_, t_ = pnl.shape

    # Value
    value = np.zeros((j_, t_))
    for t in range(t_ - 1):
        value[:, t] = (1 - rho)**(t + 1) * strat[:, t] * pnl[:, t+1]
    value = np.cumsum(value, axis=1)

    # Costs
    cost = np.zeros((j_, t_))
    for t in range(1, t_):
        delta_strat = strat[:, t] - strat[:, t-1]

        resc_Sigma = price[:, t]**2 * Sigma
        resc_Lambda = price[:, t]**2 * Lambda

        cost[:, t] =\
            gamma/2 * (1 - rho)**(t + 1) * strat[:, t]*resc_Sigma*strat[:, t] +\
            0.5*(1 - rho)**t * delta_strat*resc_Lambda*delta_strat
    cost = np.cumsum(cost, axis=1)

    # Wealth
    wealth = value - cost

    return wealth.squeeze(), value.squeeze(), cost.squeeze()
