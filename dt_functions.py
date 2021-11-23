# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:16:57 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from enum import Enum
from scipy.optimize import (dual_annealing, shgo, differential_evolution)


# -----------------------------------------------------------------------------
# enum: ReturnDynamicsType
# -----------------------------------------------------------------------------

class ReturnDynamicsType(Enum):

    R1 = 'R1'
    R2 = 'R2'


# -----------------------------------------------------------------------------
# enum: FactorDynamicsType
# -----------------------------------------------------------------------------

class FactorDynamicsType(Enum):

    F1 = 'F1'
    F2 = 'F2'
    F3 = 'F3'
    F4 = 'F4'
    F5 = 'F5'


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

        if self._returnDynamicsType == ReturnDynamicsType.R1:

            set_linear_parameters(self._parameters, param_dict)

        elif self._returnDynamicsType == ReturnDynamicsType.R2:

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

        if self._factorDynamicsType == FactorDynamicsType.F1:

            set_linear_parameters(self._parameters, param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.F2:

            set_threshold_parameters(self._parameters, param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.F3:

            set_garch_parameters(self._parameters, param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.F4:

            set_tarch_parameters(self._parameters, param_dict)

        elif self._factorDynamicsType == FactorDynamicsType.F5:

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

    def __init__(self, marketDynamics: MarketDynamics):

        self._marketDynamics = marketDynamics
        self._simulations = {}

    def simulate(self, j_, t_):

        self._simulate_factor(j_, t_)
        self._simulate_return()

    def _simulate_factor(self, j_, t_):

        factorDynamicsType =\
            self._marketDynamics._factorDynamics._factorDynamicsType
        parameters = self._marketDynamics._factorDynamics._parameters

        f = np.zeros((j_, t_))
        norm = np.random.randn(j_, t_)

        if factorDynamicsType == FactorDynamicsType.F1:

            mu = parameters['mu']
            B = parameters['B']
            sig2 = parameters['sig2']

            for t in range(1, t_):

                f[:, t] = B*f[:, t-1] + mu + np.sqrt(sig2)*norm[:, t]

        elif factorDynamicsType == FactorDynamicsType.F2:

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

                f[ind_0, t] = B_0*f[ind_0, t-1] + mu_0 + np.sqrt(sig2_0)*norm[ind_0, t]
                f[ind_1, t] = B_1*f[ind_1, t-1] + mu_1 + np.sqrt(sig2_1)*norm[ind_1, t]

        elif factorDynamicsType == FactorDynamicsType.F3:

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

        elif factorDynamicsType == FactorDynamicsType.F4:

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
                sig2[epsi[:, t-1] < 0] += gamma*epsi[epsi[:, t-1] < c, t-1]
                sig[:, t] = np.sqrt(sig2)
                epsi[:, t] = sig[:, t]*norm[:, t]
                f[:, t] = f[:, t-1] + mu + epsi[:, t]

            self._simulations['sig'] = sig

        elif factorDynamicsType == FactorDynamicsType.F5:

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
                sig2[epsi[:, t-1] < 0] += gamma*epsi[epsi[:, t-1] < c, t-1]
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

        if returnDynamicsType == ReturnDynamicsType.R1:

            mu = parameters['mu']
            B = parameters['B']
            sig2 = parameters['sig2']

            r[:, 1:] = mu + B*f[:, :-1] + np.sqrt(sig2)*norm[:, 1:]

        elif returnDynamicsType == ReturnDynamicsType.R2:

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
# class: Optimizers
# -----------------------------------------------------------------------------

class Optimizers:

    def __init__(self):
        self._shgo = 0
        self._dual_annealing = 0
        self._differential_evolution = 0

    def __repr__(self):

        return 'Used optimizers:\n' +\
            '  shgo: ' + str(self._shgo) + '\n' +\
            '  dual_annealing: ' + str(self._dual_annealing) + '\n' +\
            '  differential_evolution: ' + str(self._differential_evolution)

    def __str__(self):

        return 'Used optimizers:\n' +\
            '  shgo: ' + str(self._shgo) + '\n' +\
            '  dual_annealing: ' + str(self._dual_annealing) + '\n' +\
            '  differential_evolution: ' + str(self._differential_evolution)


# -----------------------------------------------------------------------------
# simulate_market
# -----------------------------------------------------------------------------


def simulate_market(j_, t_, n_batches, B, mu_u, Sigma, Phi, mu_eps, Omega):

    f = np.zeros((j_, n_batches, t_))
    f[:, :, 0] = mu_eps + np.sqrt(Omega)*np.random.randn(j_, n_batches)
    for t in range(1, t_-1):
        f[:, :, t] = mu_eps + (1 - Phi)*f[:, :, t-1] +\
                np.sqrt(Omega)*np.random.randn(j_, n_batches)

    r = np.zeros((j_, n_batches, t_))
    r[:, :, 0] = 0.
    r[:, :, 1:] = mu_u + B*f[:, :, :-1] +\
        np.sqrt(Sigma)*np.random.randn(j_, n_batches, t_-1)

    return r.squeeze(), f.squeeze()


# -----------------------------------------------------------------------------
# reward
# -----------------------------------------------------------------------------


def reward(x_tm1, x_t, f_t,
           Lambda, B, mu_u, Sigma,
           rho, gamma):

    delta_x = x_t - x_tm1

    return -0.5*delta_x*Lambda*delta_x +\
        (1 - rho)*(x_t*(B*f_t + mu_u) - 0.5*gamma*x_t*Sigma*x_t)


# -----------------------------------------------------------------------------
# maxAction
# -----------------------------------------------------------------------------

def maxAction(q_value, state, lot_size, optimizers, optimizer=None):
    """
    This function determines the q-greedy action for a given
    q-value function and state
    """

    # function
    def fun(a): return -q_value(state, a)

    if optimizer == 'best':
        n = np.array([optimizers._shgo, optimizers._dual_annealing,
                      optimizers._differential_evolution])
        i = np.argmax(n)
        if i == 0:
            optimizer = 'shgo'
        elif i == 1:
            optimizer = 'dual_annealing'
        elif i == 2:
            optimizer = 'differential_evolution'
        else:
            print('Wrong optimizer')

    if optimizer is None:

        # optimizations
        res1 = shgo(fun, bounds=[(-lot_size, lot_size)])
        res2 = dual_annealing(fun, bounds=[(-lot_size, lot_size)])
        res3 = differential_evolution(fun, bounds=[(-lot_size, lot_size)])

        res_x = np.array([res1.x, res2.x, res3.x])
        res_fun = np.array([res1.fun, res2.fun, res3.fun])

        i = np.argmax(res_fun)

        if i == 0:
            optimizers._shgo += 1
        elif i == 1:
            optimizers._dual_annealing += 1
        elif i == 2:
            optimizers._differential_evolution += 1
        else:
            print('Wrong optimizer')

        return res_x[i]

    elif optimizer == 'shgo':
        res = shgo(fun, bounds=[(-lot_size, lot_size)])
        return res.x

    elif optimizer == 'dual_annealing':
        res = dual_annealing(fun, bounds=[(-lot_size, lot_size)])
        return res.x

    elif optimizer == 'differential_evolution':
        res = differential_evolution(fun, bounds=[(-lot_size, lot_size)])
        return res.x

    else:
        print('Wrong optimizer')


# -----------------------------------------------------------------------------
# generate_episode
# -----------------------------------------------------------------------------

def generate_episode(
                     # dummy for parallel computing
                     j,
                     # market parameters
                     Lambda, B, mu_u, Sigma,
                     # market simulations
                     f,
                     # RL parameters
                     eps, rho, q_value, alpha, gamma, lot_size,
                     optimizers,
                     optimizer):
    """
    Given a market simulation f, this function generates an episode for the
    reinforcement learning agent training
    """

    reward_total = 0
    cost_total = 0

    t_ = f.shape[1]

    x_episode = np.zeros((t_-1, 3))
    y_episode = np.zeros(t_-1)

    # state at t=0
    state = np.array([0, f[j, 0]])

    # choose action
    if np.random.rand() < eps:
        action = np.random.randint(-lot_size, lot_size, dtype=np.int64)
    else:
        action = maxAction(q_value, state, lot_size, optimizers, optimizer)

    for t in range(1, t_):

        # Observe s'
        state_ = [state[0] + action, f[j, t]]

        # Choose a' from s' using policy derived from q_value

        if np.random.rand() < eps:
            action_ = np.random.randint(-lot_size, lot_size, dtype=np.int64)
        else:
            action_ = maxAction(q_value, state_, lot_size, optimizers, optimizer)

        # Observe r

        reward_t = reward(state[0], state_[0], f[j, t], Lambda, B, mu_u, Sigma,
                          rho, gamma)
        reward_total += reward_t
        cost_total += -0.5*((state_[0]-state[0])*Lambda*(state_[0]-state[0]) +
                            (1 - rho)*gamma*state_[0]*Sigma*state_[0])

        # Update value function
        y = q_value(state, action) +\
            alpha*(reward_t +
                   (1-rho)*q_value(state_, action_) -
                   q_value(state, action))

        # update pairs
        x_episode[t-1] = np.r_[state, action]
        y_episode[t-1] = y

        # update state and action
        state = state_
        action = action_

    return x_episode, y_episode, j, reward_total, cost_total


# -----------------------------------------------------------------------------
# q_hat
# -----------------------------------------------------------------------------

def q_hat(state, action,
          B, qb_list,
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

def compute_markovitz(f, gamma, B, Sigma):

    if f.ndim == 1:
        t_ = f.shape[0]
        j_ = 1
        f = f.reshape((j_, t_))
    elif f.ndim == 2:
        j_, t_ = f.shape

    Markovitz = np.zeros((j_, t_))
    for t in range(t_):
        Markovitz[:, t] = (gamma*Sigma)**(-1)*B*f[:, t]

    return Markovitz.squeeze()


# -----------------------------------------------------------------------------
# compute_optimal
# -----------------------------------------------------------------------------

def compute_optimal(f, gamma, lam, rho, B, Sigma, Phi):

    if f.ndim == 1:
        t_ = f.shape[0]
        j_ = 1
        f = f.reshape((j_, t_))
    elif f.ndim == 2:
        j_, t_ = f.shape

    a = (-(gamma*(1 - rho) + lam*rho) +
         np.sqrt((gamma*(1-rho) + lam*rho)**2 +
                 4*gamma*lam*(1-rho)**2)) / (2*(1-rho))

    x = np.zeros((j_, t_))
    for t in range(t_):
        if t == 0:
            x[:, t] = a/lam * 1/(gamma*Sigma) * (B/(1+Phi*a/gamma))*f[:, t]

        else:
            x[:, t] = (1 - a/lam)*x[:, t-1] +\
                a/lam * 1/(gamma*Sigma) * (B/(1+Phi*a/gamma))*f[:, t]

    return x.squeeze()


# -----------------------------------------------------------------------------
# compute_rl
# -----------------------------------------------------------------------------

def compute_rl(j, f, q_value, lot_size, optimizers, optimizer=None):

    if f.ndim ==1:
        t_ = f.shape[0]
    else:
        f = f[j, :]
        t_ = f.shape[0]

    shares = np.zeros(t_)

    for t in range(t_):

        if t == 0:
            state = np.array([0, f[t]])
            action = maxAction(q_value, state, lot_size, optimizers,
                               optimizer=optimizer)
            shares[t] = state[0] + action
        else:
            state = np.array([shares[t-1], f[t]])
            action = maxAction(q_value, state, lot_size, optimizers,
                               optimizer=optimizer)
            shares[t] = state[0] + action

    return shares


# -----------------------------------------------------------------------------
# compute_wealth
# -----------------------------------------------------------------------------

def compute_wealth(r, strat, gamma, Lambda, rho, B, Sigma, Phi):

    if r.ndim == 1:
        t_ = r.shape[0]
        j_ = 1
        r = r.reshape((j_, t_))
        strat = strat.reshape((j_, t_))
    elif r.ndim == 2:
        j_, t_ = r.shape

    # Value
    value = np.zeros((j_, t_))
    for t in range(t_ - 1):
        value[:, t] = (1 - rho)**(t + 1) * strat[:, t] * r[:, t+1]
    value = np.cumsum(value, axis=1)

    # Costs
    cost = np.zeros((j_, t_))
    for t in range(1, t_):
        delta_strat = strat[:, t] - strat[:, t-1]
        cost[:, t] =\
            gamma/2 * (1 - rho)**(t + 1) * strat[:, t]*Sigma*strat[:, t] +\
            0.5*(1 - rho)**t * delta_strat*Lambda*delta_strat
    cost = np.cumsum(cost, axis=1)

    # Wealth
    wealth = value - cost

    return wealth.squeeze(), value.squeeze(), cost.squeeze()
