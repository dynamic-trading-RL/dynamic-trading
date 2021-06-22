# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:16:57 2021

@author: Giorgi
"""

import numpy as np
from scipy.optimize import (dual_annealing, shgo, differential_evolution,
                            basinhopping)


# -----------------------------------------------------------------------------
# class: Optimizers
# -----------------------------------------------------------------------------

class Optimizers:

    def __init__(self):
        self._shgo = 0
        self._dual_annealing = 0
        self._differential_evolution = 0
        self._basinhopping = 0


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

    return r, f


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
                      optimizers._differential_evolution,
                     optimizers._basinhopping])
        i = np.argmax(n)
        if i == 0:
            optimizer = 'shgo'
        elif i == 1:
            optimizer = 'dual_annealing'
        elif i == 2:
            optimizer = 'differential_evolution'
        elif i == 3:
            optimizer = 'basinhopping'
        else:
            print('Wrong optimizer')

    if optimizer is None:

        # optimizations
        res1 = shgo(fun, bounds=[(-lot_size, lot_size)])
        res2 = dual_annealing(fun, bounds=[(-lot_size, lot_size)])
        res3 = differential_evolution(fun, bounds=[(-lot_size, lot_size)])
        res4 = basinhopping(fun, x0=0)

        res_x = np.array([res1.x, res2.x, res3.x, res4.x])
        res_fun = np.array([res1.fun, res2.fun, res3.fun, res4.fun])

        i = np.argmax(res_fun)

        if i == 0:
            optimizers._shgo += 1
        elif i == 1:
            optimizers._dual_annealing += 1
        elif i == 2:
            optimizers._differential_evolution += 1
        elif i == 3:
            optimizers._basinhopping += 1
        else:
            print('Wrong optimizer')

        return np.round(res_x[i])

    elif optimizer == 'shgo':
        res = shgo(fun, bounds=[(-lot_size, lot_size)])
        return np.round(res.x)

    elif optimizer == 'dual_annealing':
        res = dual_annealing(fun, bounds=[(-lot_size, lot_size)])
        return np.round(res.x)

    elif optimizer == 'differential_evolution':
        res = differential_evolution(fun, bounds=[(-lot_size, lot_size)])
        return np.round(res.x)

    elif optimizer == 'basinhopping':
        res = basinhopping(fun, x0=0)
        return np.round(res.x)

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
                     optimizers):
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
        action = maxAction(q_value, state, lot_size, optimizers)

    for t in range(1, t_):

        # Observe s'
        state_ = [state[0] + action, f[j, t]]

        # Choose a' from s' using policy derived from q_value

        if np.random.rand() < eps:
            action_ = np.random.randint(-lot_size, lot_size, dtype=np.int64)
        else:
            action_ = maxAction(q_value, state_, lot_size, optimizers)

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
