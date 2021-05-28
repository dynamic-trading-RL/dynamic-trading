# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:16:57 2021

@author: Giorgi
"""

import numpy as np


# -----------------------------------------------------------------------------
# simulate_market
# -----------------------------------------------------------------------------


def simulate_market(j_, t_, n_batches,
                    df_factor,
                    B, mu_u, Sigma,
                    df_return,
                    Phi, mu_eps, Omega):

    f = np.zeros((j_, n_batches, t_))
    f[:, :, 0] = df_factor.iloc[0]
    for t in range(1, t_-1):
        f[:, :, t] =\
            mu_eps + (1 - Phi)*f[:, :, t-1] + np.sqrt(Omega)*np.random.randn(j_, n_batches)

    r = np.zeros((j_, n_batches, t_))
    r[:, :, 0] = df_return.iloc[0]
    r[:, :, 1] = df_return.iloc[1]
    r[:, :, 2:] = mu_u + B*f[:, :, 1:-1] + np.sqrt(Sigma)*np.random.randn(j_, n_batches, t_-2)

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

def maxAction(q_value, state, actions):
    """
    This function determines the q-greedy action for a given
    q-value function, state, and action space
    """

    values = [q_value(state, a) for a in actions]
    action_index = values.index(max(values))
    return actions[action_index]


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
                     eps, rho, q_value, alpha, eta, gamma
                     ):
    """
    Given a market simulation f, this function generates an episode for the
    reinforcement learning agent training
    """

    lot_size = 100  # ???

    reward_total = 0
    cost_total = 0

    t_ = f.shape[1]

    x_episode = np.zeros((t_-1, 3))
    y_episode = np.zeros(t_-1)

    # state at t=0
    state = np.array([0, f[j, 0]])

    # choose action
    actions = np.array([n for n in np.arange(0, lot_size+1, 1)])
    if np.random.rand() < eps:
        action = actions[np.random.randint(0, len(actions))]
    else:
        action = maxAction(q_value, state, actions)

    for i in range(1, t_):

        # Observe s'
        state_ = [state[0] + action, f[j, i]]

        # Choose a' from s' using policy derived from q_value
        # update actions space
        actions =\
            np.array([n for n in np.arange(-lot_size,
                                           lot_size+1,
                                           1)])

        if np.random.rand() < eps:
            action_ = actions[np.random.randint(0, len(actions))]
        else:
            action_ = maxAction(q_value, state_, actions)

        # Observe r

        reward_t = reward(state[0], state_[0], f[j, i], Lambda, B, mu_u, Sigma,
                          rho, gamma)
        reward_total += reward_t
        cost_total += -0.5*((state_[0]-state[0])*Lambda*(state_[0]-state[0]) +
                            (1 - rho)*gamma*state_[0]*Sigma*state_[0])

        # Update value function
        y = q_value(state, action) +\
            alpha*(reward_t +
                   eta*q_value(state_, action_) -
                   q_value(state, action))

        # update pairs
        x_episode[i-1] = np.r_[state, action]
        y_episode[i-1] = y

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
