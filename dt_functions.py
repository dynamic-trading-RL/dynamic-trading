# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:16:57 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from scipy.optimize import (dual_annealing, shgo, differential_evolution,
                            brute, minimize)
from scipy.stats import multivariate_normal
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from enums import RiskDriverDynamicsType, FactorDynamicsType, FactorType


# -----------------------------------------------------------------------------
# generate_episode
# -----------------------------------------------------------------------------

def generate_episode(
                     # dummy for parallel computing
                     j,
                     # market simulations
                     price, pnl, f, market, factorType,
                     # reward/cost parameters
                     rho, gamma, Sigma, Lambda, return_is_pnl,
                     # RL parameters
                     eps, qb_list, flag_qaverage, alpha,
                     optimizers, optimizer,
                     b,
                     bound=400,
                     rescale_n_a=True, predict_r=True,
                     dyn_update_q_value=True, random_act_batch0=False):
    """
    Given a market simulation, this function generates an episode for the
    reinforcement learning agent training
    """

    if rescale_n_a:
        resc_n_a = bound
    else:
        resc_n_a = 1.

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

    if dyn_update_q_value:
        anchor_points = np.zeros(t_-1)
        dist_list = []

    # Define initial value function:
    # If b=0, it is set = 0. If b>0, it averages on supervised_regressors fitted on previous
    # batches
    q_value = get_q_value(b, qb_list, flag_qaverage)
    q_value_iter = q_value

    # Observe state
    if factorType == FactorType.Observable:
        state = [0., f[j, 0]]
    elif factorType == FactorType.Latent:
        state = [0.]
    else:
        raise NameError('Invalid factorType: ' + factorType.value)

    # Choose action

    lb = -bound / resc_n_a
    ub = bound / resc_n_a
    if np.random.rand() < eps:
        action = lb + (ub - lb)*np.random.rand()
    else:
        if random_act_batch0:
            action = maxAction(q_value, state, [lb, ub], b,
                               optimizers, optimizer)
        else:
            action = maxAction(q_value, state, [lb, ub], 1,
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
        lb = -bound / resc_n_a - state_[0]
        ub = bound / resc_n_a - state_[0]
        if np.random.rand() < eps:
            action_ = lb + (ub - lb)*np.random.rand()
        else:
            if random_act_batch0:
                action_ = maxAction(q_value_iter, state_, [lb, ub], b,
                                    optimizers, optimizer)
            else:
                action_ = maxAction(q_value_iter, state_, [lb, ub], 1,
                                    optimizers, optimizer)

        # Observe reward
        n = state_[0] * resc_n_a
        if predict_r:

            if return_is_pnl:
                r = market.next_step(f[j, t-1])
            else:
                r = price[j, t-1]*market.next_step(f[j, t-1])
        else:
            r = pnl[j, t]

        Sigma_t, Lambda_t = get_rescSigmaLambda(return_is_pnl, Sigma, Lambda,
                                                price[j, t-1])

        dn = action * resc_n_a

        cost_t = cost(n, dn, rho, gamma, Sigma_t, Lambda_t)
        reward_t = reward(n, r, cost_t, rho)

        cost_total += cost_t
        reward_total += reward_t

        # Update value function
        y = q_value_iter(state, action) +\
            alpha*(reward_t +
                   (1 - rho)*q_value_iter(state_, action_) -
                   q_value_iter(state, action))

        # Update fitting pairs
        x_episode[t-1] = np.r_[state, action]
        y_episode[t-1] = y

        # Update state and action
        state = state_
        action = action_

        if dyn_update_q_value:
            # update q_value by imposing that q_value(x_episode) = y_episode
            q_value_iter = get_q_value_iter(q_value, anchor_points, x_episode,
                                            y_episode, t, dist_list)

    return x_episode, y_episode, j, reward_total, cost_total


# -----------------------------------------------------------------------------
# reward
# -----------------------------------------------------------------------------

def reward(n, r, cost_t, rho):

    return (1 - rho)*n*r - cost_t


# -----------------------------------------------------------------------------
# cost
# -----------------------------------------------------------------------------

def cost(n, dn, rho, gamma, Sigma, Lambda):

    return 0.5*((1 - rho)*gamma*n*Sigma*n + dn*Lambda*dn)


# -----------------------------------------------------------------------------
# maxAction
# -----------------------------------------------------------------------------

def maxAction(q_value, state, bounds, b, optimizers, optimizer=None):
    """
    This function determines the q-greedy action for a given
    q-value function and state
    """

    if b == 0:

        return bounds[0] + (bounds[1] - bounds[0])*np.random.rand()

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
                         Ns=200,
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
                            Ns=200,
                            finish=None)

            return x_brute

        elif optimizer == 'local':
            optimizers._local += 1
            res = minimize(fun, x0=np.array([0]), bounds=bounds)

        else:
            raise NameError('Wrong optimizer: ' + optimizer)

        return res.x[0]


# -----------------------------------------------------------------------------
# set_regressor_parameters_ann
# -----------------------------------------------------------------------------

def set_regressor_parameters_ann(sup_model):

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
        alpha_ann = 0.0001

        return hidden_layer_sizes, max_iter, n_iter_no_change, alpha_ann

    elif sup_model == 'ann_small':
        hidden_layer_sizes = (10,)
        max_iter = 50
        n_iter_no_change = 10
        alpha_ann = 0.0001

        return hidden_layer_sizes, max_iter, n_iter_no_change, alpha_ann

    else:

        raise NameError('sup_model must be either ann_fast or ann_deep or '
                        + 'ann_small')


# -----------------------------------------------------------------------------
# set_regressor_parameters_tree
# -----------------------------------------------------------------------------

def set_regressor_parameters_tree():
    n_estimators = 80
    min_samples_split = 0.01
    max_samples = 0.9
    warm_start = True
    verbose = 0

    return n_estimators, min_samples_split, max_samples, warm_start, verbose


def set_regressor_parameters_gb():

    n_estimators, min_samples_split, _, warm_start, verbose =\
        set_regressor_parameters_tree()

    n_estimators = n_estimators
    learning_rate = 10/n_estimators
    subsample = 0.8
    min_samples_split = min_samples_split
    warm_start = warm_start
    n_iter_no_change = 100
    verbose = verbose

    return learning_rate, n_estimators, subsample, min_samples_split,\
        warm_start, n_iter_no_change, verbose


# -----------------------------------------------------------------------------
# q_hat
# -----------------------------------------------------------------------------

def q_hat(state, action,
          qb_list,
          flag_qaverage=True, n_models=None):
    """
    This function evaluates the estimated q-value function in a given state and
    action pair. The other parameters are given to include the cases of
    model averaging and data/data_tmp rescaling.
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

def compute_markovitz(f, gamma, B, Sigma, price, mu_r, return_is_pnl):

    if type(f) in (pd.core.series.Series, pd.core.frame.DataFrame):
        f = f.to_numpy()

    if type(price) in (pd.core.series.Series, pd.core.frame.DataFrame):
        price = price.to_numpy()

    if f.ndim == 1:
        t_ = f.shape[0]
        j_ = 1
        f = f.reshape((j_, t_))
        price = price.reshape((j_, t_))
    elif f.ndim == 2:
        j_, t_ = f.shape

    Markovitz = np.zeros((j_, t_))
    for t in range(t_):

        if return_is_pnl:
            resc_f = f[:, t] + mu_r/B
            resc_Sigma = Sigma

        else:
            resc_f = price[:, t]*(f[:, t] + mu_r/B)
            resc_Sigma = price[:, t]**2 * Sigma

        Markovitz[:, t] = (gamma*resc_Sigma)**(-1)*B*resc_f

    return Markovitz.squeeze()


# -----------------------------------------------------------------------------
# compute_GP
# -----------------------------------------------------------------------------

def compute_GP(f, gamma, lam, rho, B, Sigma, Phi, price, mu_r, return_is_pnl):

    if type(f) in (pd.core.series.Series, pd.core.frame.DataFrame):
        f = f.to_numpy()

    if type(price) in (pd.core.series.Series, pd.core.frame.DataFrame):
        price = price.to_numpy()

    if f.ndim == 1:
        t_ = f.shape[0]
        j_ = 1
        f = f.reshape((j_, t_))
        price = price.reshape((j_, t_))
    elif f.ndim == 2:
        j_, t_ = f.shape

    x = np.zeros((j_, t_))
    for t in range(t_):

        if return_is_pnl:
            resc_f = f[:, t] + mu_r/B
            resc_Sigma = Sigma

        else:
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

def compute_rl(j, f, qb_list, factorType, optimizers, optimizer=None,
               bound=400, rescale_n_a=True, GP=None):

    if type(f) in (pd.core.series.Series, pd.core.frame.DataFrame):
        f = f.to_numpy()

    if rescale_n_a:
        resc_n_a = bound
    else:
        resc_n_a = 1.

    # if f.ndim == 1:
    #     t_ = f.shape[0]
    # else:
    #     t_ = f.shape[1]
    if f.ndim == 1:
        t_ = f.shape[0]
        f = f.reshape((1, t_))
    elif f.ndim == 2:
        _, t_ = f.shape

    shares = np.zeros(t_)
    q_value = get_q_value(1, qb_list, flag_qaverage=True)

    for t in range(t_):

        if t == 0:

            if factorType == FactorType.Observable:
                state = np.array([0, f[j, t]])
            elif factorType == FactorType.Latent:
                state = np.array([0])
            else:
                raise NameError('Invalid factorType: ' + factorType.value)

            lb = -bound / resc_n_a - state[0]
            ub = bound / resc_n_a - state[0]
            action = maxAction(q_value, state, [lb, ub], 1, optimizers,
                               optimizer=optimizer)
            shares[t] = (state[0] + action) * resc_n_a
        else:

            if factorType == FactorType.Observable:
                state = np.array([shares[t-1] / resc_n_a, f[j, t]])
            elif factorType == FactorType.Latent:
                state = np.array([shares[t-1] / resc_n_a])
            else:
                raise NameError('Invalid factorType: ' + factorType.value)

            lb = -bound / resc_n_a - state[0]
            ub = bound / resc_n_a - state[0]
            action = maxAction(q_value, state, [lb, ub], 1, optimizers,
                               optimizer=optimizer)
            shares[t] = (state[0] + action) * resc_n_a

    return shares


# -----------------------------------------------------------------------------
# compute_wealth
# -----------------------------------------------------------------------------

def compute_wealth(pnl, strat, gamma, Lambda, rho, Sigma, price,
                   return_is_pnl):

    if type(pnl) in (pd.core.series.Series, pd.core.frame.DataFrame):
        pnl = pnl.to_numpy()

    if type(price) in (pd.core.series.Series, pd.core.frame.DataFrame):
        price = price.to_numpy()

    if pnl.ndim == 1:
        t_ = pnl.shape[0]
        j_ = 1
        pnl = pnl.reshape((j_, t_))
        strat = strat.reshape((j_, t_))
        price = price.reshape((j_, t_))
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

        resc_Sigma, resc_Lambda = get_rescSigmaLambda(return_is_pnl, Sigma,
                                                      Lambda, price[:, t])

        cost[:, t] =\
            gamma/2 * (1 - rho)**(t + 1) * strat[:, t]*resc_Sigma*strat[:, t] +\
            0.5*(1 - rho)**t * delta_strat*resc_Lambda*delta_strat
    cost = np.cumsum(cost, axis=1)

    # Wealth
    wealth = value - cost

    return wealth.squeeze(), value.squeeze(), cost.squeeze()


# -----------------------------------------------------------------------------
# get_q_value
# -----------------------------------------------------------------------------

def get_q_value(b, qb_list, flag_qaverage):

    if b == 0:  # initialize q_value arbitrarily

        def q_value(state, action):

            return np.random.randn()

    else:  # average supervised_regressors across previous batches

        def q_value(state, action):
            return q_hat(state, action, qb_list,
                         flag_qaverage=flag_qaverage)

    return q_value


# -----------------------------------------------------------------------------
# get_q_value_iter
# -----------------------------------------------------------------------------

def get_q_value_iter(q_value, anchor_points, x_episode, y_episode, t,
                     dist_list):

    if t == 1:
        h = 1
    else:
        x_last = x_episode[t-1]
        for i in range(t-1):
            dist_list.append(np.linalg.norm(x_last - x_episode[i]))

        dist = min(dist_list)/2
        c = 0.01
        h = -dist**2/(2*np.log(c))

    def q_value_iter(state, action):

        cov = h*np.eye(len(x_episode[0]))

        anchor_points[t-1] = y_episode[t-1] - q_value(x_episode[t-1][:-1],
                                                      x_episode[t-1][-1])

        moll = multivariate_normal.pdf(x=x_episode,
                                       mean=np.r_[state, action],
                                       cov=cov)

        moll *= np.sqrt(np.linalg.det(2*np.pi*cov))

        return q_value(state, action) + anchor_points@moll

    return q_value_iter


# -----------------------------------------------------------------------------
# get_dynamics_params
# -----------------------------------------------------------------------------

def get_dynamics_params(market):

    if (market.marketDynamics.riskDriverDynamics.riskDriverDynamicsType
            == RiskDriverDynamicsType.Linear):
        B = market.marketDynamics.riskDriverDynamics.parameters['B']
        mu_r = market.marketDynamics.riskDriverDynamics.parameters['mu']
    else:
        B_0 = market.marketDynamics.riskDriverDynamics.parameters['B_0']
        B_1 = market.marketDynamics.riskDriverDynamics.parameters['B_1']
        B = 0.5*(B_0 + B_1)
        mu_0 = market.marketDynamics.riskDriverDynamics.parameters['mu_0']
        mu_1 = market.marketDynamics.riskDriverDynamics.parameters['mu_1']
        mu_r = 0.5*(mu_0 + mu_1)

    if (market.marketDynamics.factorDynamics.factorDynamicsType
            in (FactorDynamicsType.AR, FactorDynamicsType.AR_TARCH)):

        Phi = 1 - market.marketDynamics.factorDynamics.parameters['B']
        mu_f = 1 - market.marketDynamics.factorDynamics.parameters['mu']

    elif (market.marketDynamics.factorDynamics.factorDynamicsType
          == FactorDynamicsType.SETAR):

        Phi_0 = 1 - market.marketDynamics.factorDynamics.parameters['B_0']
        Phi_1 = 1 - market.marketDynamics.factorDynamics.parameters['B_1']
        mu_f_0 = 1 - market.marketDynamics.factorDynamics.parameters['mu_0']
        mu_f_1 = 1 - market.marketDynamics.factorDynamics.parameters['mu_1']
        Phi = 0.5*(Phi_0 + Phi_1)
        mu_f = .5*(mu_f_0 + mu_f_1)

    else:

        Phi = 0.
        mu_f = market.marketDynamics.factorDynamics.parameters['mu']

    return B, mu_r, Phi, mu_f


# -----------------------------------------------------------------------------
# get_bound
# -----------------------------------------------------------------------------

def get_bound(return_is_pnl, f, price, gamma, lam, rho, B, mu_r,
              Sigma, Phi, resc_by_M=True):

    if resc_by_M:
        Markowitz = compute_markovitz(f.flatten(), gamma, B, Sigma,
                                      price.flatten(), mu_r, return_is_pnl)
        bound = np.percentile(np.abs(Markowitz), 95)
    else:
        GP = compute_GP(f.flatten(), gamma, lam, rho, B, Sigma, Phi,
                        price.flatten(), mu_r, return_is_pnl)
        bound = np.percentile(np.abs(GP), 95)

    return bound


# -----------------------------------------------------------------------------
# get_rescSigmaLambda
# -----------------------------------------------------------------------------

def get_rescSigmaLambda(return_is_pnl, Sigma, Lambda, price):

    if return_is_pnl:
        rescSigma = Sigma
        rescLambda = Lambda

    else:
        rescSigma = price**2*Sigma
        rescLambda = price**2*Lambda

    return rescSigma, rescLambda


# -----------------------------------------------------------------------------
# perform_ttest
# -----------------------------------------------------------------------------

def perform_ttest(final_wealth_GP, final_wealth_M, final_wealth_RL,
                  final_value_GP, final_value_M, final_value_RL,
                  final_cost_GP, final_cost_M, final_cost_RL):

    # General statistics
    df = pd.DataFrame(columns=['Quantity', 'Model', 'Mean', 'Standard deviation'],
                      data=[['Final wealth', 'GP', np.mean(final_wealth_GP), np.std(final_wealth_GP)],
                            ['Final wealth', 'M', np.mean(final_wealth_M), np.std(final_wealth_M)],
                            ['Final wealth', 'RL', np.mean(final_wealth_RL), np.std(final_wealth_RL)],
                            ['Final value', 'GP', np.mean(final_value_GP), np.std(final_value_GP)],
                            ['Final value', 'M', np.mean(final_value_M), np.std(final_value_M)],
                            ['Final value', 'RL', np.mean(final_value_RL), np.std(final_value_RL)],
                            ['Final cost', 'GP', np.mean(final_cost_GP), np.std(final_cost_GP)],
                            ['Final cost', 'M', np.mean(final_cost_M), np.std(final_cost_M)],
                            ['Final cost', 'RL', np.mean(final_cost_RL), np.std(final_cost_RL)]])

    df.to_csv('reports/general_statistics.csv', index=False)

    # Execute tests
    t_wealth_GP_M = ttest_ind(final_wealth_GP,
                              final_wealth_M,
                              usevar='unequal',
                              alternative='larger')

    t_wealth_RL_M = ttest_ind(final_wealth_RL,
                              final_wealth_M,
                              usevar='unequal',
                              alternative='larger')

    t_wealth_RL_GP = ttest_ind(final_wealth_RL,
                               final_wealth_GP,
                               usevar='unequal',
                               alternative='larger')

    t_value_GP_M = ttest_ind(final_value_GP,
                             final_value_M,
                             usevar='unequal',
                             alternative='larger')

    t_value_RL_M = ttest_ind(final_value_RL,
                             final_value_M,
                             usevar='unequal',
                             alternative='larger')

    t_value_RL_GP = ttest_ind(final_value_RL,
                              final_value_GP,
                              usevar='unequal',
                              alternative='larger')

    t_cost_GP_M = ttest_ind(final_cost_GP,
                            final_cost_M,
                            usevar='unequal',
                            alternative='smaller')

    t_cost_RL_M = ttest_ind(final_cost_RL,
                            final_cost_M,
                            usevar='unequal',
                            alternative='smaller')

    t_cost_RL_GP = ttest_ind(final_cost_RL,
                             final_cost_GP,
                             usevar='unequal',
                             alternative='smaller')

    # Prepare output dataframe
    df = pd.DataFrame(columns=['H0', 'H1', 't-statistic', 'p-value'],
                      data=[['GPw=Mw', 'GPw>Mw', t_wealth_GP_M[0],
                             t_wealth_GP_M[1]],
                             ['RLw=Mw', 'RLw>Mw', t_wealth_RL_M[0],
                              t_wealth_RL_M[1]],
                             ['RLw=GPw', 'RLw>GPw', t_wealth_RL_GP[0],
                              t_wealth_RL_GP[1]],
                             ['GPv=Mv', 'GPv>Mv', t_value_GP_M[0],
                              t_value_GP_M[1]],
                             ['RLv=Mv', 'RLv>Mv', t_value_RL_M[0],
                              t_value_RL_M[1]],
                             ['RLv=GPv', 'RLv>GPv', t_value_RL_GP[0],
                              t_value_RL_GP[1]],
                             ['GPc=Mc', 'GPc<Mc', t_cost_GP_M[0],
                              t_cost_GP_M[1]],
                             ['RLc=Mc', 'RLc<Mc', t_cost_RL_M[0],
                              t_cost_RL_M[1]],
                             ['RLc=GPc', 'RLc<GPc', t_cost_RL_GP[0],
                              t_cost_RL_GP[1]]])

    # Output
    df.to_csv('reports/t_tests.csv', index=False)

    # Print results
    print('\n\n\nWelch\'s tests (absolute):\n')
    print('    H0: GPw=Mw, H1: GPw>Mw. t: %.4f, p-value: %.4f' %
          (t_wealth_GP_M[0], t_wealth_GP_M[1]))
    print('    H0: RLw=Mw, H1: RLw>Mw. t: %.4f, p-value: %.4f' %
          (t_wealth_RL_M[0], t_wealth_RL_M[1]))
    print('    H0: RLw=GPw, H1: RLw>GPw. t: %.4f, p-value: %.4f' %
          (t_wealth_RL_GP[0], t_wealth_RL_GP[1]))

    print('\n    H0: GPv=Mv, H1: GPv>Mv. t: %.4f, p-value: %.4f' %
          (t_value_GP_M[0], t_value_GP_M[1]))
    print('    H0: RLv=Mv, H1: RLv>Mv. t: %.4f, p-value: %.4f' %
          (t_value_RL_M[0], t_value_RL_M[1]))
    print('    H0: RLv=GPv, H1: RLv>GPv. t: %.4f, p-value: %.4f' %
          (t_value_RL_GP[0], t_value_RL_GP[1]))

    print('\n    H0: GPc=Mc, H1: GPc<Mc. t: %.4f, p-value: %.4f' %
          (t_cost_GP_M[0], t_cost_GP_M[1]))
    print('    H0: RLc=Mc, H1: RLc<Mc. t: %.4f, p-value: %.4f' %
          (t_cost_RL_M[0], t_cost_RL_M[1]))
    print('    H0: RLc=GPc, H1: RLc<GPc. t: %.4f, p-value: %.4f' %
          (t_cost_RL_GP[0], t_cost_RL_GP[1]))


# -----------------------------------------------------------------------------
# perform_linear_regression
# -----------------------------------------------------------------------------

def perform_linear_regression(final_wealth_RL, final_wealth_GP):

    linreg_GP_RL = OLS(final_wealth_RL, add_constant(final_wealth_GP)).fit()

    print('\n\n\n Linear regression of X=GP vs Y=RL:\n')
    print(linreg_GP_RL.summary())
    print('\n\n H0: beta=1; H1: beta!=1')
    print(linreg_GP_RL.t_test(([0., 1.], 1.)).summary())

    with open('reports/final_linear_regression_results.txt', 'w+') as fh:
        fh.write(linreg_GP_RL.summary().as_text())
    with open('reports/final_linear_regression_ttest.txt', 'w+') as fh:
        fh.write(linreg_GP_RL.t_test(([0., 1.], 1.)).summary().as_text())

    return linreg_GP_RL