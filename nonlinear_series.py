# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 09:59:53 2021

@author: Giorgi
"""

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from joblib import load, dump
import multiprocessing as mp
from functools import partial
from dt_functions import (simulate_market, q_hat, generate_episode, Optimizers,
                          compute_markovitz, compute_optimal, compute_rl,
                          compute_wealth)


# ------------------------------------- Parameters ----------------------------

parallel_computing = False       # True for parallel computing
n_cores_max = 80                # maximum number of cores if parallel_computing
n_batches = 5                   # number of batches
eps = 0.1                       # eps greedy
alpha = 1                       # learning rate
t_ = 50
j_ = 10000                      # number of episodes
optimizer = None
nonlinear = True

# RL model
sup_model = 'ann_fast'  # or random_forest or ann_deep

Phi = 0.23
mu_eps = 0.
Omega = 0.13

# Real returns dynamics
B1 = 0.09
B2 = -0.12
B3 = -0.06
mu_u_pol = 0.04
B_list = [mu_u_pol, B1, B2, B3]
sig_pol = 1.71

lam = 10**-2
Lambda = lam*sig_pol  # Lambda is the true cost multiplier
gamma = 10**-3
rho = 1-np.exp(-0.02/260)


# ------------------------------------- Mkt simulations -----------------------

# Assume this is the TRUE market, i.e. driven by parameters B_list and sig_pol

r, f = simulate_market(j_, t_, n_batches, 0, 0, 0, Phi, mu_eps, Omega,
                       nonlinear=True, # if True, the parameters below are used
                       nonlineartype='polynomial',  # can be 'nn' or 'polynomial'
                       nn=None, sig_nn=None,  # nn parameters
                       B_list=B_list, sig_pol=sig_pol  # polynomial parameters
                       )

# ??? we can add stochastic volatility keeping a linear model:
# r_{t+1} = mu_u + B*f_{t} + sig_{t+1}*u_{t+1}
# ln(sig_{t+1}) = alpha + beta*ln(sig_{t}) + eta_{t+1}


# ------------------------------------- Fit linear model ----------------------

# A trader following Garleanu-Pedersen would use these fitted parameters

# Fit linear model for the returns
reg = LinearRegression().fit(X=f[:, :, :-1].flatten().reshape(-1, 1),
                             y=r[:, :, 1:].flatten().reshape(-1, 1))

B = reg.coef_[0, 0]
mu_u = reg.intercept_[0]
Sigma = (r[:, :, 1:].flatten() -
         reg.predict(f[:, :, :-1].flatten().reshape(-1, 1)).flatten()).var()


# ------------------------------------- Fit polynomial model ------------------

# A trader following RL would use these fitted parameters

# Fit non-linear model for the returns

poly = PolynomialFeatures(3, include_bias=False)
X = poly.fit_transform(f[:, :, :-1].flatten().reshape(-1, 1))

reg_pol = LinearRegression().fit(X=X,
                                 y=r[:, :, 1:].flatten().reshape(-1, 1))

B_list_fitted = [reg_pol.intercept_[0]] + list(reg_pol.coef_[0])

sig_pol_fitted = (r[:, :, 1:].flatten().reshape(-1, 1) -
                  reg_pol.predict(X)).var()


# ------------------------------------- Printing ------------------------------

print('######## Training RL agent')

if parallel_computing:
    print('Number of cores available: %d' % mp.cpu_count())
    n_cores = min(mp.cpu_count(), n_cores_max)
    print('Number of cores used: %d' % n_cores)


if sup_model == 'random_forest':
    from sklearn.ensemble import RandomForestRegressor
elif sup_model == 'ann_fast':
    from sklearn.neural_network import MLPRegressor
    hidden_layer_sizes = (64, 32, 8)
    max_iter = 10
    n_iter_no_change = 2
    alpha_ann = 0.0001
elif sup_model == 'ann_deep':
    from sklearn.neural_network import MLPRegressor
    hidden_layer_sizes = (70, 50, 30, 10)
    max_iter = 200
    n_iter_no_change = 10
    alpha_ann = 0.001


# ------------------------------------- Reinforcement learning ----------------

# The trader using RL simulates the market according to B_list_fitted and
# sig_pol_fitted

r, f = simulate_market(j_, t_, n_batches, 0, 0, 0, Phi, mu_eps, Omega,
                       nonlinear=True,
                       nonlineartype='polynomial',
                       nn=None, sig_nn=None,
                       B_list=B_list_fitted, sig_pol=sig_pol_fitted
                       )

# used only to determine decent bounds for the RL optimization
Markovitz = compute_markovitz(f[:, 0, :].flatten(), gamma, B, Sigma)
lot_size = np.max(np.abs(np.diff(Markovitz)))
print('lot_size =', lot_size)

qb_list = []  # list to store models

optimizers = Optimizers()


def next_step(f_t):

    f = np.array([f_t, f_t**2, f_t**3]).reshape(1, -1)

    return reg_pol.predict(f)


for b in range(n_batches):  # loop on batches
    print('Creating batch %d of %d; eps=%f' % (b+1, n_batches, eps))
    X = []  # simulations
    Y = []
    j_sort = []
    reward_sort = []
    cost_sort = []

    # definition of value function:
    if b == 0:  # initialize q_value arbitrarily

        def q_value(state, action):
            return np.random.randn()

    else:  # average models across previous batches

        qb_list.append(load('models/q%d.joblib' % (b-1)))  # import regressors

        def q_value(state, action):
            return q_hat(state, action, qb_list,
                         flag_qaverage=False,
                         n_models=None)

    # generate episodes
    # create alias for generate_episode that fixes all the parameters but j
    # this way we can iterate it via multiprocessing.Pool.map()

    gen_ep_part = partial(generate_episode,
                          # market parameters
                          Lambda=Lambda, next_step=next_step,
                          sig=sig_pol_fitted,
                          # market simulations
                          f=f[:, b, :],
                          # RL parameters
                          eps=eps, rho=rho, q_value=q_value, alpha=alpha,
                          gamma=gamma, lot_size=lot_size,
                          optimizers=optimizers,
                          optimizer=optimizer)

    if parallel_computing:
        if __name__ == '__main__':
            p = mp.Pool(n_cores)
            episodes = p.map(gen_ep_part, range(j_))
            p.close()
            p.join()
        # unpack episodes into arrays
        for j in range(len(episodes)):
            X.append(episodes[j][0])
            Y.append(episodes[j][1])
            j_sort.append(episodes[j][2])
            reward_sort.append(episodes[j][3])
            cost_sort.append(episodes[j][4])

    else:
        for j in range(j_):
            print('Computing episode '+str(j+1)+' on '+str(j_))
            episodes = gen_ep_part(j)
            X.append(episodes[0])
            Y.append(episodes[1])
            j_sort.append(episodes[2])
            reward_sort.append(episodes[3])
            cost_sort.append(episodes[4])

    X = np.array(X).reshape((j_*(t_-1), 3))
    Y = np.array(Y).reshape((j_*(t_-1)))

    ind_sort = np.argsort(j_sort)
    j_sort = np.sort(j_sort)
    reward = np.array(reward_sort)[ind_sort]
    cost = np.array(cost_sort)[ind_sort]

    # used as ylim in plots below
    if b == 0:
        min_Y = np.min(Y)
        max_Y = np.max(Y)
    else:
        min_Y = min(np.min(Y), min_Y)
        max_Y = max(np.max(Y), max_Y)

    print('Fitting model %d of %d' % (b+1, n_batches))
    if sup_model == 'random_forest':
        model = RandomForestRegressor(n_estimators=20, max_features=0.333,
                                      min_samples_split=0.01,
                                      max_samples=0.9,
                                      oob_score=True,
                                      n_jobs=1,
                                      verbose=0,
                                      warm_start=True)
    elif sup_model == 'ann_fast' or sup_model == 'ann_deep':
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             alpha=alpha_ann,
                             max_iter=max_iter,
                             n_iter_no_change=n_iter_no_change
                             )

    dump(model.fit(X, Y), 'models/q%d.joblib' % b)  # export regressor
    print('    Score: %.3f' % model.score(X, Y))
    print('    Average reward: %.3f' % np.mean(reward))
    print(optimizers)

    eps = max(eps/3, 0.00001)  # update epsilon


# ------------------------------------- Dump data -----------------------------

dump(lot_size, 'data/lot_size.joblib')
dump(n_batches, 'data/n_batches.joblib')
dump(optimizers, 'data/optimizers.joblib')


# ------------------------------------- Out of sample -------------------------

print('######## Out of sample')

j_oos = 100  # number of out-of-sample paths


# ------------------------------------- Simulate ------------------------------

# To test out-of-sample, the trader uses the polynomial fit. However, we
# compute the Markovitz and GP strategies using the linear fit

# Simulate market
r, f = simulate_market(j_oos, t_, 1, 0, 0, 0, Phi, mu_eps, Omega,
                       nonlinear=True, # if True, the parameters below are used
                       nonlineartype='polynomial',  # can be 'nn' or 'polynomial'
                       nn=None, sig_nn=None,  # nn parameters
                       B_list=B_list_fitted, sig_pol=sig_pol_fitted  # polynomial parameters
                       )


# Markovitz portfolio
print('#### Computing Markovitz strategy')
Markovitz = compute_markovitz(f, gamma, B, Sigma)


# Optimal portfolio
print('#### Computing optimal strategy')
x = compute_optimal(f, gamma, Lambda, rho, B, Sigma, Phi)


# RL portfolio
print('##### Computing RL strategy')

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, qb_list, flag_qaverage=False, n_models=None)


if parallel_computing:
    if __name__ == '__main__':

        compute_rl_part = partial(compute_rl, f=f, q_value=q_value,
                                  lot_size=lot_size, optimizers=optimizers,
                                  optimizer=optimizer)

        p = mp.Pool(n_cores)
        shares = p.map(compute_rl_part, range(j_oos))
        p.close()
        p.join()
    shares = np.array(shares)

else:
    shares = np.zeros((j_oos, t_))
    for j in range(j_oos):
        print('Simulation', j+1, 'on', j_oos)
        shares[j, :] = compute_rl(j, f, q_value, lot_size, optimizers,
                                  optimizer=optimizer)

# Wealth
# the wealth must be computed according to the simulated market
wealth_opt, value_opt, cost_opt = compute_wealth(r, x, gamma, Lambda, rho,
                                                 sig_pol_fitted)

wealth_m, value_m, cost_m = compute_wealth(r, Markovitz, gamma, Lambda, rho,
                                           sig_pol_fitted)

wealth_rl, value_rl, cost_rl = compute_wealth(r, shares, gamma, Lambda, rho,
                                              sig_pol_fitted)


# ------------------------------------- Plots ---------------------------------

plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

plt.hist(wealth_m[:, -1], 90, label='Markovitz', density=True, alpha=0.5)
plt.hist(wealth_rl[:, -1], 90, label='RL', density=True, alpha=0.5)
plt.hist(wealth_opt[:, -1], 90, label='Optimal', density=True, alpha=0.5)

results_str = 'Markovitz (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_m[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_m[:, -1])) + ') \n' +\
    'RL (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_rl[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_rl[:, -1])) + ')\n' +\
    'Optimal (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_opt[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_opt[:, -1])) + ')'

plt.annotate(results_str, xy=(0, 1), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points')
plt.legend(loc='upper right')
plt.title('Total wealth')

plt.savefig('figures/out-of-sample.png')


plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.scatter(wealth_opt[:, -1], wealth_rl[:, -1])
xx = [min(wealth_opt[:, -1].min(), wealth_rl[:, -1].min()),
      max(wealth_opt[:, -1].max(), wealth_rl[:, -1].max())]
plt.plot(xx, xx, color='r')

xlim = [min(np.quantile(wealth_opt[:, -1], 0.05),
            np.quantile(wealth_rl[:, -1], 0.05)),
        max(np.quantile(wealth_opt[:, -1], 0.95),
            np.quantile(wealth_rl[:, -1], 0.95))]
plt.xlim(xlim)
plt.ylim(xlim)
plt.title('GP vs RL')

plt.savefig('figures/out-of-sample-gpvsrl.png')
