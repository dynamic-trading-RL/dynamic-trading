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
import matplotlib.pyplot as plt
from joblib import load, dump
import multiprocessing as mp
from functools import partial
from dt_functions import (simulate_market, q_hat, generate_episode, Optimizers,
                          compute_markovitz, compute_optimal, compute_rl,
                          compute_wealth)


# ------------------------------------- Parameters ----------------------------

df_factor = load('data/df_factor.joblib')
t_ = load('data/t_.joblib')
nn = load('data/nn.joblib')
B = 1
mu_u = load('data/mu_u.joblib')
Sigma = load('data/Sigma.joblib')
sig_nn = load('data/sig_nn.joblib')
Phi = load('data/Phi.joblib')
mu_eps = load('data/mu_eps.joblib')
Omega = load('data/Omega.joblib')
lam = load('data/lam.joblib')
Lambda = lam*Sigma
gamma = load('data/gamma.joblib')
rho = load('data/rho.joblib')


parallel_computing = True       # True for parallel computing
n_cores_max = 80                # maximum number of cores if parallel_computing
n_batches = 5                   # number of batches
eps = 0.1                       # eps greedy
alpha = 1                       # learning rate
j_ = 15000                      # number of episodes
optimizer = None
nonlinear = True
nonlineartype = 'quadratic'

# RL model
sup_model = 'ann_fast'  # or random_forest or ann_deep


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


r, f = simulate_market(j_, t_, n_batches, B, mu_u, Sigma,
                       Phi, mu_eps, Omega, nonlinear=nonlinear, nn=nn,
                       sig_nn=sig_nn, nonlineartype=nonlineartype)


# ------------------------------------- Markovitz portfolio -------------------
# used only to determine bounds for RL optimization

Markovitz = compute_markovitz(df_factor.to_numpy(), gamma, B, Sigma)

lot_size = np.max(np.abs(np.diff(Markovitz)))*1.5
print('lot_size =', lot_size)


# ------------------------------------- Reinforcement learning ----------------

qb_list = []  # list to store models

optimizers = Optimizers()

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
            return q_hat(state, action, n_batches, qb_list,
                         flag_qaverage=False,
                         n_models=None)

    # generate episodes
    # create alias for generate_episode that fixes all the parameters but j
    # this way we can iterate it via multiprocessing.Pool.map()

    gen_ep_part = partial(generate_episode,
                          # market parameters
                          Lambda=Lambda, B=B, mu_u=mu_u, Sigma=Sigma,
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


j_ = 10000  # number of out-of-sample paths
optimizer = None
parallel_computing = True  # set to True if you want to use parallel computing
n_cores_max = 80               # maximum number of cores if parallel_computing
nonlinear = True

# Import parameters from previous scripts
t_ = load('data/t_.joblib')
nn = load('data/nn.joblib')
B = load('data/B.joblib')
mu_u = load('data/mu_u.joblib')
Sigma = load('data/Sigma.joblib')
sig_nn = load('data/sig_nn.joblib')
Phi = load('data/Phi.joblib')
mu_eps = load('data/mu_eps.joblib')
Omega = load('data/Omega.joblib')
Lambda = load('data/Lambda.joblib')
lam = load('data/lam.joblib')
gamma = load('data/gamma.joblib')
rho = load('data/rho.joblib')
n_batches = load('data/n_batches.joblib')
lot_size = load('data/lot_size.joblib')
optimizers = load('data/optimizers.joblib')

if parallel_computing:
    print('Number of cores available: %d' % mp.cpu_count())
    n_cores = min(mp.cpu_count(), n_cores_max)
    print('Number of cores used: %d' % n_cores)


# ------------------------------------- Simulate ------------------------------

# Simulate market
r, f = simulate_market(j_, t_, 1, B, mu_u, Sigma, Phi, mu_eps, Omega,
                       nonlinear=nonlinear, nn=nn, sig_nn=sig_nn,
                       nonlineartype=nonlineartype)

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
    return q_hat(state, action, B, qb_list, flag_qaverage=False, n_models=None)


if parallel_computing:
    if __name__ == '__main__':

        compute_rl_part = partial(compute_rl, f=f, q_value=q_value,
                                  lot_size=lot_size, optimizers=optimizers,
                                  optimizer=optimizer)

        p = mp.Pool(n_cores)
        shares = p.map(compute_rl_part, range(j_))
        p.close()
        p.join()
    shares = np.array(shares)

else:
    shares = np.zeros((j_, t_))
    for j in range(j_):
        print('Simulation', j+1, 'on', j_)
        shares[j, :] = compute_rl(j, f, q_value, lot_size, optimizers,
                                  optimizer=optimizer)

# Wealth
wealth_opt, value_opt, cost_opt = compute_wealth(r, x, gamma, Lambda, rho, B,
                                                 Sigma, Phi)

wealth_m, value_m, cost_m = compute_wealth(r, Markovitz, gamma, Lambda, rho, B,
                                           Sigma, Phi)

wealth_rl, value_rl, cost_rl = compute_wealth(r, shares, gamma, Lambda, rho, B,
                                              Sigma, Phi)


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
