# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:34:45 2021

@author: Giorgi
"""

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from joblib import load, dump
import multiprocessing as mp
from functools import partial
from dt_functions import (simulate_market, q_hat, generate_episode, maxAction)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Set parameters
parallel_computing = False     # True for parallel computing
n_batches = 2                 # number of batches
dump(n_batches, 'data/n_batches.joblib')
eps = 0.5                     # eps greedy
alpha = 1                     # learning rate
j_ = 1                     # number of episodes

# RL model
sup_model = 'ann_fast'
if sup_model == 'random_forest':
    from sklearn.ensemble import RandomForestRegressor
elif sup_model == 'ann_fast':
    from sklearn.neural_network import MLPRegressor
    hidden_layer_sizes = (64, 32, 8)
    # max_iter = 200  # these are sklearn default settings for MLPRegressor
    # n_iter_no_change = 10
    # alpha_ann = 0.001
    max_iter = 10
    n_iter_no_change = 2
    alpha_ann = 0.0001
elif sup_model == 'ann_deep':
    from sklearn.neural_network import MLPRegressor
    hidden_layer_sizes = (70, 50, 30, 10)
    max_iter = 200
    n_iter_no_change = 10
    alpha_ann = 0.001


# Import parameters
t_ = load('data/t_.joblib')
B = load('data/B.joblib')
mu_u = load('data/mu_u.joblib')
Sigma = load('data/Sigma.joblib')
Phi = load('data/Phi.joblib')
mu_eps = load('data/mu_eps.joblib')
Omega = load('data/Omega.joblib')
Lambda = load('data/Lambda.joblib')
lam = load('data/lam.joblib')
gamma = load('data/gamma.joblib')
rho = load('data/rho.joblib')


# ------------------------------------- REINFORCEMENT LEARNING ----------------

print('##### Training RL agent')

lot_size = 200  # ???
dump(lot_size, 'data/lot_size.joblib')

print('lot_size =', lot_size)

qb_list = []  # list to store models

r, f = simulate_market(j_, t_, n_batches, B, mu_u, Sigma, Phi, mu_eps, Omega)

if parallel_computing:
    print('Number of cores available: %d' % mp.cpu_count())
    n_cores = min(mp.cpu_count(), 40)
    print('Number of cores used: %d' % n_cores)

optimizers = {'shgo': {'n': 0, 'times': []},
              'dual_annealing': {'n': 0, 'times': []},
              'differential_evolution': {'n': 0, 'times': []},
              'basinhopping': {'n': 0, 'times': []}}

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
            return q_hat(state, action, n_batches, qb_list, flag_qaverage=True,
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
                          optimizers=optimizers)

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
            optimizers.append(episodes[j][5])
    else:
        for j in range(j_):
            print('Computing episode '+str(j+1)+' on '+str(j_))
            episodes = gen_ep_part(j)
            X.append(episodes[0])
            Y.append(episodes[1])
            j_sort.append(episodes[2])
            reward_sort.append(episodes[3])
            cost_sort.append(episodes[4])
            optimizers = episodes[5]

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

    eps = max(eps/3, 10**-6)  # update epsilon

dump(optimizers, 'data/optimizers.joblib')
