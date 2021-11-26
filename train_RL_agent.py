# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:41:52 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from joblib import load, dump
from functools import partial
import multiprocessing as mp
from dt_functions import (ReturnDynamicsType, FactorDynamicsType,
                          instantiate_market,
                          get_Sigma,
                          simulate_market,
                          q_hat,
                          generate_episode,
                          Optimizers,
                          set_regressor_parameters,
                          compute_rl)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

np.random.seed(7890)


# ------------------------------------- Parameters ----------------------------

# RL parameters
j_episodes = 1000
n_batches = 3
t_ = 50

parallel_computing = False
n_cores_max = 20
alpha = 1.
eps = 0.1
optimizer = 'brute'
# None, 'differential_evolution', 'shgo', 'dual_annealing', 'best'

# Market parameters
returnDynamicsType = ReturnDynamicsType.Linear
factorDynamicsType = FactorDynamicsType.AR
gamma = 10**-3  # risk aversion
lam = 10**-2  # costs
rho = 1 - np.exp(-.02/252)  # discount

# RL model
sup_model = 'ann_fast'  # or random_forest or ann_deep


# ------------------------------------- Reinforcement learning ----------------
calibration_parameters = pd.read_excel('data/calibration_parameters.xlsx',
                                       index_col=0)
scale = calibration_parameters.loc['scale']
qb_list = []  # list to store models
optimizers = Optimizers()
hidden_layer_sizes, max_iter, n_iter_no_change, alpha_ann =\
    set_regressor_parameters(sup_model)

if parallel_computing:
    print('Number of cores available: %d' % mp.cpu_count())
    n_cores = min(mp.cpu_count(), n_cores_max)
    print('Number of cores used: %d' % n_cores)


# Instantiate market
market = instantiate_market(returnDynamicsType, factorDynamicsType)

# Simulations
r, f = simulate_market(market, j_episodes, n_batches, t_, scale)
Sigma = get_Sigma(market)
Lambda = lam*Sigma

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
                          # market simulations
                          r=r[:, b, :],
                          # reward/cost parameters
                          rho=rho, gamma=gamma, Sigma=Sigma, Lambda=Lambda,
                          # RL parameters
                          eps=eps, q_value=q_value, alpha=alpha,
                          optimizers=optimizers, optimizer=optimizer,
                          b=b)

    if parallel_computing:
        if __name__ == '__main__':
            p = mp.Pool(n_cores)
            episodes = p.map(gen_ep_part, range(j_episodes))
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
        for j in range(j_episodes):
            print('Computing episode '+str(j+1)+' on '+str(j_episodes))
            episodes = gen_ep_part(j)
            X.append(episodes[0])
            Y.append(episodes[1])
            j_sort.append(episodes[2])
            reward_sort.append(episodes[3])
            cost_sort.append(episodes[4])

    X = np.array(X).reshape((j_episodes*(t_-1), 3))
    Y = np.array(Y).reshape((j_episodes*(t_-1)))

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

    eps = max(eps/3, 0.00001)  # update epsilon


# ------------------------------------- Dump data -----------------------------
print(optimizers)
dump(n_batches, 'data/n_batches.joblib')
dump(optimizers, 'data/optimizers.joblib')
