# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:41:52 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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
                          compute_markovitz)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

np.random.seed(7890)


# ------------------------------------- Parameters ----------------------------

# RL parameters
j_episodes = 6000
n_batches = 7
t_ = 50

parallel_computing = True
n_cores_max = 50
alpha = 1.
eps = 0.1
optimizer = 'shgo'
# None, 'differential_evolution', 'shgo', 'dual_annealing', 'best'

# Market parameters
returnDynamicsType = ReturnDynamicsType.Linear
factorDynamicsType = FactorDynamicsType.AR
gamma = 3  # risk aversion
lam_perc = .01  # costs: percentage of unit trade value
rho = 1 - np.exp(-.02/252)  # discount

# RL model
sup_model = 'ann_fast'  # or random_forest or ann_deep or ann_fast


# ------------------------------------- Reinforcement learning ----------------
calibration_parameters = pd.read_excel('data/calibration_parameters.xlsx',
                                       index_col=0)
startPrice = calibration_parameters.loc['startPrice', 'calibration-parameters']
qb_list = []  # list to store models
optimizers = Optimizers()

if sup_model in ('ann_fast', 'ann_deep'):
    hidden_layer_sizes, max_iter, n_iter_no_change, alpha_ann =\
        set_regressor_parameters(sup_model)

if parallel_computing:
    print('Number of cores available: %d' % mp.cpu_count())
    n_cores = min(mp.cpu_count(), n_cores_max)
    print('Number of cores used: %d' % n_cores)

# Instantiate market
market = instantiate_market(returnDynamicsType, factorDynamicsType, startPrice)

# Simulations
price, pnl, f = simulate_market(market, j_episodes, n_batches, t_)
Sigma_r = get_Sigma(market)
lam = lam_perc / Sigma_r
Lambda_r = lam*Sigma_r

print('Approximate cost per unit trade: $ %.0f' % (price.mean()*Lambda_r))


# Use Markowitz to determine bounds
if (market._marketDynamics._returnDynamics._returnDynamicsType
        == ReturnDynamicsType.Linear):
    B = market._marketDynamics._returnDynamics._parameters['B']
else:
    B_0 = market._marketDynamics._returnDynamics._parameters['B_0']
    B_1 = market._marketDynamics._returnDynamics._parameters['B_1']
    B = .5*(B_0 + B_1)

Markowitz = compute_markovitz(f.flatten(), gamma, B*price.mean(),
                              Sigma_r*price.mean())

bound = np.abs(Markowitz).max()

reward = np.zeros((n_batches, j_episodes))
cost = np.zeros((n_batches, j_episodes))

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
            # return np.random.randn()
            return 0.

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
                          # market simulations
                          price=price[:, b, ], pnl=pnl[:, b, :],
                          # reward/cost parameters
                          rho=rho, gamma=gamma, Sigma_r=Sigma_r,
                          Lambda_r=Lambda_r,
                          # RL parameters
                          eps=eps, q_value=q_value, alpha=alpha,
                          optimizers=optimizers, optimizer=optimizer,
                          b=b, bound=bound)

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
    reward[b, :] = np.array(reward_sort)[ind_sort]
    cost[b, :] = np.array(cost_sort)[ind_sort]

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
    print('    Average reward: %.3f' % np.mean(reward[b, :]))
    print('    Average cost: %.3f' % np.mean(cost[b, :]))

    eps = max(eps/3, 0.00001)  # update epsilon


# ------------------------------------- Dump data -----------------------------
print(optimizers)
dump(n_batches, 'data/n_batches.joblib')
dump(optimizers, 'data/optimizers.joblib')
dump(lam_perc, 'data/lam_perc.joblib')
dump(gamma, 'data/gamma.joblib')
dump(rho, 'data/rho.joblib')


# ------------------------------------- Plots ---------------------------------

X_plot = X.reshape((j_episodes, t_-1, 3))

j_plot = min(X_plot.shape[0], 50)

color = cm.Greens(np.linspace(0, 1, n_batches))

plt.figure()
for j in range(j_plot):
    plt.plot(X_plot[j, :, 0], color='k', alpha=0.5)
plt.title('shares')
plt.savefig('figures/shares.png')

plt.figure()
for j in range(j_plot):
    plt.plot(X_plot[j, :, 1], color='k', alpha=0.5)
plt.title('pnl')
plt.savefig('figures/pnl.png')

plt.figure()
for j in range(j_plot):
    plt.bar(range(t_-1), X_plot[j, :, 2], color='k', alpha=0.5)
plt.title('trades')
plt.savefig('figures/trades.png')

plt.figure()
for b in range(n_batches):
    plt.plot(reward[b, :], label='Batch: %d' % b, alpha=0.5, color=color[b])
plt.legend()
plt.title('reward')
plt.savefig('figures/reward.png')

plt.figure()
for b in range(n_batches):
    plt.plot(np.cumsum(reward[b, :]), label='Batch: %d' % b, alpha=0.5,
             color=color[b])
plt.legend()
plt.title('cum-reward')
plt.savefig('figures/cum-reward.png')

plt.figure()
plt.plot(np.cumsum(reward[-1, :]), alpha=0.5)
plt.title('cum-reward-final')
plt.savefig('figures/cum-reward-final.png')

plt.figure()
for b in range(n_batches):
    plt.plot(cost[b, :], label='Batch: %d' % b, alpha=0.5, color=color[b])
plt.legend()
plt.title('cost')
plt.savefig('figures/cost.png')

plt.figure()
for b in range(n_batches):
    plt.plot(np.cumsum(cost[b, :]), label='Batch: %d' % b, alpha=0.5,
             color=color[b])
plt.legend()
plt.title('cum-cost')
plt.savefig('figures/cum-cost.png')

plt.figure()
plt.plot(np.cumsum(cost[-1, :]), alpha=0.5)
plt.title('cum-cost-final')
plt.savefig('figures/cum-cost-final.png')
