# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:41:04 2021

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
parallel_computing = True     # True for parallel computing
n_batches = 5                 # number of batches
eps = 0.5                     # eps greedy
alpha = 1                     # learning rate
j_ = 1000                     # number of episodes

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
df_return = load('data/df_return.joblib')
df_factor = load('data/df_factor.joblib')
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


# Markovitz portfolio
Markovitz = np.zeros(t_)
for t in range(t_):
    Markovitz[t] = (gamma*Sigma)**(-1)*B*df_factor.iloc[t]
Markovitz = np.round(Markovitz)


# Optimal portfolio
a = (-(gamma*(1 - rho) + lam*rho) +
     np.sqrt((gamma*(1-rho) + lam*rho)**2 +
             4*gamma*lam*(1-rho)**2)) / (2*(1-rho))

x = np.zeros(t_)
x[0] = Markovitz[0]
for t in range(1, t_):
    x[t] = (1 - a/lam)*x[t-1] +\
        a/lam * 1/(gamma*Sigma) * (B/(1+Phi*a/gamma))*df_factor.iloc[t]
x = np.round(x)


# ------------------------------------- REINFORCEMENT LEARNING ----------------

print('##### Training RL agent')


inf_Markovitz = Markovitz.mean() - Markovitz.std()
sup_Markovitz = Markovitz.mean() + Markovitz.std()

lot_size = int(max(np.abs(inf_Markovitz), np.abs(sup_Markovitz)))

qb_list = []  # list to store models

r, f = simulate_market(j_, t_, n_batches, df_factor, B, mu_u, Sigma, df_return,
                       Phi, mu_eps, Omega)

if parallel_computing:
    print('Number of cores available: %d' % mp.cpu_count())
    n_cores = min(mp.cpu_count(), 40)
    print('Number of cores used: %d' % n_cores)

optimizers = []

for b in range(n_batches):  # loop on batches
    print('Creating batch %d of %d; eps=%f' % (b+1, n_batches, eps))
    X = []  # simulations
    Y = []
    j_sort = []
    reward_sort = []
    cost_sort = []

    if b == 0 or b == 1:
        draw_opt = False
    else:
        draw_opt = True

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
                          optimizers=optimizers, draw_opt=draw_opt)

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


print('##### Computing RL strategy')

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, B, qb_list, flag_qaverage=True, n_models=None)


shares = np.zeros(t_)
for t in range(t_):
    progress = t/t_*100
    print('    Progress: %.2f %%' % progress)

    if t == 0:
        state = np.array([0, df_factor.iloc[t]])
        action, optimizers = maxAction(q_value, state, lot_size, optimizers,
                                       draw_opt=True)
        shares[t] = state[0] + action
    else:
        state = np.array([shares[t-1], df_factor.iloc[t]])
        action, optimizers = maxAction(q_value, state, lot_size, optimizers,
                                       draw_opt=True)
        shares[t] = state[0] + action

dump(optimizers, 'data/optimizers.joblib')


df_strategies = pd.DataFrame(data=np.c_[x, np.r_[0, np.diff(x)],
                                        Markovitz,
                                        np.r_[0, np.diff(Markovitz)],
                                        shares, np.r_[0, np.diff(shares)]],
                              index=df_return.index)
df_strategies.columns = ['Optimal shares', 'Optimal trades',
                          'Markovitz shares', 'Markovitz trades',
                          'RL shares', 'RL trades']

# Value
value = np.zeros(t_)
for t in range(t_ - 1):
    value[t] = (1 - rho)**(t + 1) * x[t]*df_return.iloc[t+1]

value_m = np.zeros(t_)
for t in range(t_ - 1):
    value_m[t] = (1 - rho)**(t + 1) * Markovitz[t]*df_return.iloc[t+1]

value_rl = np.zeros(t_)
for t in range(t_ - 1):
    value_rl[t] = (1 - rho)**(t + 1) * shares[t]*df_return.iloc[t+1]


# Costs
cost = np.zeros(t_)
for t in range(1, t_):
    cost[t] = gamma/2 * (1 - rho)**(t + 1)*x[t]*Sigma*x[t] +\
        (1 - rho)**t/2*(x[t] - x[t-1])*Lambda*(x[t]-x[t-1])

cost_m = np.zeros(t_)
for t in range(1, t_):
    cost_m[t] = gamma/2 * (1 - rho)**(t + 1)*Markovitz[t]*Sigma*Markovitz[t] +\
        (1 - rho)**t/2*(Markovitz[t] -
                        Markovitz[t-1])*Lambda*(Markovitz[t]-Markovitz[t-1])

cost_rl = np.zeros(t_)
for t in range(1, t_):
    cost_rl[t] = gamma/2 * (1 - rho)**(t + 1)*shares[t]*Sigma*shares[t] +\
        (1 - rho)**t/2*(shares[t] -
                        shares[t-1])*Lambda*(shares[t]-shares[t-1])


# Wealth
df_wealth = pd.DataFrame(data=np.c_[np.cumsum(value), np.cumsum(value_m), np.cumsum(value_rl),
                                    np.cumsum(cost), np.cumsum(cost_m), np.cumsum(cost_rl),
                                    np.cumsum(value) - np.cumsum(cost),
                                    np.cumsum(value_m) - np.cumsum(cost_m),
                                    np.cumsum(value_rl) - np.cumsum(cost_rl)])
df_wealth.columns = ['Value (optimal)', 'Value (Markovitz)', 'Value (RL)',
                      'Costs (optimal)', 'Costs (Markovitz)', 'Costs (RL)',
                      'Wealth (optimal)', 'Wealth (Markovitz)', 'Wealth (RL)']


# Plots

def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


formatter = FuncFormatter(human_format)

fig, ax = plt.subplots()
ax.plot(df_strategies['Markovitz shares'], '--', color='b', label='Markovitz')
ax.plot(df_strategies['Optimal shares'], color='r', label='Optimal')
ax.plot(df_strategies['RL shares'], color='g', label='RL')
ax.set_title('Shares')
plt.legend()
ax.yaxis.set_major_formatter(formatter)
plt.savefig('figures/shares.png')

fig, ax = plt.subplots()
ax.plot(df_strategies['Markovitz trades'], '--', color='b', label='Markovitz')
ax.plot(df_strategies['Optimal trades'], color='r', label='Optimal')
ax.plot(df_strategies['RL trades'], color='g', label='RL')
ax.set_title('Trades')
plt.legend()
ax.yaxis.set_major_formatter(formatter)
plt.savefig('figures/trades.png')

fig, ax = plt.subplots()
ax.plot(df_wealth['Value (Markovitz)'], '--', color='b', label='Markovitz')
ax.plot(df_wealth['Value (optimal)'], color='r', label='Optimal')
ax.plot(df_wealth['Value (RL)'], color='g', label='RL')
ax.set_title('Value')
plt.legend()
ax.yaxis.set_major_formatter(formatter)
plt.savefig('figures/value.png')

fig, ax = plt.subplots()
ax.plot(df_wealth['Costs (Markovitz)'], '--', color='b', label='Markovitz')
ax.plot(df_wealth['Costs (optimal)'], color='r', label='Optimal')
ax.plot(df_wealth['Costs (RL)'], color='g', label='RL')
ax.set_title('Costs')
plt.legend()
ax.yaxis.set_major_formatter(formatter)
plt.savefig('figures/costs.png')

fig, ax = plt.subplots()
ax.plot(df_wealth['Wealth (Markovitz)'], '--', color='b', label='Markovitz')
ax.plot(df_wealth['Wealth (optimal)'], color='r', label='Optimal')
ax.plot(df_wealth['Wealth (RL)'], color='g', label='RL')
ax.set_title('Wealth')
plt.legend()
ax.yaxis.set_major_formatter(formatter)
plt.savefig('figures/wealth.png')
