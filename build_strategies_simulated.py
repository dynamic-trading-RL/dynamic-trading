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
n_batches = load('data/n_batches.joblib')
lot_size = load('data/lot_size.joblib')
optimizers = load('data/optimizers.joblib')


# Simulate market
j_ = 2
r, f = simulate_market(j_, t_, 1, B, mu_u, Sigma, Phi, mu_eps, Omega)
r = r.squeeze()
f = f.squeeze()


# Markovitz portfolio
Markovitz = np.zeros((j_, t_))
for t in range(t_):
    Markovitz[:, t] = (gamma*Sigma)**(-1)*B*f[:, t]
Markovitz = np.round(Markovitz)


# Optimal portfolio
a = (-(gamma*(1 - rho) + lam*rho) +
     np.sqrt((gamma*(1-rho) + lam*rho)**2 +
             4*gamma*lam*(1-rho)**2)) / (2*(1-rho))

x = np.zeros((j_, t_))
x[:, 0] = Markovitz[:, 0]
for t in range(1, t_):
    x[:, t] = (1 - a/lam)*x[:, t-1] +\
        a/lam * 1/(gamma*Sigma) * (B/(1+Phi*a/gamma))*f[:, t]
x = np.round(x)


# RL portfolio
print('##### Computing RL strategy')

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, B, qb_list, flag_qaverage=True, n_models=None)


shares = np.zeros((j_, t_))
for j in range(j_):
    print('Out of sample path: ', j+1, 'on', j_)
    for t in range(t_):
        progress = t/t_*100
        print('    Progress: %.2f %%' % progress)

        if t == 0:
            state = np.array([0, f[j, t]])
            action, optimizers = maxAction(q_value, state, lot_size,
                                           optimizers, t)
            shares[j, t] = state[0] + action
        else:
            state = np.array([shares[j, t-1], f[j, t]])
            action, optimizers = maxAction(q_value, state, lot_size,
                                           optimizers, t)
            shares[j, t] = state[0] + action

dump(optimizers, 'data/optimizers.joblib')


# Value
value = np.zeros((j_, t_))
for t in range(t_ - 1):
    value[:, t] = (1 - rho)**(t + 1) * x[:, t]*f[:, t+1]

value_m = np.zeros((j_, t_))
for t in range(t_ - 1):
    value_m[:, t] = (1 - rho)**(t + 1) * Markovitz[:, t]*f[:, t+1]

value_rl = np.zeros((j_, t_))
for t in range(t_ - 1):
    value_rl[:, t] = (1 - rho)**(t + 1) * shares[:, t]*f[:, t+1]


# Costs
cost = np.zeros((j_, t_))
for t in range(1, t_):
    cost[:, t] = gamma/2 * (1 - rho)**(t + 1)*x[:, t]*Sigma*x[:, t] +\
        (1 - rho)**t/2*(x[:, t] - x[:, t-1])*Lambda*(x[:, t]-x[:, t-1])

cost_m = np.zeros((j_, t_))
for t in range(1, t_):
    cost_m[:, t] = gamma/2 * (1 - rho)**(t + 1)*Markovitz[:, t]*Sigma*Markovitz[:, t] +\
        (1 - rho)**t/2*(Markovitz[:, t] -
                        Markovitz[:, t-1])*Lambda*(Markovitz[:, t]-Markovitz[:, t-1])

cost_rl = np.zeros((j_, t_))
for t in range(1, t_):
    cost_rl[:, t] = gamma/2 * (1 - rho)**(t + 1)*shares[:, t]*Sigma*shares[:, t] +\
        (1 - rho)**t/2*(shares[:, t] -
                        shares[:, t-1])*Lambda*(shares[:, t]-shares[:, t-1])


# Wealth
wealth = value - cost
wealth_m = value_m - cost_m
wealth_rl = value_rl - cost_rl

# Plots
plt.hist(wealth_m[:, -1], 90, label='Markovitz')
plt.hist(wealth_rl[:, -1], 90, label='RL')
plt.hist(wealth[:, -1], 90, label='Optimal')


# # Plots

# def human_format(num, pos):
#     magnitude = 0
#     while abs(num) >= 1000:
#         magnitude += 1
#         num /= 1000.0
#     # add more suffixes if you need them
#     return '%.f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


# formatter = FuncFormatter(human_format)

# fig, ax = plt.subplots()
# ax.plot(df_strategies['Markovitz shares'], '--', color='b', label='Markovitz')
# ax.plot(df_strategies['Optimal shares'], color='r', label='Optimal')
# ax.plot(df_strategies['RL shares'], color='g', label='RL')
# ax.set_title('Shares')
# plt.legend()
# ax.yaxis.set_major_formatter(formatter)
# plt.savefig('figures/shares.png')

# fig, ax = plt.subplots()
# ax.plot(df_strategies['Markovitz trades'], '--', color='b', label='Markovitz')
# ax.plot(df_strategies['Optimal trades'], color='r', label='Optimal')
# ax.plot(df_strategies['RL trades'], color='g', label='RL')
# ax.set_title('Trades')
# plt.legend()
# ax.yaxis.set_major_formatter(formatter)
# plt.savefig('figures/trades.png')

# fig, ax = plt.subplots()
# ax.plot(df_wealth['Value (Markovitz)'], '--', color='b', label='Markovitz')
# ax.plot(df_wealth['Value (optimal)'], color='r', label='Optimal')
# ax.plot(df_wealth['Value (RL)'], color='g', label='RL')
# ax.set_title('Value')
# plt.legend()
# ax.yaxis.set_major_formatter(formatter)
# plt.savefig('figures/value.png')

# fig, ax = plt.subplots()
# ax.plot(df_wealth['Costs (Markovitz)'], '--', color='b', label='Markovitz')
# ax.plot(df_wealth['Costs (optimal)'], color='r', label='Optimal')
# ax.plot(df_wealth['Costs (RL)'], color='g', label='RL')
# ax.set_title('Costs')
# plt.legend()
# ax.yaxis.set_major_formatter(formatter)
# plt.savefig('figures/costs.png')

# fig, ax = plt.subplots()
# ax.plot(df_wealth['Wealth (Markovitz)'], '--', color='b', label='Markovitz')
# ax.plot(df_wealth['Wealth (optimal)'], color='r', label='Optimal')
# ax.plot(df_wealth['Wealth (RL)'], color='g', label='RL')
# ax.set_title('Wealth')
# plt.legend()
# ax.yaxis.set_major_formatter(formatter)
# plt.savefig('figures/wealth.png')
