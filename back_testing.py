# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 08:22:45 2021

@author: Giorgi
"""

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from joblib import load, dump
from dt_functions import (q_hat, compute_markovitz, compute_optimal,
                          compute_rl, compute_wealth)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


print('######## Backtesting')

# ------------------------------------- Parameters ----------------------------

optimizer = None

# Import parameters from previos scripts
df_return = load('data/df_return.joblib')
df_factor = load('data/df_factor.joblib')
t_ = load('data/t_.joblib')
B = load('data/B.joblib')
mu_u = load('data/mu_u.joblib')
Sigma = load('data/Sigma.joblib')
Phi = load('data/Phi.joblib')
Lambda = load('data/Lambda.joblib')
lam = load('data/lam.joblib')
gamma = load('data/gamma.joblib')
rho = load('data/rho.joblib')
n_batches = load('data/n_batches.joblib')
lot_size = load('data/lot_size.joblib')
optimizers = load('data/optimizers.joblib')


# ------------------------------------- Markovitz portfolio -------------------

print('#### Computing Markovitz strategy')

Markovitz = compute_markovitz(df_factor.to_numpy(), gamma, B, Sigma)


# ------------------------------------- Optimal portfolio ---------------------

print('#### Computing optimal strategy')

x = compute_optimal(df_factor.to_numpy(), gamma, Lambda, rho, B, Sigma, Phi)


# ------------------------------------- RL portfolio ---------------------

print('#### Computing RL strategy')

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, qb_list, flag_qaverage=False, n_models=None)


shares = compute_rl(0, df_factor.to_numpy(), q_value, lot_size, optimizers,
                    optimizer=optimizer)


# ------------------------------------- Results -------------------------------

# Strategies

df_strategies = pd.DataFrame(data=np.c_[x, np.r_[0, np.diff(x)],
                                        Markovitz,
                                        np.r_[0, np.diff(Markovitz)],
                                        shares, np.r_[0, np.diff(shares)]],
                             index=df_return.index)
df_strategies.columns = ['Optimal shares', 'Optimal trades',
                         'Markovitz shares', 'Markovitz trades',
                         'RL shares', 'RL trades']

# Wealth

wealth_opt, value_opt, cost_opt = compute_wealth(df_return.to_numpy(), x,
                                                 gamma, Lambda, rho, B, Sigma,
                                                 Phi)

wealth_m, value_m, cost_m = compute_wealth(df_return.to_numpy(), Markovitz,
                                           gamma, Lambda, rho, B, Sigma,
                                           Phi)

wealth_rl, value_rl, cost_rl = compute_wealth(df_return.to_numpy(), shares,
                                              gamma, Lambda, rho, B, Sigma,
                                              Phi)

df_wealth = pd.DataFrame(data=np.c_[value_opt, value_m, value_rl,
                                    cost_opt, cost_m, cost_rl,
                                    wealth_opt, wealth_m, wealth_rl])
df_wealth.columns = ['Value (optimal)', 'Value (Markovitz)', 'Value (RL)',
                     'Costs (optimal)', 'Costs (Markovitz)', 'Costs (RL)',
                     'Wealth (optimal)', 'Wealth (Markovitz)', 'Wealth (RL)']


# ------------------------------------- Dump data -----------------------------

dump(df_wealth, 'data/df_wealth_bktst.joblib')
dump(df_strategies, 'data/df_strategies_bktst.joblib')


# ------------------------------------- Plots ---------------------------------

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
plt.savefig('figures/value_opt.png')

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
