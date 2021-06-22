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
from dt_functions import q_hat, maxAction
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


print('######## Backtesting')


# Import parameters
df_return = load('data/df_return.joblib')
df_factor = load('data/df_factor.joblib')
t_ = load('data/t_.joblib')
B = load('data/B.joblib')
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

Markovitz = np.zeros(t_)
for t in range(1, t_):
    Markovitz[t] = (gamma*Sigma)**(-1)*B*df_factor.iloc[t]
Markovitz = np.round(Markovitz)


# ------------------------------------- Optimal portfolio ---------------------

print('#### Computing optimal strategy')

a = (-(gamma*(1 - rho) + lam*rho) +
     np.sqrt((gamma*(1-rho) + lam*rho)**2 +
             4*gamma*lam*(1-rho)**2)) / (2*(1-rho))

x = np.zeros(t_)
for t in range(1, t_):
    x[t] = (1 - a/lam)*x[t-1] +\
        a/lam * 1/(gamma*Sigma) * (B/(1+Phi*a/gamma))*df_factor.iloc[t]
x = np.round(x)


# ------------------------------------- RL portfolio ---------------------

print('#### Computing RL strategy')

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
        shares[t] = state[0]
        action = maxAction(q_value, state, lot_size, optimizers,
                           optimizer='best')
    else:
        state = np.array([shares[t-1] + action, df_factor.iloc[t]])
        shares[t] = state[0]
        action = maxAction(q_value, state, lot_size, optimizers,
                           optimizer='best')


# ------------------------------------- Results -------------------------------

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
df_wealth = pd.DataFrame(data=np.c_[np.cumsum(value), np.cumsum(value_m),
                                    np.cumsum(value_rl),
                                    np.cumsum(cost), np.cumsum(cost_m),
                                    np.cumsum(cost_rl),
                                    np.cumsum(value) - np.cumsum(cost),
                                    np.cumsum(value_m) - np.cumsum(cost_m),
                                    np.cumsum(value_rl) - np.cumsum(cost_rl)])
df_wealth.columns = ['Value (optimal)', 'Value (Markovitz)', 'Value (RL)',
                     'Costs (optimal)', 'Costs (Markovitz)', 'Costs (RL)',
                     'Wealth (optimal)', 'Wealth (Markovitz)', 'Wealth (RL)']


dump(df_wealth, 'data/df_wealth_bktst.joblib')
dump(df_strategies, 'data/df_strategies_bktst.joblib')


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
