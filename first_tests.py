# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:35:31 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from joblib import dump, load
from functools import partial

from dt_functions import (simulate_market, q_hat, generate_episode)
import multiprocessing as mp

parallel_computing = False  # True for parallel computing
if parallel_computing:
    print('Number of cores available: %d' % mp.cpu_count())
    n_cores = min(mp.cpu_count(), 40)
    print('Number of cores used: %d' % n_cores)
    dump(n_cores, 'data/n_cores.joblib')


# Seed
np.random.seed(19051991)

# Import data
path = 'Databases/Commodities/GASALLW_csv_2/data/'
n_ = 0
li = []

# BRENT
data_brent = pd.read_csv(path + 'DCOILBRENTEU.csv', index_col=0,
                         na_values='.').fillna(method='pad')
data_brent.columns = ['BRENT']
data_brent.index = pd.to_datetime(data_brent.index)
n_ = n_+1
li.append(data_brent)

# WTI
data_wti = pd.read_csv(path + 'DCOILWTICO.csv', index_col=0,
                       na_values='.').fillna(method='pad')
data_wti.columns = ['WTI']
data_wti.index = pd.to_datetime(data_wti.index)
n_ = n_+1
li.append(data_wti)

# Gold
data_gold = pd.read_csv(path + 'GOLDAMGBD228NLBM.csv', index_col=0,
                        na_values='.').fillna(method='pad')
data_gold.columns = ['Gold']
data_gold.index = pd.to_datetime(data_gold.index)
n_ = n_+1
li.append(data_gold)

# Henry Hub Natural Gas
data_hhng = pd.read_csv(path + 'DHHNGSP.csv', index_col=0,
                        na_values='.').fillna(method='pad')
data_hhng.columns = ['Henry Hub Natural Gas']
data_hhng.index = pd.to_datetime(data_hhng.index)
n_ = n_+1
li.append(data_hhng)

# Kerosene-Type Jet Fuel
data_ktjf = pd.read_csv(path + 'DJFUELUSGULF.csv', index_col=0,
                        na_values='.').fillna(method='pad')
data_ktjf.columns = ['Kerosene-Type Jet Fuel']
data_ktjf.index = pd.to_datetime(data_ktjf.index)
n_ = n_+1
li.append(data_ktjf)

# Propane
data_propane = pd.read_csv(path + 'DPROPANEMBTX.csv', index_col=0,
                           na_values='.').fillna(method='pad')
data_propane.columns = ['Propane']
data_propane.index = pd.to_datetime(data_propane.index)
n_ = n_+1
li.append(data_propane)

# Conventional Gasoline Prices: New York Harbor
data_gpny = pd.read_csv(path + 'DGASNYH.csv', index_col=0,
                        na_values='.').fillna(method='pad')
data_gpny.columns = ['Conventional Gasoline Prices: New York Harbor']
data_gpny.index = pd.to_datetime(data_gpny.index)
n_ = n_+1
li.append(data_gpny)

# Conventional Gasoline Prices: U.S. Gulf Coast
data_gpusg = pd.read_csv(path + 'DGASUSGULF.csv', index_col=0,
                         na_values='.').fillna(method='pad')
data_gpusg.columns = ['Conventional Gasoline Prices: U.S. Gulf Coast']
data_gpusg.index = pd.to_datetime(data_gpusg.index)
n_ = n_+1
li.append(data_gpusg)

# Merge dataframes
names = ['Brent', 'WTI', 'Gold',
         'Henry Hub Natural Gas', 'Kerosene-Type Jet Fuel',
         'Propane', 'Conventional Gasoline Prices: New York Harbor',
         'Conventional Gasoline Prices: U.S. Gulf Coast']

df_values = pd.concat(li, axis=1)
df_returns = df_values.diff().copy()
df_returns.dropna(inplace=True)

# Factors
window = 5
df_factors = df_returns.rolling(window=window).mean().copy()
df_factors.dropna(inplace=True)
dates = df_factors.index
df_returns = df_returns.loc[dates].copy()


# Fit of factors dynamics
params = {}
aic = {}
sig2 = {}
best = df_factors.columns[0]
for column in df_factors.columns:
    res = AutoReg(df_factors[column], lags=1).fit()
    params[column] = [res.params.iloc[0], res.params.iloc[1]]
    sig2[column] = res.sigma2
    aic[column] = res.aic
    if aic[column] < aic[best]:
        best = column


# Select best time series for the experiment
df_return = df_returns[best].copy()
df_factor = df_factors[best].copy()


# Fit model for the returns
reg = LinearRegression().fit(X=np.array(df_factor[:-1]).reshape(-1, 1),
                             y=np.array(df_return.iloc[1:]).reshape(-1, 1))


# Time series parameters
t_ = len(df_return)

B = reg.coef_[0, 0]
mu_u = reg.intercept_[0]
Sigma = (np.array(df_return.iloc[1:]) -
         B*np.array(df_factor[:-1]) -
         mu_u).var()

Phi = 1 - params[best][1]
mu_eps = params[best][0]
Omega = sig2[best]

# Model parameters
lam = 3*10**(-7)
Lambda = lam*Sigma
gamma = 10**-9
rho = 1-np.exp(-0.02/260)


# Example 1: Timing a single security

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
    x[t] = (1 - a/lam)*x[t-1] + a/lam * 1/(gamma*Sigma) * (B/(1+Phi*a/gamma))*df_factor.iloc[t]
x = np.round(x)


# ------------------------------------- REINFORCEMENT LEARNING ----------------

qb_list = []  # list to store models
n_batches = 3
eps = 0.5
eta = 1  # discount parameter
alpha = 1  # learning rate
j_ = 10

sup_model = 'ann_fast'
if sup_model == 'random_forest':
    from sklearn.ensemble import RandomForestRegressor
elif sup_model == 'ann_fast':
    from sklearn.neural_network import MLPRegressor
    hidden_layer_sizes = (64, 32, 8)
    # max_iter = 200  # these are sklearn default settings fro MLPRegressor
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

r, f = simulate_market(j_, t_, n_batches, df_factor, B, mu_u, Sigma, df_return, Phi,
                       mu_eps, Omega)

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
                          eta=eta, gamma=gamma)

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

    # export batch
    np.save('data/X%d' % b, X)
    np.save('data/Y%d' % b, Y)
    np.save('data/reward%d' % b, reward)
    np.save('data/cost%d' % b, cost)

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






df_strategies = pd.DataFrame(data=np.c_[x, np.r_[0, np.diff(x)],
                                        Markovitz, np.r_[0, np.diff(Markovitz)]],
                             index=df_return.index)
df_strategies.columns = ['Optimal shares', 'Optimal trades',
                         'Markovitz shares', 'Markovitz trades']



# Value
value = np.zeros(t_)
for t in range(t_ - 1):
    value[t] = (1 - rho)**(t + 1) * x[t]*df_return.iloc[t+1]

value_m = np.zeros(t_)
for t in range(t_ - 1):
    value_m[t] = (1 - rho)**(t + 1) * Markovitz[t]*df_return.iloc[t+1]


# Costs
cost = np.zeros(t_)
for t in range(1, t_):
    cost[t] = gamma/2 * (1 - rho)**(t + 1)*x[t]*Sigma*x[t] +\
        (1 - rho)**t/2*(x[t] - x[t-1])*Lambda*(x[t]-x[t-1])

cost_m = np.zeros(t_)
for t in range(1, t_):
    cost_m[t] = gamma/2 * (1 - rho)**(t + 1)*Markovitz[t]*Sigma*Markovitz[t] +\
        (1 - rho)**t/2*(Markovitz[t] - Markovitz[t-1])*Lambda*(Markovitz[t]-Markovitz[t-1])


# Wealth
df_wealth = pd.DataFrame(data=np.c_[np.cumsum(value), np.cumsum(value_m),
                                    np.cumsum(cost), np.cumsum(cost_m),
                                    np.cumsum(value) - np.cumsum(cost),
                                    np.cumsum(value_m) - np.cumsum(cost_m)])
df_wealth.columns = ['Value (optimal)', 'Value (Markovitz)',
                     'Costs (optimal)', 'Costs (Markovitz)',
                     'Wealth (optimal)', 'Wealth (Markovitz)']


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
ax.set_title('Shares')
plt.legend()
ax.yaxis.set_major_formatter(formatter)

fig, ax = plt.subplots()
ax.plot(df_strategies['Markovitz trades'], '--', color='b', label='Markovitz')
ax.plot(df_strategies['Optimal trades'], color='r', label='Optimal')
ax.set_title('Trades')
plt.legend()
ax.yaxis.set_major_formatter(formatter)

fig, ax = plt.subplots()
ax.plot(df_wealth['Value (Markovitz)'], '--', color='b', label='Markovitz')
ax.plot(df_wealth['Value (optimal)'], color='r', label='Optimal')
ax.set_title('Value')
plt.legend()
ax.yaxis.set_major_formatter(formatter)

fig, ax = plt.subplots()
ax.plot(df_wealth['Costs (Markovitz)'], '--', color='b', label='Markovitz')
ax.plot(df_wealth['Costs (optimal)'], color='r', label='Optimal')
ax.set_title('Costs')
plt.legend()
ax.yaxis.set_major_formatter(formatter)

fig, ax = plt.subplots()
ax.plot(df_wealth['Wealth (Markovitz)'], '--', color='b', label='Markovitz')
ax.plot(df_wealth['Wealth (optimal)'], color='r', label='Optimal')
ax.set_title('Wealth')
plt.legend()
ax.yaxis.set_major_formatter(formatter)
