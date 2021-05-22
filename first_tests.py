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
r_f = 0  # risk-free return

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
