# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:35:31 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt


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

df = pd.concat(li, axis=1)
df.dropna(inplace=True)

# Factors
window = 5
df_factor = df.diff().rolling(window=window).mean()
df_factor.dropna(inplace=True)
dates = df_factor.index
df = df.loc[dates]

# Plots
for column in df.columns:
    plt.figure()
    plt.plot(df[column], label='Value')
    plt.plot(df_factor[column], label='Factor')
    plt.title(column)
    plt.legend()


# Fit
fits = {}
for column in df_factor.columns:
    res = AutoReg(df_factor[column], lags=1).fit()
    fits[column] = res














# Time series parameters
t_ = 1000  # length of the time series
r_f = 0  # risk-free return
# S = 1  # number of securities
# K = 2  # number of risk factors

# Model parameters
# B = np.random.randn(S, K)  # regression parameters
# Sigma = np.random.randn(S, S)  # residual covariance
# Sigma = Sigma@Sigma.T
# Phi = np.random.randn(K, K)
# Omega = np.random.randn(K, K)  # noise covariance
# Omega = Omega@Omega.T
B = 4
Sigma = 0.05**2
Phi = 0.1
Omega = 0.02**2


# Generate time series

epsi = np.sqrt(Omega)*np.random.randn(t_)
u = np.sqrt(Sigma)*np.random.randn(t_)
f = np.zeros(t_)
pnl = np.zeros(t_)
for t in range(t_-1):
    f[t+1] = f[t] - Phi*f[t] + epsi[t+1]
    pnl[t+1] = B*f[t] + u[t+1]

# Example 1
lam = 10**-4
gamma = 10**-7
rho = 1-np.exp(-0.02/260)
a = (-(gamma*(1 - rho) + lam*rho) +
     np.sqrt((gamma*(1-rho) + lam*rho)**2 +
             4*gamma*lam*(1-rho)**2)) / (2*(1-rho))

x = np.zeros(t_)
for t in range(1, t_):
    x[t] = (1 - a/lam)*x[t-1] + a/lam * 1/(gamma*Sigma) * (B/(1+Phi*a/gamma))*f[t]
x = x.astype(int)


# Costs
cost = np.zeros(t_)
for t in range(1, t_):
    cost[t] = (x[t]-x[t-1])*lam*(x[t]-x[t-1])


# Plots
plt.figure()
plt.plot(f, label='Factor')
plt.plot(pnl, label='Return')
plt.legend()

plt.figure()
plt.plot(x, label='Shares')
plt.bar(range(len(np.diff(x))), 5*np.diff(x), label='Trades', color='r')
plt.legend()

plt.figure()
plt.plot(np.cumsum(x*pnl - cost), label='Wealth')
plt.legend()
