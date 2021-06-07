# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:34:59 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from joblib import dump

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


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

t_ = 50

df_values = pd.concat(li, axis=1).iloc[-t_:]
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

best = 'WTI'  # ??? picked for testing

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
lam = 10**-2  # ??? should be calibrated
Lambda = lam*Sigma
gamma = 10**-3  # 1/gamma is the magnitude of money under management
rho = 1-np.exp(-0.02/260)

# Export data
dump(df_return, 'data/df_return.joblib')
dump(df_factor, 'data/df_factor.joblib')
dump(t_, 'data/t_.joblib')
dump(B, 'data/B.joblib')
dump(mu_u, 'data/mu_u.joblib')
dump(Sigma, 'data/Sigma.joblib')
dump(Phi, 'data/Phi.joblib')
dump(mu_eps, 'data/mu_eps.joblib')
dump(Omega, 'data/Omega.joblib')
dump(lam, 'data/lam.joblib')
dump(Lambda, 'data/Lambda.joblib')
dump(gamma, 'data/gamma.joblib')
dump(rho, 'data/rho.joblib')
