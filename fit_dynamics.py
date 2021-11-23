# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:27:18 2021

@author: -
"""

import numpy as np
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from arch import arch_model
from arch.univariate import ARX, GARCH
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt


# ------------------------------------- Download data -------------------------

snp500 = yf.download('^GSPC')['Adj Close']
first_valid_loc = snp500.first_valid_index()
snp500 = snp500.loc[first_valid_loc:]
snp500.name = '^GSPC'
snp500 = snp500.to_frame()
df = snp500.copy().iloc[-8000:]
df['r'] = 1000*np.log(df['^GSPC']).diff()

# Factors
window = 5
df['f'] = df['r'].rolling(window).mean()

df.dropna(inplace=True)


# ------------------------------------- Fit of dynamics -----------------------

# ------------------ RETURNS

# Linear prediction
df_reg = df[['f', 'r']].copy()
df_reg['r'] = df_reg['r'].shift(1)
df_reg.dropna(inplace=True)

reg = OLS(df_reg['r'], add_constant(df_reg['f'])).fit()

B = reg.params['f']
mu_u = reg.params['const']
Sigma2_u = reg.mse_resid


# Non-linear prediction
c = 0.

ind_0 = df_reg['f'] < c
ind_1 = df_reg['f'] >= c

reg_0 = OLS(df_reg['r'].loc[ind_0], add_constant(df_reg['f'].loc[ind_0])).fit()

B_0 = reg_0.params['f']
mu_u_0 = reg_0.params['const']
Sigma2_u_0 = reg_0.mse_resid
print(reg_0.summary())

reg_1 = OLS(df_reg['r'].loc[ind_1], add_constant(df_reg['f'].loc[ind_1])).fit()

B_1 = reg_1.params['f']
mu_u_1 = reg_1.params['const']
Sigma2_u_1 = reg_1.mse_resid
print(reg_1.summary())


# ------------------ FACTORS

# GARCH(1, 1) on factors
garch = arch_model(df['f'].diff().dropna(), p=1, q=1, rescale=False)
res_garch = garch.fit()
resid = res_garch.resid
sigma_garch = res_garch.conditional_volatility
epsi_garch = np.divide(resid, sigma_garch)
mu_garch = res_garch.params['mu']
omega_garch = res_garch.params['omega']
alpha_garch = res_garch.params['alpha[1]']
beta_garch = res_garch.params['beta[1]']
print(res_garch.summary())

# TARCH(1, 1, 1) on factors
tarch = arch_model(df['f'].diff().dropna(), p=1, o=1, q=1, rescale=False)
res_tarch = tarch.fit()
resid = res_tarch.resid
sigma_tarch = res_tarch.conditional_volatility
epsi_tarch = np.divide(resid, sigma_tarch)
mu_tarch = res_tarch.params['mu']
omega_tarch = res_tarch.params['omega']
alpha_tarch = res_tarch.params['alpha[1]']
gamma_tarch = res_tarch.params['gamma[1]']
beta_tarch = res_tarch.params['beta[1]']
print(res_tarch.summary())

# AR-TARCH(1, 1, 1) on factors
arx = ARX(df['f'], lags=1, rescale=False)
arx.volatility = GARCH(p=1, o=1, q=1)
res_arx = arx.fit()
Phi_arx = 1 - res_arx.params['f[1]']
mu_arx = res_arx.params['Const']
omega_arx = res_arx.params['omega']
alpha_arx = res_arx.params['alpha[1]']
gamma_arx = res_arx.params['gamma[1]']
beta_arx = res_arx.params['beta[1]']
print(res_arx.summary())


# AR(1) on factors
res_ar = AutoReg(df['f'], lags=1, old_names=False).fit()
mu_ar = res_ar.params.iloc[0]
Phi_ar = 1 - res_ar.params.iloc[1]
Omega_ar = res_ar.sigma2
epsi_ar = df['f'].iloc[1:] - Phi_ar*df['f'].iloc[:-1] - mu_ar
print(res_ar.summary())

# SETAR on factors
ind_0 = df['f'] < 0
ind_1 = df['f'] >= 0

res_ar_0 = AutoReg(df['f'].loc[ind_0], lags=1, old_names=False).fit()
mu_ar_0 = res_ar_0.params.iloc[0]
Phi_ar_0 = 1 - res_ar_0.params.iloc[1]
Omega_ar_0 = res_ar_0.sigma2
epsi_ar_0 = df['f'].loc[ind_0].iloc[1:] - Phi_ar_0*df['f'].loc[ind_0].iloc[:-1] - mu_ar_0
print(res_ar_0.summary())

res_ar_1 = AutoReg(df['f'].loc[ind_1], lags=1, old_names=False).fit()
mu_ar_1 = res_ar_1.params.iloc[0]
Phi_ar_1 = 1 - res_ar_1.params.iloc[1]
Omega_ar_1 = res_ar_1.sigma2
epsi_ar_1 = df['f'].loc[ind_1].iloc[1:] - Phi_ar_1*df['f'].loc[ind_1].iloc[:-1] - mu_ar_1
print(res_ar_1.summary())


###############################################################################
###############################################################################
###############################################################################

from dt_functions import (ReturnDynamics, FactorDynamics,
                          ReturnDynamicsType, FactorDynamicsType,
                          MarketDynamics, Market)

returnDynamicsType = ReturnDynamicsType.R2
factorDynamicsType = FactorDynamicsType.F5

returnDynamics = ReturnDynamics(returnDynamicsType)
factorDynamics = FactorDynamics(factorDynamicsType)

if returnDynamicsType == ReturnDynamicsType.R1:
    return_parameters = {'B': B, 'mu': mu_u, 'sig2': Sigma2_u}
elif returnDynamicsType == ReturnDynamicsType.R2:
    return_parameters = {'B_0': B_0, 'mu_0': mu_u_0, 'sig2_0': Sigma2_u_0,
                         'B_1': B_1, 'mu_1': mu_u_1, 'sig2_1': Sigma2_u_1,
                         'c': c}
else:
    raise NameError('Invalid returnDynamicsType')

if factorDynamicsType == FactorDynamicsType.F1:
    factor_parameters = {'B': 1-Phi_ar, 'mu': mu_ar, 'sig2': Omega_ar}
elif factorDynamicsType == FactorDynamicsType.F2:
    factor_parameters = {'B_0': 1-Phi_ar_0, 'mu_0': mu_ar_0,
                         'sig2_0': Omega_ar_0,
                         'B_1': 1-Phi_ar_1, 'mu_1': mu_ar_1,
                         'sig2_1': Omega_ar_1,
                         'c': c}
elif factorDynamicsType == FactorDynamicsType.F3:
    factor_parameters = {'mu': mu_garch, 'omega': omega_garch,
                         'alpha': alpha_garch, 'beta': beta_garch}
elif factorDynamicsType == FactorDynamicsType.F4:
    factor_parameters = {'mu': mu_tarch, 'omega': omega_tarch,
                         'alpha': alpha_tarch, 'beta': beta_tarch,
                         'gamma': gamma_tarch, 'c': c}
elif factorDynamicsType == FactorDynamicsType.F5:
    factor_parameters = {'B': 1-Phi_arx,'mu': mu_arx, 'omega': omega_arx,
                         'alpha': alpha_arx, 'beta': beta_arx,
                         'gamma': gamma_arx, 'c': c}


returnDynamics.set_parameters(return_parameters)
factorDynamics.set_parameters(factor_parameters)

marketDynamics = MarketDynamics(returnDynamics=returnDynamics,
                                factorDynamics=factorDynamics)

market = Market(marketDynamics)
j_ = 1
t_ = 8000
market.simulate(j_=j_, t_=t_)

f = market._simulations['f']
r = market._simulations['r']

fig = plt.figure()
for j in range(min(50, j_)):
    plt.plot(r[j, :], color='k', alpha=0.6)
plt.title('Return')

fig = plt.figure()
for j in range(min(50, j_)):
    plt.plot(f[j, :], color='r', alpha=0.6)
plt.title('Factor')

fig = plt.figure()
plt.scatter(f[:, :-1].flatten(), r[:, 1:].flatten(), s=2)
plt.title('Factor vs Return')

