# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 14:51:24 2021

@author: feder
"""

from arch.univariate import ARX
from arch.univariate import GARCH


import numpy as np
from arch import arch_model
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from matplotlib import pyplot as plt


# ------------------------------------- Parameters ----------------------------

# Model parameters
lam = 10**-2               # costs factor: ??? should be calibrated
gamma = 10**-3             # 1/gamma is the magnitude of money under management
rho = 1-np.exp(-0.02/260)  # discount factor (2% annualized)


# ------------------------------------- Import --------------------------------

# # Import data
# path = 'Databases/Commodities/GASALLW_csv_2/data/'


# # WTI
# data_wti = pd.read_csv(path + 'DCOILWTICO.csv', index_col=0,
#                        na_values='.').fillna(method='pad')
# data_wti.columns = ['WTI']
# data_wti.index = pd.to_datetime(data_wti.index)
# df = data_wti.copy()
# df['r'] = 100*df['WTI'].pct_change()

snp500 = yf.download('^GSPC')['Adj Close']
first_valid_loc = snp500.first_valid_index()
snp500 = snp500.loc[first_valid_loc:]
snp500.name = '^GSPC'
snp500 = snp500.to_frame()
df = snp500.copy().iloc[-8000:]
df['r'] = np.log(df['^GSPC']).diff()

# Factors
window = 5
df['f'] = df['r'].rolling(window).mean()

df.dropna(inplace=True)


# ------------------------------------- Fit of dynamics -----------------------

# GARCH(1, 1) on factors
garch = arch_model(df['f'].diff().dropna(), p=1, q=1, rescale=True)
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
tarch = arch_model(df['f'].diff().dropna(), p=1, o=1, q=1, rescale=True)
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

# Linear prediction
df_reg = df[['f', 'r']].copy()
df_reg['r'] = df_reg['r'].shift(1)
df_reg.dropna(inplace=True)

reg = OLS(df_reg['r'], add_constant(df_reg['f'])).fit()

B = reg.params['f']
mu_u = reg.params['const']
Sigma2_u = reg.mse_resid
Lambda_lin = lam*Sigma2_u
print(reg.summary())

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

Lambda_nonlin = lam*(Sigma2_u_0 + Sigma2_u_1)/2
