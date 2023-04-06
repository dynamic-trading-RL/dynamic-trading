# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:27:18 2021

@author: -
"""

import numpy as np
import pandas as pd
from joblib import dump
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from arch import arch_model
from arch.univariate import ARX, GARCH
from statsmodels.tsa.ar_model import AutoReg

# ------------------------------------- Parameters ----------------------------

fit_stock = False
return_is_pnl = True
t_ = 50
dump(fit_stock, 'data/fit_stock.joblib')
dump(return_is_pnl, 'data/return_is_pnl.joblib')

if fit_stock:
    ticker = '^GSPC'  # '^GSPC'
    end_date = '2021-12-31'
    c = 0.
    scale = 1
    scale_f = 1  # or "= scale"
    t_past = 8000
    window = 5

    # ------------------------------------- Download data ---------------------

    stock = yf.download(ticker, end=end_date)['Adj Close']
    first_valid_loc = stock.first_valid_index()
    stock = stock.loc[first_valid_loc:]
    startPrice = stock.iloc[-1]
    stock.name = ticker
    stock = stock.to_frame()
    df = stock.copy().iloc[-t_past:]

else:
    ticker = 'WTI'
    c = 0.
    scale = 1
    scale_f = 1  # or "= scale"
    t_past = 8000
    window = 5

    # ------------------------------------- Import data -----------------------

    data_wti = pd.read_csv('source_data/DCOILWTICO.csv', index_col=0,
                           na_values='.').fillna(method='pad')
    data_wti.columns = ['WTI']
    data_wti.index = pd.to_datetime(data_wti.index)
    df = data_wti.copy().iloc[-t_past:]
    end_date = str(df.index[-1])[:10]
    startPrice = data_wti.iloc[-1]['WTI']

if return_is_pnl:
    df['r'] = scale*df[ticker].diff()
else:
    df['r'] = scale*df[ticker].pct_change()

# NB: returns and log returns are almost equal

# Factors
if fit_stock:
    df['f'] = df['r'].rolling(window).mean() / df['r'].rolling(window).std()
else:
    df['f'] = df['r'].rolling(window).mean()

df.dropna(inplace=True)

df.to_csv('data/df.csv')

dump(ticker, 'data/ticker.joblib')

calibration_parameters = pd.DataFrame(index=['ticker', 'end_date',
                                             'startPrice', 't_past',
                                             'window'],
                                      data=[ticker, end_date, startPrice,
                                            t_past, window],
                                      columns=['calibration-parameters'])


# ------------------------------------- Fit of dynamics -----------------------
# hold-out
df = df.iloc[:-t_].copy()

# ------------------ RETURNS

# Linear prediction
df_reg = df[['f', 'r']].copy()
df_reg['r'] = df_reg['r'].shift(-1)
df_reg.dropna(inplace=True)

reg = OLS(df_reg['r'], add_constant(df_reg['f'])).fit()

B = reg.params['f']
mu_u = reg.params['const'] / scale
Sigma2_u = reg.mse_resid / scale**2

res_linear = pd.DataFrame(index=['mu', 'B', 'sig2'],
                          data=[mu_u, B, Sigma2_u],
                          columns=['param'])

with open('reports/' + ticker + '-return_linear.txt', 'w+') as fh:
    fh.write(reg.summary().as_text())


# Non-linear prediction
ind_0 = df_reg['f'] < c
ind_1 = df_reg['f'] >= c

reg_0 = OLS(df_reg['r'].loc[ind_0], add_constant(df_reg['f'].loc[ind_0])).fit()

B_0 = reg_0.params['f']
mu_u_0 = reg_0.params['const'] / scale
Sigma2_u_0 = reg_0.mse_resid / scale**2

reg_1 = OLS(df_reg['r'].loc[ind_1], add_constant(df_reg['f'].loc[ind_1])).fit()

B_1 = reg_1.params['f']
mu_u_1 = reg_1.params['const'] / scale
Sigma2_u_1 = reg_1.mse_resid / scale**2

res_non_linear = pd.DataFrame(index=['mu_0', 'B_0', 'sig2_0',
                                     'mu_1', 'B_1', 'sig2_1',
                                     'c'],
                              data=[mu_u_0, B_0, Sigma2_u_0,
                                    mu_u_1, B_1, Sigma2_u_1,
                                    c],
                              columns=['param'])

with open('reports/' + ticker + '-return_nonlinear_0.txt', 'w+') as fh:
    fh.write(reg_0.summary().as_text())

with open('reports/' + ticker + '-return_nonlinear_1.txt', 'w+') as fh:
    fh.write(reg_1.summary().as_text())


# ------------------ FACTORS

# AR(1) on factors
res_ar = AutoReg(df['f'], lags=1, old_names=False).fit()
mu_ar = res_ar.params.iloc[0] / scale_f
Phi_ar = 1 - res_ar.params.iloc[1]
Omega_ar = res_ar.sigma2 / scale_f**2
epsi_ar =\
    df['f'].iloc[1:] / scale_f - Phi_ar*df['f'].iloc[:-1] / scale_f - mu_ar

with open('reports/' + ticker + '-factor_AR.txt', 'w+') as fh:
    fh.write(res_ar.summary().as_text())

res_ar = pd.DataFrame(index=['mu', 'B', 'sig2'],
                      data=[mu_ar, 1 - Phi_ar, Omega_ar],
                      columns=['param'])

# SETAR on factors
ind_0 = df['f'] < c
ind_1 = df['f'] >= c

res_ar_0 = AutoReg(df['f'].loc[ind_0], lags=1, old_names=False).fit()
mu_ar_0 = res_ar_0.params.iloc[0] / scale_f
Phi_ar_0 = 1 - res_ar_0.params.iloc[1]
Omega_ar_0 = res_ar_0.sigma2 / scale_f**2
epsi_ar_0 = df['f'].loc[ind_0].iloc[1:] / scale_f -\
    Phi_ar_0*df['f'].loc[ind_0].iloc[:-1] / scale_f - mu_ar_0

res_ar_1 = AutoReg(df['f'].loc[ind_1], lags=1, old_names=False).fit()
mu_ar_1 = res_ar_1.params.iloc[0] / scale_f
Phi_ar_1 = 1 - res_ar_1.params.iloc[1]
Omega_ar_1 = res_ar_1.sigma2 / scale_f**2
epsi_ar_1 = df['f'].loc[ind_1].iloc[1:] / scale_f -\
    Phi_ar_1*df['f'].loc[ind_1].iloc[:-1] / scale_f - mu_ar_1

with open('reports/' + ticker + '-factor_SETAR_0.txt', 'w+') as fh:
    fh.write(res_ar_0.summary().as_text())

with open('reports/' + ticker + '-factor_SETAR_0.txt', 'w+') as fh:
    fh.write(res_ar_1.summary().as_text())

res_setar = pd.DataFrame(index=['mu_0', 'B_0', 'sig2_0',
                                'mu_1', 'B_1', 'sig2_1',
                                'c'],
                         data=[mu_ar_0, 1 - Phi_ar_0, Omega_ar_0,
                               mu_ar_1, 1 - Phi_ar_1, Omega_ar_1,
                               c],
                         columns=['param'])


# GARCH(1, 1) on factors
garch = arch_model(df['f'].diff().dropna(), p=1, q=1, rescale=False)
res_garch = garch.fit()
resid = res_garch.resid / scale_f
sigma_garch = res_garch.conditional_volatility / scale_f
epsi_garch = np.divide(resid, sigma_garch)
mu_garch = res_garch.params['mu'] / scale_f
omega_garch = res_garch.params['omega'] / scale_f**2
alpha_garch = res_garch.params['alpha[1]'] / scale_f**2
beta_garch = res_garch.params['beta[1]'] / scale_f**2

with open('reports/' + ticker + '-factor_GARCH.txt', 'w+') as fh:
    fh.write(res_garch.summary().as_text())

res_garch = pd.DataFrame(index=['mu', 'omega', 'alpha', 'beta'],
                         data=[mu_garch, omega_garch, alpha_garch, beta_garch],
                         columns=['param'])


# TARCH(1, 1, 1) on factors
tarch = arch_model(df['f'].diff().dropna(), p=1, o=1, q=1, rescale=False)
res_tarch = tarch.fit()
resid = res_tarch.resid / scale_f
sigma_tarch = res_tarch.conditional_volatility / scale_f
epsi_tarch = np.divide(resid, sigma_tarch)
mu_tarch = res_tarch.params['mu'] / scale_f
omega_tarch = res_tarch.params['omega'] / scale_f**2
alpha_tarch = res_tarch.params['alpha[1]'] / scale_f**2
gamma_tarch = res_tarch.params['gamma[1]'] / scale_f**2
beta_tarch = res_tarch.params['beta[1]'] / scale_f**2

with open('reports/' + ticker + '-factor_TARCH.txt', 'w+') as fh:
    fh.write(res_tarch.summary().as_text())

res_tarch = pd.DataFrame(index=['mu', 'omega', 'alpha', 'gamma', 'beta', 'c'],
                         data=[mu_tarch, omega_tarch, alpha_tarch, gamma_tarch,
                               beta_tarch, 0],
                         columns=['param'])


# AR-TARCH(1, 1, 1) on factors
ar_tarch = ARX(df['f'], lags=1, rescale=False)
ar_tarch.volatility = GARCH(p=1, o=1, q=1)
res_ar_tarch = ar_tarch.fit()
resid = res_ar_tarch.resid / scale_f
sigma_ar_tarch = res_ar_tarch.conditional_volatility / scale_f
epsi_ar_tarch = np.divide(resid, sigma_ar_tarch)
Phi_ar_tarch = 1 - res_ar_tarch.params['f[1]']
mu_ar_tarch = res_ar_tarch.params['Const'] / scale_f
omega_ar_tarch = res_ar_tarch.params['omega'] / scale_f**2
alpha_ar_tarch = res_ar_tarch.params['alpha[1]'] / scale_f**2
gamma_ar_tarch = res_ar_tarch.params['gamma[1]'] / scale_f**2
beta_ar_tarch = res_ar_tarch.params['beta[1]'] / scale_f**2

with open('reports/' + ticker + '-factor_AR_TARCH.txt', 'w+') as fh:
    fh.write(res_ar_tarch.summary().as_text())

res_ar_tarch = pd.DataFrame(index=['mu', 'B', 'omega', 'alpha', 'gamma',
                                   'beta', 'c'],
                            data=[mu_ar_tarch, 1 - Phi_ar_tarch,
                                  omega_ar_tarch, alpha_ar_tarch,
                                  gamma_ar_tarch, beta_ar_tarch, 0],
                            columns=['param'])


# ------------------------------------- Output --------------------------------

# ---------- Calibration parameters
writer = pd.ExcelWriter('data/calibration_parameters.xlsx')
workbook = writer.book

# write sheets
worksheet = workbook.add_worksheet('calibration-parameters')
writer.sheets['calibration-parameters'] = worksheet
calibration_parameters.to_excel(writer, sheet_name='calibration-parameters')

writer.close()


# ---------- Return dynamics
writer = pd.ExcelWriter('data/return_calibrations.xlsx')
workbook = writer.book

# write sheets
worksheet = workbook.add_worksheet('linear')
writer.sheets['linear'] = worksheet
res_linear.to_excel(writer, sheet_name='linear')

worksheet = workbook.add_worksheet('non-linear')
writer.sheets['non-linear'] = worksheet
res_non_linear.to_excel(writer, sheet_name='non-linear')

writer.close()


# ---------- Factor dynamics
writer = pd.ExcelWriter('data/factor_calibrations.xlsx')
workbook = writer.book

# write sheets
worksheet = workbook.add_worksheet('AR')
writer.sheets['AR'] = worksheet
res_ar.to_excel(writer, sheet_name='AR')

worksheet = workbook.add_worksheet('SETAR')
writer.sheets['SETAR'] = worksheet
res_setar.to_excel(writer, sheet_name='SETAR')

worksheet = workbook.add_worksheet('GARCH')
writer.sheets['GARCH'] = worksheet
res_garch.to_excel(writer, sheet_name='GARCH')

worksheet = workbook.add_worksheet('TARCH')
writer.sheets['TARCH'] = worksheet
res_tarch.to_excel(writer, sheet_name='TARCH')

worksheet = workbook.add_worksheet('AR-TARCH')
writer.sheets['AR-TARCH'] = worksheet
res_ar_tarch.to_excel(writer, sheet_name='AR-TARCH')

writer.close()
