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


ticker = '^GSPC'  # '^GSPC'
end_date = '2018-10-01'
t_ = 50
c = 0.
scale = 1
scale_f = 1  # or "= scale"
t_past = 8000
window = 5

# ------------------------------------- Download data -------------------------

stock = yf.download(ticker, end=end_date)['Adj Close']
first_valid_loc = stock.first_valid_index()
stock = stock.loc[first_valid_loc:]
startPrice = stock.iloc[-1]
stock.name = ticker
stock = stock.to_frame()
df = stock.copy().iloc[-t_past:]

# ------------------------------------- Get return ----------------------------

df['r'] = scale*df[ticker].pct_change()
# NB: returns and log returns are almost equal

# ------------------------------------- Get PnL -------------------------------

df['pnl'] = scale*df[ticker].diff()

# ------------------------------------- Factors -------------------------------

df['f_r'] = df['r'].rolling(window).mean() / df['r'].rolling(window).std()
df['f_pnl'] = df['pnl'].rolling(window).mean() / df['pnl'].rolling(window).std()

# ------------------------------------- Dump data -----------------------------

df.dropna(inplace=True)
df.to_csv('data/df.csv')
dump(ticker, 'data/ticker.joblib')

calibration_parameters = pd.DataFrame(index=['ticker', 'end_date',
                                             'startPrice', 't_past',
                                             'window'],
                                      data=[ticker, end_date, startPrice,
                                            t_past, window],
                                      columns=['calibration-parameters'])

writer = pd.ExcelWriter('data/calibration_parameters.xlsx')
workbook = writer.book

# write sheets
worksheet = workbook.add_worksheet('calibration-parameters')
writer.sheets['calibration-parameters'] = worksheet
calibration_parameters.to_excel(writer, sheet_name='calibration-parameters')

writer.close()

# ------------------------------------- Fit of dynamics -----------------------
# hold-out
df = df.iloc[:-t_].copy()


# ------------------ RETURNS

# Linear prediction
df_reg = df[['f_r', 'r']].copy()
df_reg['r'] = df_reg['r'].shift(-1)
df_reg.dropna(inplace=True)

reg = OLS(df_reg['r'], add_constant(df_reg['f_r'])).fit()

B = reg.params['f_r']
mu_u = reg.params['const'] / scale
Sigma2_u = reg.mse_resid / scale**2

# ---------- OUTPUT

with open('reports/' + ticker + '-return_linear.txt', 'w+') as fh:
    fh.write(reg.summary().as_text())

res_linear = pd.DataFrame(index=['mu', 'B', 'sig2'],
                          data=[mu_u, B, Sigma2_u],
                          columns=['param'])


writer = pd.ExcelWriter('data/return_calibrations.xlsx')
workbook = writer.book

# write sheets
worksheet = workbook.add_worksheet('linear')
writer.sheets['linear'] = worksheet
res_linear.to_excel(writer, sheet_name='linear')

writer.close()


# ------------------ PnL

# Linear prediction
df_reg = df[['f_pnl', 'pnl']].copy()
df_reg['pnl'] = df_reg['pnl'].shift(-1)
df_reg.dropna(inplace=True)

reg = OLS(df_reg['pnl'], add_constant(df_reg['f_pnl'])).fit()

B = reg.params['f_pnl']
mu_u = reg.params['const'] / scale
Sigma2_u = reg.mse_resid / scale**2

# ---------- OUTPUT

with open('reports/' + ticker + '-pnl_linear.txt', 'w+') as fh:
    fh.write(reg.summary().as_text())

res_linear = pd.DataFrame(index=['mu', 'B', 'sig2'],
                          data=[mu_u, B, Sigma2_u],
                          columns=['param'])


writer = pd.ExcelWriter('data/pnl_calibrations.xlsx')
workbook = writer.book

# write sheets
worksheet = workbook.add_worksheet('linear')
writer.sheets['linear'] = worksheet
res_linear.to_excel(writer, sheet_name='linear')

writer.close()


# ------------------ FACTORS - RETURNS

# AR(1) on factors
res_ar = AutoReg(df['f_r'], lags=1, old_names=False).fit()
mu_ar = res_ar.params.iloc[0] / scale_f
Phi_ar = 1 - res_ar.params.iloc[1]
Omega_ar = res_ar.sigma2 / scale_f**2
epsi_ar =\
    df['f_r'].iloc[1:] / scale_f - Phi_ar*df['f_r'].iloc[:-1] / scale_f - mu_ar

# ---------- OUTPUT

with open('reports/' + ticker + '-factor_r_AR.txt', 'w+') as fh:
    fh.write(res_ar.summary().as_text())

res_ar = pd.DataFrame(index=['mu', 'B', 'sig2'],
                      data=[mu_ar, 1 - Phi_ar, Omega_ar],
                      columns=['param'])

writer = pd.ExcelWriter('data/factor_r_calibrations.xlsx')
workbook = writer.book

# write sheets

worksheet = workbook.add_worksheet('AR')
writer.sheets['AR'] = worksheet
res_ar.to_excel(writer, sheet_name='AR')

writer.close()


# ------------------ FACTORS - PNL

# AR(1) on factors
res_ar = AutoReg(df['f_pnl'], lags=1, old_names=False).fit()
mu_ar = res_ar.params.iloc[0] / scale_f
Phi_ar = 1 - res_ar.params.iloc[1]
Omega_ar = res_ar.sigma2 / scale_f**2
epsi_ar =\
    df['f_pnl'].iloc[1:] / scale_f - Phi_ar*df['f_pnl'].iloc[:-1] / scale_f - mu_ar

# ---------- OUTPUT

with open('reports/' + ticker + '-factor_pnl_AR.txt', 'w+') as fh:
    fh.write(res_ar.summary().as_text())

res_ar = pd.DataFrame(index=['mu', 'B', 'sig2'],
                      data=[mu_ar, 1 - Phi_ar, Omega_ar],
                      columns=['param'])

writer = pd.ExcelWriter('data/factor_pnl_calibrations.xlsx')
workbook = writer.book

# write sheets

worksheet = workbook.add_worksheet('AR')
writer.sheets['AR'] = worksheet
res_ar.to_excel(writer, sheet_name='AR')

writer.close()