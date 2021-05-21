# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:35:31 2021

@author: Giorgi
"""

import numpy as np
import matplotlib.pyplot as plt

# Seed
np.random.seed(19051991)

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
