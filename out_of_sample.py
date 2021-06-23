# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:41:04 2021

@author: Giorgi
"""

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
from joblib import load
from scipy.stats import iqr
from dt_functions import (simulate_market, q_hat, compute_markovitz,
                          compute_optimal, compute_rl, compute_wealth)
import matplotlib.pyplot as plt


print('######## Out of sample')

# ------------------------------------- Parameters ----------------------------

j_ = 1000  # number of out-of-sample paths

# Import parameters from previous scripts
t_ = load('data/t_.joblib')
B = load('data/B.joblib')
mu_u = load('data/mu_u.joblib')
Sigma = load('data/Sigma.joblib')
Phi = load('data/Phi.joblib')
mu_eps = load('data/mu_eps.joblib')
Omega = load('data/Omega.joblib')
Lambda = load('data/Lambda.joblib')
lam = load('data/lam.joblib')
gamma = load('data/gamma.joblib')
rho = load('data/rho.joblib')
n_batches = load('data/n_batches.joblib')
lot_size = load('data/lot_size.joblib')
optimizers = load('data/optimizers.joblib')


# ------------------------------------- Simulate ------------------------------

# Simulate market
r, f = simulate_market(j_, t_, 1, B, mu_u, Sigma, Phi, mu_eps, Omega)

# Markovitz portfolio
print('#### Computing Markovitz strategy')
Markovitz = compute_markovitz(f, gamma, B, Sigma)


# Optimal portfolio
print('#### Computing optimal strategy')
x = compute_optimal(f, gamma, lam, rho, B, Sigma, Phi)


# RL portfolio
print('##### Computing RL strategy')

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, B, qb_list, flag_qaverage=False, n_models=None)


shares = compute_rl(f, q_value, lot_size, optimizers, optimizer=None)

# Wealth
wealth_opt, value_opt, cost_opt = compute_wealth(r, x, gamma, Lambda, rho, B,
                                                 Sigma, Phi)

wealth_m, value_m, cost_m = compute_wealth(r, Markovitz, gamma, Lambda, rho, B,
                                           Sigma, Phi)

wealth_rl, value_rl, cost_rl = compute_wealth(r, shares, gamma, Lambda, rho, B,
                                              Sigma, Phi)


# ------------------------------------- Plots ---------------------------------

plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

plt.hist(wealth_m[:, -1], 50, label='Markovitz', alpha=0.5)
plt.hist(wealth_rl[:, -1], 50, label='RL', alpha=0.5)
plt.hist(wealth_opt[:, -1], 50, label='Optimal', alpha=0.5)

results_str = 'Markovitz (med, iqr) = (' +\
    '{:.2f}'.format(np.median(wealth_m[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(iqr(wealth_m[:, -1])) + ') \n' +\
    'RL (med, iqr) = (' +\
    '{:.2f}'.format(np.median(wealth_rl[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(iqr(wealth_rl[:, -1])) + ')\n' +\
    'Optimal (med, iqr) = (' +\
    '{:.2f}'.format(np.median(wealth_opt[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(iqr(wealth_opt[:, -1])) + ')'

plt.annotate(results_str, xy=(0, 1), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points')
plt.legend(loc='upper right')
plt.title('Total wealth')

plt.savefig('figures/out-of-sample.png')
