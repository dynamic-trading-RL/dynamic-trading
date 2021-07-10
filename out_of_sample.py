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
import multiprocessing as mp
from functools import partial
from dt_functions import (simulate_market, q_hat, compute_markovitz,
                          compute_optimal, compute_rl, compute_wealth)
import matplotlib.pyplot as plt


print('######## Out of sample')

# ------------------------------------- Parameters ----------------------------

j_ = 10000  # number of out-of-sample paths
optimizer = None
parallel_computing = True  # set to True if you want to use parallel computing
n_cores_max = 80               # maximum number of cores if parallel_computing

# Import parameters from previous scripts
nonlinear = load('data/nonlinear.joblib')

t_ = load('data/t_.joblib')


B = load('data/B.joblib')
mu_u = load('data/mu_u.joblib')
Sigma = load('data/Sigma.joblib')

reg_pol = load('data/reg_pol.joblib')
B_list_fitted = load('data/B_list_fitted.joblib')
sig_pol_fitted = load('data/sig_pol_fitted.joblib')

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

if parallel_computing:
    print('Number of cores available: %d' % mp.cpu_count())
    n_cores = min(mp.cpu_count(), n_cores_max)
    print('Number of cores used: %d' % n_cores)


# ------------------------------------- Simulate ------------------------------

# Simulate market
if nonlinear:
    r, f = simulate_market(j_, t_, 1, 0, 0, 0, Phi, mu_eps, Omega,
                           nonlinear=nonlinear,
                           nonlineartype='polynomial',
                           nn=None, sig_nn=None,
                           B_list=B_list_fitted, sig_pol=sig_pol_fitted)
else:
    r, f = simulate_market(j_, t_, 1, B, mu_u, Sigma, Phi, mu_eps, Omega,
                           nonlinear=nonlinear,
                           nonlineartype=None,
                           nn=None, sig_nn=None,
                           B_list=None, sig_pol=None)


# Markovitz portfolio
print('#### Computing Markovitz strategy')
Markovitz = compute_markovitz(f, gamma, B, Sigma)


# Optimal portfolio
print('#### Computing optimal strategy')
x = compute_optimal(f, gamma, Lambda, rho, B, Sigma, Phi)


# RL portfolio
print('##### Computing RL strategy')

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, qb_list, flag_qaverage=False, n_models=None)


if parallel_computing:
    if __name__ == '__main__':

        compute_rl_part = partial(compute_rl, f=f, q_value=q_value,
                                  lot_size=lot_size, optimizers=optimizers,
                                  optimizer=optimizer)

        p = mp.Pool(n_cores)
        shares = p.map(compute_rl_part, range(j_))
        p.close()
        p.join()
    shares = np.array(shares)

else:
    shares = np.zeros((j_, t_))
    for j in range(j_):
        print('Simulation', j+1, 'on', j_)
        shares[j, :] = compute_rl(j, f, q_value, lot_size, optimizers,
                                  optimizer=optimizer)

# Wealth
if nonlinear:
    sig = sig_pol_fitted
else:
    sig = Sigma

wealth_opt, value_opt, cost_opt = compute_wealth(r, x, gamma, Lambda, rho, sig)

wealth_m, value_m, cost_m = compute_wealth(r, Markovitz, gamma, Lambda, rho, sig)

wealth_rl, value_rl, cost_rl = compute_wealth(r, shares, gamma, Lambda, rho, sig)


# ------------------------------------- Plots ---------------------------------

plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

plt.hist(wealth_m[:, -1], 90, label='Markovitz', density=True, alpha=0.5)
plt.hist(wealth_rl[:, -1], 90, label='RL', density=True, alpha=0.5)
plt.hist(wealth_opt[:, -1], 90, label='Optimal', density=True, alpha=0.5)

results_str = 'Markovitz (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_m[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_m[:, -1])) + ') \n' +\
    'RL (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_rl[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_rl[:, -1])) + ')\n' +\
    'Optimal (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_opt[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_opt[:, -1])) + ')'

plt.annotate(results_str, xy=(0, 1), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points')
plt.legend(loc='upper right')
plt.title('Total wealth')

plt.savefig('figures/out-of-sample.png')
