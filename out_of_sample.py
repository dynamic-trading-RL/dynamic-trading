# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:32:40 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
from dt_functions import (ReturnDynamicsType, FactorDynamicsType,
                          instantiate_market,
                          get_Sigma,
                          simulate_market,
                          q_hat,
                          compute_markovitz,
                          compute_GP,
                          compute_rl,
                          compute_wealth,
                          get_dynamics_params)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# ------------------------------------- Parameters ----------------------------


j_oos = 100
t_ = 50

returnDynamicsType = ReturnDynamicsType.Linear
factorDynamicsType = FactorDynamicsType.AR


# ------- Implied parameters

calibration_parameters = pd.read_excel('data/calibration_parameters.xlsx',
                                       index_col=0)
startPrice = calibration_parameters.loc['startPrice', 'calibration-parameters']

n_batches = load('data/n_batches.joblib')
optimizers = load('data/optimizers.joblib')
optimizer = load('data/optimizer.joblib')
lam = load('data/lam.joblib')
gamma = load('data/gamma.joblib')
rho = load('data/rho.joblib')
factorType = load('data/factorType.joblib')
flag_qaverage = load('data/flag_qaverage.joblib')
bound = load('data/bound.joblib')
rescale_n_a = load('data/rescale_n_a.joblib')
return_is_pnl = load('data/return_is_pnl.joblib')


# ------------------------------------- Simulations ---------------------------

# Instantiate market
market = instantiate_market(returnDynamicsType, factorDynamicsType, startPrice,
                            return_is_pnl)

# Simulations
price, pnl, f = simulate_market(market, j_episodes=j_oos, n_batches=1, t_=t_)

price = price.squeeze()
pnl = pnl.squeeze()
f = f.squeeze()

Sigma = get_Sigma(market)
Lambda = lam*Sigma

B, mu_r, Phi, mu_f = get_dynamics_params(market)


# ------------------------------------- Markowitz -----------------------------

Markowitz = compute_markovitz(f, gamma, B, Sigma, price, mu_r, return_is_pnl)

wealth_M, value_M, cost_M =\
    compute_wealth(pnl, Markowitz, gamma, Lambda, rho, Sigma, price,
                   return_is_pnl)


# ------------------------------------- GP ------------------------------------

GP = compute_GP(f, gamma, lam, rho, B, Sigma, Phi, price, mu_r, return_is_pnl)

wealth_GP, value_GP, cost_GP =\
    compute_wealth(pnl, GP, gamma, Lambda, rho, Sigma, price,
                   return_is_pnl)


# ------------------------------------- RL ------------------------------------

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, qb_list, flag_qaverage=flag_qaverage)


RL = np.zeros((j_oos, t_))

for j in range(j_oos):

    RL[j] = compute_rl(j, f=f[j], q_value=q_value, factorType=factorType,
                       optimizers=optimizers, optimizer=optimizer, bound=bound,
                       rescale_n_a=rescale_n_a)

wealth_RL, value_RL, cost_RL =\
    compute_wealth(pnl, RL, gamma, Lambda, rho, Sigma, price,
                   return_is_pnl)


# ------------------------------------- Plots ---------------------------------

plt.figure()
plt.plot(Markowitz[0, :], color='m', label='Markowitz', alpha=0.5)
plt.plot(GP[0, :], color='g', label='GP', alpha=0.5)
plt.plot(RL[0, :], color='r', label='RL', alpha=0.5)
for j in range(1, min(50, j_oos)):
    plt.plot(Markowitz[j, :], color='m', alpha=0.5)
    plt.plot(GP[j, :], color='g', alpha=0.5)
    plt.plot(RL[j, :], color='r', alpha=0.5)
plt.legend()
plt.title('out-of-sample-shares')
plt.savefig('figures/out-of-sample-shares.png')


for j in range(min(7, j_oos)):
    plt.figure()
    plt.plot(Markowitz[j, :], color='m', label='Markowitz')
    plt.plot(GP[j, :], color='g', label='GP')
    plt.plot(RL[j, :], color='r', label='RL')
    plt.title('out-of-sample-shares %d' % j)
    plt.legend()
    plt.savefig('figures/out-of-sample-shares-%d.png' % j)


plt.figure()
plt.plot(np.cumsum(wealth_M[0]), color='m', label='Markowitz', alpha=0.5)
plt.plot(np.cumsum(wealth_GP[0]), color='g', label='GP', alpha=0.5)
plt.plot(np.cumsum(wealth_RL[0]), color='r', label='RL', alpha=0.5)
for j in range(1, min(50, j_oos)):
    plt.plot(np.cumsum(wealth_M[j]), color='m', alpha=0.5)
    plt.plot(np.cumsum(wealth_GP[j]), color='g', alpha=0.5)
    plt.plot(np.cumsum(wealth_RL[j]), color='r', alpha=0.5)
plt.title('wealth')
plt.legend()
plt.savefig('figures/wealth.png')


plt.figure()
plt.hist(np.cumsum(wealth_M, axis=1)[:, -1], color='m', density=True,
         alpha=0.5, label='Markowitz', bins='auto')
plt.hist(np.cumsum(wealth_GP, axis=1)[:, -1], color='g', density=True,
         alpha=0.5, label='GP', bins='auto')
plt.hist(np.cumsum(wealth_RL, axis=1)[:, -1], color='r', density=True,
         alpha=0.5, label='RL', bins='auto')
plt.title('final-wealth')
plt.legend()
plt.savefig('figures/final-wealth.png')


plt.figure()
plt.plot(np.cumsum(cost_M[0]), color='m', label='Markowitz', alpha=0.5)
plt.plot(np.cumsum(cost_GP[0]), color='g', label='GP', alpha=0.5)
plt.plot(np.cumsum(cost_RL[0]), color='r', label='RL', alpha=0.5)
for j in range(1, min(50, j_oos)):
    plt.plot(np.cumsum(cost_M[j]), color='m', alpha=0.5)
    plt.plot(np.cumsum(cost_GP[j]), color='g', alpha=0.5)
    plt.plot(np.cumsum(cost_RL[j]), color='r', alpha=0.5)
plt.title('cost')
plt.legend()
plt.savefig('figures/cost.png')


print('#### END')
