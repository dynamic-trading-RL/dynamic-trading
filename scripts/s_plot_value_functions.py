# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:43:45 2022

@author: feder
"""

import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from dt_functions import (get_q_value, maxAction, get_dynamics_params)
from market import instantiate_market, get_Sigma

############ PARAMETERS

n_batches = load('../data/data_tmp/n_batches.joblib')
bound = load('../data/data_tmp/bound.joblib')
rescale_n_a = load('../data/data_tmp/rescale_n_a.joblib')
optimizers = load('../data/data_tmp/optimizers.joblib')
optimizer = load('../data/data_tmp/optimizer.joblib')
riskDriverDynamicsType = load('../data/data_tmp/riskDriverDynamicsType.joblib')
factorDynamicsType = load('../data/data_tmp/factorDynamicsType.joblib')
lam = load('../data/data_tmp/lam.joblib')
gamma = load('../data/data_tmp/gamma.joblib')
rho = load('../data/data_tmp/rho.joblib')
factorType = load('../data/data_tmp/factorType.joblib')
flag_qaverage = load('../data/data_tmp/flag_qaverage.joblib')
return_is_pnl = load('../data/data_tmp/return_is_pnl.joblib')


N = 50

# shares space
nn = np.linspace(-10, 10, N)

# factor space
ff = np.linspace(-2, 2, N)


############ RL

if rescale_n_a:
    resc_n_a = bound
else:
    resc_n_a = 1.

qb_list = []
for b in range(n_batches):
    qb_list.append(load('supervised_regressors/q%d.joblib' % b))

q_value = get_q_value(1, qb_list, flag_qaverage=True)

# next share
nn_RL = np.zeros((len(nn), len(ff)))

for i in range(len(nn)):
    for j in range(len(ff)):

        state = [nn[i]/resc_n_a, ff[j]]
        lb = -bound / resc_n_a - state[0]
        ub = bound / resc_n_a - state[0]
        a = maxAction(q_value, state, [lb, ub], 1, optimizers,
                      optimizer=optimizer)

        nn_RL[j, i] = (state[0] + a) * resc_n_a

############ GP

# Instantiate market
market = instantiate_market(riskDriverDynamicsType, factorDynamicsType,
                            100., return_is_pnl)

Sigma = market.get_sig()

B, mu_r, Phi, mu_f = get_dynamics_params(market)

nn_GP = np.zeros((len(nn), len(ff)))

for i in range(len(nn)):
    for j in range(len(ff)):

        resc_f = ff[j] + mu_r/B
        resc_Sigma = Sigma

        a = (-(gamma*(1 - rho) + lam*rho) +
             np.sqrt((gamma*(1-rho) + lam*rho)**2 +
                     4*gamma*lam*(1-rho)**2)) / (2*(1-rho))

        aim_t = (gamma*resc_Sigma)**(-1) * (B/(1+Phi*a/gamma))*resc_f

        nn_GP[j, i] = (1 - a/lam)*nn[i] + a/lam * aim_t


########## PLOTS

vmin = min(np.min(nn_GP), np.min(nn_RL))
vmax = max(np.max(nn_GP), np.max(nn_RL))

fig,ax = plt.subplots()
contourf_ = ax.contourf(nn, ff, nn_RL, levels=100, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(contourf_)
plt.xlabel('$n_{t-1}$')
plt.ylabel('$f_t$')
plt.title('Shares RL')

fig,ax = plt.subplots()
contourf_ = ax.contourf(nn, ff, nn_GP, levels=100, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(contourf_)
plt.xlabel('$n_{t-1}$')
plt.ylabel('$f_t$')
plt.title('Shares GP')
