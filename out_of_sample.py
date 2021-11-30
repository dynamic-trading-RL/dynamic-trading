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
                          compute_wealth)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

np.random.seed(7890)

# ------------------------------------- Parameters ----------------------------

j_oos = 1000
t_ = 50

optimizer = 'shgo'

returnDynamicsType = ReturnDynamicsType.Linear
factorDynamicsType = FactorDynamicsType.AR


# ------- Implied parameters

calibration_parameters = pd.read_excel('data/calibration_parameters.xlsx',
                                       index_col=0)
startPrice = calibration_parameters.loc['startPrice', 'calibration-parameters']

n_batches = load('data/n_batches.joblib')
optimizers = load('data/optimizers.joblib')
lam_perc = load('data/lam_perc.joblib')
gamma = load('data/gamma.joblib')
rho = load('data/rho.joblib')

# ------------------------------------- Simulations ---------------------------

# Instantiate market
market = instantiate_market(returnDynamicsType, factorDynamicsType, startPrice)

# Simulations
price, pnl, f = simulate_market(market, j_episodes=j_oos, n_batches=1, t_=t_)

price = price.squeeze()
pnl = pnl.squeeze()
f = f.squeeze()

Sigma_r = get_Sigma(market)
lam = lam_perc / Sigma_r
Lambda_r = lam*Sigma_r

if (market._marketDynamics._returnDynamics._returnDynamicsType
        == ReturnDynamicsType.Linear):
    B = market._marketDynamics._returnDynamics._parameters['B']
    mu_r = market._marketDynamics._returnDynamics._parameters['mu']
else:
    B_0 = market._marketDynamics._returnDynamics._parameters['B_0']
    B_1 = market._marketDynamics._returnDynamics._parameters['B_1']
    mu_0 = market._marketDynamics._returnDynamics._parameters['mu_0']
    mu_1 = market._marketDynamics._returnDynamics._parameters['mu_1']
    B = .5*(B_0 + B_1)
    mu_r = .5*(mu_0 + mu_1)


if (market._marketDynamics._factorDynamics._factorDynamicsType
        in (FactorDynamicsType.AR, FactorDynamicsType.AR_TARCH)):

    Phi = 1 - market._marketDynamics._factorDynamics._parameters['B']
    mu_f = 1 - market._marketDynamics._factorDynamics._parameters['mu']

elif (market._marketDynamics._factorDynamics._factorDynamicsType
      == FactorDynamicsType.SETAR):

    Phi_0 = 1 - market._marketDynamics._factorDynamics._parameters['B_0']
    Phi_1 = 1 - market._marketDynamics._factorDynamics._parameters['B_1']
    mu_f_0 = 1 - market._marketDynamics._factorDynamics._parameters['mu_0']
    mu_f_1 = 1 - market._marketDynamics._factorDynamics._parameters['mu_1']
    Phi = 0.5*(Phi_0 + Phi_1)
    mu_f = .5*(mu_f_0 + mu_f_1)

else:

    Phi = 0.
    mu_f = market._marketDynamics._factorDynamics._parameters['mu']


# ------------------------------------- Markowitz -----------------------------


Markowitz = compute_markovitz((f + mu_r)*price.mean(), gamma, B,
                              Sigma_r*price.mean())
bound = np.abs(Markowitz).max()

wealth_M, value_M, cost_M =\
    compute_wealth(pnl, Markowitz, gamma, Lambda_r*price.mean(), rho,
                   Sigma_r*price.mean())


# ------------------------------------- GP ------------------------------------

GP = compute_GP((f + mu_r)*price.mean(),
                gamma,
                lam,
                rho,
                B,
                Sigma_r*price.mean(),
                Phi)

wealth_GP, value_GP, cost_GP =\
    compute_wealth(pnl, GP, gamma, Lambda_r*price.mean(), rho,
                   Sigma_r*price.mean())


# ------------------------------------- RL ------------------------------------

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, qb_list, flag_qaverage=False, n_models=None)


RL = np.zeros((j_oos, t_))

for j in range(j_oos):

    RL[j] = compute_rl(j, pnl=pnl[j], q_value=q_value, optimizers=optimizers,
                       optimizer=optimizer, bound=bound)

wealth_RL, value_RL, cost_RL =\
    compute_wealth(pnl, RL, gamma, Lambda_r*price.mean(), rho,
                   Sigma_r*price.mean())


# ------------------------------------- Plots ---------------------------------

plt.figure()
for j in range(min(50, j_oos)):
    plt.plot(Markowitz[j, :], color='m', alpha=0.5)
    plt.plot(GP[j, :], color='g', alpha=0.5)
    plt.plot(RL[j, :], color='r', alpha=0.5)
plt.title('out-of-sample-shares')
plt.savefig('figures/out-of-sample-shares.png')


plt.figure()
plt.plot(np.cumsum(wealth_M[0]), color='m', label='Markowitz')
plt.plot(np.cumsum(wealth_GP[0]), color='g', label='GP')
plt.plot(np.cumsum(wealth_RL[0]), color='r', label='RL')
for j in range(min(50, j_oos)):
    plt.plot(np.cumsum(wealth_M[j]), color='m')
    plt.plot(np.cumsum(wealth_GP[j]), color='g')
    plt.plot(np.cumsum(wealth_RL[j]), color='r')
plt.title('wealth')
plt.legend()
plt.savefig('figures/wealth.png')


plt.figure()
plt.plot(np.cumsum(cost_M[0]), color='m', label='Markowitz')
plt.plot(np.cumsum(cost_GP[0]), color='g', label='GP')
plt.plot(np.cumsum(cost_RL[0]), color='r', label='RL')
for j in range(min(50, j_oos)):
    plt.plot(np.cumsum(cost_M[j]), color='m')
    plt.plot(np.cumsum(cost_GP[j]), color='g')
    plt.plot(np.cumsum(cost_RL[j]), color='r')
plt.title('cost')
plt.legend()
plt.savefig('figures/cost.png')


print('#### END')
