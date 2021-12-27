# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 18:59:47 2021

@author: feder
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


# ------------------------------------- Parameters ----------------------------

j_oos = 1000
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
bound = load('data/bound.joblib')


# ------------------------------------- Simulations ---------------------------

# Instantiate market
market = instantiate_market(returnDynamicsType, factorDynamicsType, startPrice)

# Simulations
price, pnl, f = simulate_market(market, j_episodes=j_oos, n_batches=1, t_=t_)

price = price.squeeze()
pnl = pnl.squeeze()
f = f.squeeze()

Sigma_r = get_Sigma(market)
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


# ------------------------------------- RL ------------------------------------

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, qb_list, flag_qaverage=True, n_models=None)


aa = np.linspace(-2, 2, 50)

plt.figure()
for n in np.linspace(-1, 1, 100):
    state = [n, f.flatten()[np.random.randint(f.flatten().shape[0])]]
    qq = np.zeros(len(aa))
    for i in range(len(qq)):
        qq[i] = q_value(state, aa[i])
    plt.plot(aa, qq, label='n=%.3f, f=%.3f' % (state[0], state[1]), alpha=0.5)
plt.savefig('figures/qvalue.png')
