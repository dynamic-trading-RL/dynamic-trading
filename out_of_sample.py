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
from joblib import load, dump
from dt_functions import (simulate_market, q_hat, maxAction)
import matplotlib.pyplot as plt


# Import parameters
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


# Simulate market
j_ = 1000
r, f = simulate_market(j_, t_, 1, B, mu_u, Sigma, Phi, mu_eps, Omega)
r = r.squeeze()
f = f.squeeze()


# Markovitz portfolio
Markovitz = np.zeros((j_, t_))
for t in range(t_):
    Markovitz[:, t] = (gamma*Sigma)**(-1)*B*f[:, t]
Markovitz = np.round(Markovitz)


# Optimal portfolio
a = (-(gamma*(1 - rho) + lam*rho) +
     np.sqrt((gamma*(1-rho) + lam*rho)**2 +
             4*gamma*lam*(1-rho)**2)) / (2*(1-rho))

x = np.zeros((j_, t_))
x[:, 0] = Markovitz[:, 0]
for t in range(1, t_):
    x[:, t] = (1 - a/lam)*x[:, t-1] +\
        a/lam * 1/(gamma*Sigma) * (B/(1+Phi*a/gamma))*f[:, t]
x = np.round(x)


# RL portfolio
print('##### Computing RL strategy')

qb_list = []
for b in range(n_batches):
    qb_list.append(load('models/q%d.joblib' % b))


def q_value(state, action):
    return q_hat(state, action, B, qb_list, flag_qaverage=True, n_models=None)


shares = np.zeros((j_, t_))

# empirical found out that differential_evolution is chosen more
# often:
# basinhopping {'n': 386}
# differential_evolution {'n': 507}
# dual_annealing {'n': 436}
# shgo {'n': 496}
# if this is confirmed, comment the above and uncomment the below

for j in range(j_):
    print('Out of sample path: ', j+1, 'on', j_)
    for t in range(t_):
        progress = t/t_*100
        print('    Progress: %.2f %%' % progress)

        if t == 0:
            state = np.array([0, f[j, t]])
            action = maxAction(q_value, state, lot_size, optimizers,
                               optimizer='best')
            shares[j, t] = state[0] + action
        else:
            state = np.array([shares[j, t-1], f[j, t]])
            action = maxAction(q_value, state, lot_size, optimizers,
                               optimizer='best')
            shares[j, t] = state[0] + action


# Value
value = np.zeros((j_, t_))
for t in range(t_ - 1):
    value[:, t] = (1 - rho)**(t + 1) * x[:, t]*f[:, t+1]

value_m = np.zeros((j_, t_))
for t in range(t_ - 1):
    value_m[:, t] = (1 - rho)**(t + 1) * Markovitz[:, t]*f[:, t+1]

value_rl = np.zeros((j_, t_))
for t in range(t_ - 1):
    value_rl[:, t] = (1 - rho)**(t + 1) * shares[:, t]*f[:, t+1]


# Costs
cost = np.zeros((j_, t_))
for t in range(1, t_):
    cost[:, t] = gamma/2 * (1 - rho)**(t + 1)*x[:, t]*Sigma*x[:, t] +\
        (1 - rho)**t/2*(x[:, t] - x[:, t-1])*Lambda*(x[:, t]-x[:, t-1])

cost_m = np.zeros((j_, t_))
for t in range(1, t_):
    cost_m[:, t] = gamma/2 * (1 - rho)**(t + 1)*Markovitz[:, t]*Sigma*Markovitz[:, t] +\
        (1 - rho)**t/2*(Markovitz[:, t] -
                        Markovitz[:, t-1])*Lambda*(Markovitz[:, t]-Markovitz[:, t-1])

cost_rl = np.zeros((j_, t_))
for t in range(1, t_):
    cost_rl[:, t] = gamma/2 * (1 - rho)**(t + 1)*shares[:, t]*Sigma*shares[:, t] +\
        (1 - rho)**t/2*(shares[:, t] -
                        shares[:, t-1])*Lambda*(shares[:, t]-shares[:, t-1])


# Wealth
wealth = value - cost
wealth_m = value_m - cost_m
wealth_rl = value_rl - cost_rl

# Plots

plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

plt.hist(np.sum(wealth_m, axis=1), 50, label='Markovitz', alpha=0.5)
plt.hist(np.sum(wealth_rl, axis=1), 50, label='RL', alpha=0.5)
plt.hist(np.sum(wealth, axis=1), 50, label='Optimal', alpha=0.5)

results_str = 'Markovitz (mean, std) = (' +\
    '{:.2f}'.format(np.mean(np.sum(wealth_m, axis=1))).format('.2f') + ',' +\
    '{:.2f}'.format(np.std(np.sum(wealth_m, axis=1))) + ') \n' +\
    'RL (mean, std) = (' +\
    '{:.2f}'.format(np.mean(np.sum(wealth_rl, axis=1))).format('.2f') + ',' +\
    '{:.2f}'.format(np.std(np.sum(wealth_rl, axis=1))) + ')\n' +\
    'Optimal (mean, std) = (' +\
    '{:.2f}'.format(np.mean(np.sum(wealth, axis=1))).format('.2f') + ',' +\
    '{:.2f}'.format(np.std(np.sum(wealth, axis=1))) + ')'

plt.annotate(results_str, xy=(0, 1), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points')
plt.legend(loc='upper right')
plt.title('Total wealth')

plt.savefig('figures/out-of-sample.png')
