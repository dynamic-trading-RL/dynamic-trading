# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:41:04 2021

@author: Giorgi
"""

import sys
from statsmodels.tools import add_constant
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.regression.linear_model import OLS
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

np.random.seed(7890)


print('######## Out of sample')


# ------------------------------------- Parameters ----------------------------

j_ = 2  # number of out-of-sample paths
optimizer = None
parallel_computing = False  # set to True if you want to use parallel computing
n_cores_max = 20               # maximum number of cores if parallel_computing

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

if parallel_computing:
    print('Number of cores available: %d' % mp.cpu_count())
    n_cores = min(mp.cpu_count(), n_cores_max)
    print('Number of cores used: %d' % n_cores)


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
    print(shares.shape)

else:
    shares = np.zeros((j_, t_))
    for j in range(j_):
        print('Computing episode ', j+1, ' on ', j_)
        shares[j] = compute_rl(j, f, q_value, lot_size, optimizers,
                               optimizer=optimizer)


# Wealth
wealth_opt, value_opt, cost_opt = compute_wealth(r, x, gamma, Lambda, rho, B,
                                                 Sigma, Phi)

wealth_m, value_m, cost_m = compute_wealth(r, Markovitz, gamma, Lambda, rho, B,
                                           Sigma, Phi)

wealth_rl, value_rl, cost_rl = compute_wealth(r, shares, gamma, Lambda, rho, B,
                                              Sigma, Phi)


# ------------------------------------- Tests ---------------------------------

# # --------------------------------------------------------- Welch's t-test

# # ------------------- ABSOLUTE

final_wealth_opt = wealth_opt[:, -1]
final_wealth_m = wealth_m[:, -1]
final_wealth_rl = wealth_rl[:, -1]

final_value_opt = value_opt[:, -1]
final_value_m = value_m[:, -1]
final_value_rl = value_rl[:, -1]

final_cost_opt = cost_opt[:, -1]
final_cost_m = cost_m[:, -1]
final_cost_rl = cost_rl[:, -1]


t_wealth_opt_m = ttest_ind(final_wealth_opt,
                           final_wealth_m,
                           usevar='unequal',
                           alternative='larger')

t_wealth_rl_m = ttest_ind(final_wealth_rl,
                          final_wealth_m,
                          usevar='unequal',
                          alternative='larger')

t_wealth_rl_opt = ttest_ind(final_wealth_rl,
                            final_wealth_opt,
                            usevar='unequal',
                            alternative='two-sided')

print('\n\n\nWelch\'s tests (absolute):\n')
print('    H0: GPw=Mw, H1: GPw>Mw. t: %.4f, p-value: %.4f' % (t_wealth_opt_m[0],
                                                              t_wealth_opt_m[1]))
print('    H0: RLw=Mw, H1: RLw>Mw. t: %.4f, p-value: %.4f' % (t_wealth_rl_m[0],
                                                              t_wealth_rl_m[1]))
print('    H0: RLw=GPw, H1: RLw!=GPw. t: %.4f, p-value: %.4f' % (t_wealth_rl_opt[0],
                                                                 t_wealth_rl_opt[1]))


t_value_opt_m = ttest_ind(final_value_opt,
                          final_value_m,
                          usevar='unequal',
                          alternative='larger')

t_value_rl_m = ttest_ind(final_value_rl,
                         final_value_m,
                         usevar='unequal',
                         alternative='larger')

t_value_rl_opt = ttest_ind(final_value_rl,
                           final_value_opt,
                           usevar='unequal',
                           alternative='two-sided')

print('\n    H0: GPv=Mv, H1: GPv>Mv. t: %.4f, p-value: %.4f' % (t_value_opt_m[0],
                                                              t_value_opt_m[1]))
print('    H0: RLv=Mv, H1: RLv>Mv. t: %.4f, p-value: %.4f' % (t_value_rl_m[0],
                                                              t_value_rl_m[1]))
print('    H0: RLv=GPv, H1: RLv!=GPv. t: %.4f, p-value: %.4f' % (t_value_rl_opt[0],
                                                                 t_value_rl_opt[1]))


t_cost_opt_m = ttest_ind(final_cost_opt,
                         final_cost_m,
                         usevar='unequal',
                         alternative='smaller')

t_cost_rl_m = ttest_ind(final_cost_rl,
                        final_cost_m,
                        usevar='unequal',
                        alternative='smaller')

t_cost_rl_opt = ttest_ind(final_cost_rl,
                          final_cost_opt,
                          usevar='unequal',
                          alternative='two-sided')

print('\n    H0: GPc=Mc, H1: GPc<Mc. t: %.4f, p-value: %.4f' % (t_cost_opt_m[0],
                                                              t_cost_opt_m[1]))
print('    H0: RLc=Mc, H1: RLc<Mc. t: %.4f, p-value: %.4f' % (t_cost_rl_m[0],
                                                              t_cost_rl_m[1]))
print('    H0: RLc=GPc, H1: RLc!=GPc. t: %.4f, p-value: %.4f' % (t_cost_rl_opt[0],
                                                                 t_cost_rl_opt[1]))


# # ------------------- DIFFERENCES

final_wealth_rl_m = wealth_rl[:, -1] - wealth_m[:, -1]
final_wealth_rl_opt = wealth_rl[:, -1] - wealth_opt[:, -1]


t_wealth_diff_rl_m = ttest_ind(final_wealth_rl_m,
                               np.zeros(len(final_wealth_rl_m)),
                               usevar='unequal',
                               alternative='larger')

t_wealth_diff_rl_opt = ttest_ind(final_wealth_rl_opt,
                                 np.zeros(len(final_wealth_rl_opt)),
                                 usevar='unequal',
                                 alternative='two-sided')

print('\n\nWelch\'s tests (differences):\n')
print('    H0: RLw=Mw, H1: RLw>Mw. t: %.4f, p-value: %.4f' % (t_wealth_diff_rl_m[0],
                                                              t_wealth_diff_rl_m[1]))
print('    H0: RLw=GPw, H1: RLw!=GPw. t: %.4f, p-value: %.4f' % (t_wealth_diff_rl_opt[0],
                                                                 t_wealth_diff_rl_opt[1]))


# # --------------------------------------------------------- linear regression

linreg_opt_rl = OLS(final_wealth_rl, add_constant(final_wealth_opt)).fit()

print('\n\n\n Linear regression of X=GP vs Y=RL:\n')
print(linreg_opt_rl.summary())
print('\n\n H0: beta=1; H1: beta!=1')
print(linreg_opt_rl.t_test(([0., 1.], 1.)))


# ------------------------------------- Plots ---------------------------------

# ## Histograms of final wealth

plt.figure()

plt.hist(wealth_m[:, -1], 90, label='Markovitz', density=True, alpha=0.5)
plt.hist(wealth_rl[:, -1], 90, label='RL', density=True, alpha=0.5)
plt.hist(wealth_opt[:, -1], 90, label='GP', density=True, alpha=0.5)

results_str = 'Markovitz (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_m[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_m[:, -1])) + ') \n' +\
    'RL (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_rl[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_rl[:, -1])) + ')\n' +\
    'GP (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_opt[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_opt[:, -1])) + ')'

plt.annotate(results_str, xy=(0, 1), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points')
plt.legend(loc='upper right')
plt.title('Total wealth')

plt.savefig('figures/out-of-sample.png')


# ## Scatter plots of final wealth

# GP vs RL

plt.figure()

plt.scatter(wealth_opt[:, -1], wealth_rl[:, -1], s=1)
xx = np.array([min(wealth_opt[:, -1].min(), wealth_rl[:, -1].min()),
               max(wealth_opt[:, -1].max(), wealth_rl[:, -1].max())])
plt.plot(xx, xx, color='r', label='45° line')
plt.plot(xx, linreg_opt_rl.params[0] + linreg_opt_rl.params[1]*xx,
         label='%.2f + %.2f x'%(linreg_opt_rl.params[0], linreg_opt_rl.params[1]))

xlim = [min(np.quantile(wealth_opt[:, -1], 0.05), np.quantile(wealth_rl[:, -1],
                                                              0.05)),
        max(np.quantile(wealth_opt[:, -1], 0.95), np.quantile(wealth_rl[:, -1],
                                                              0.95))]
plt.xlim(xlim)
plt.ylim(xlim)
plt.title('Final wealth: GP vs RL')
plt.xlabel('GP')
plt.ylabel('RL')
plt.legend()
plt.savefig('figures/scatter_GPvsRL.png')


# ## Histograms of differences

plt.figure()

plt.hist(wealth_rl[:, -1] - wealth_m[:, -1], 90, label='RL - Markovitz',
         density=True, alpha=0.5)
plt.hist(wealth_rl[:, -1] - wealth_opt[:, -1], 90, label='RL - Optimal',
         density=True, alpha=0.5)

results_str = 'RL - Markovitz (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_rl[:, -1] - wealth_m[:, -1])).format('.2f')\
    + ', ' + '{:.2f}'.format(np.std(wealth_rl[:, -1] - wealth_m[:, -1])) +\
    ') \n' + 'RL - Optimal (mean, std) = (' +\
    '{:.2f}'.format(np.mean(wealth_rl[:, -1] -
                            wealth_opt[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(wealth_rl[:, -1] - wealth_opt[:, -1])) + ')'

plt.annotate(results_str, xy=(0, 1), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points')
plt.legend(loc='upper right')
plt.title('Wealth differences')

plt.savefig('figures/differences.png')


# ## Histograms of value

plt.figure()

plt.hist(value_m[:, -1], 90, label='Markovitz', density=True, alpha=0.5)
plt.hist(value_rl[:, -1], 90, label='RL', density=True, alpha=0.5)
plt.hist(value_opt[:, -1], 90, label='Optimal', density=True, alpha=0.5)

results_str = 'Markovitz (mean, std) = (' +\
    '{:.2f}'.format(np.mean(value_m[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(value_m[:, -1])) + ') \n' +\
    'RL (mean, std) = (' +\
    '{:.2f}'.format(np.mean(value_rl[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(value_rl[:, -1])) + ')\n' +\
    'Optimal (mean, std) = (' +\
    '{:.2f}'.format(np.mean(value_opt[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(value_opt[:, -1])) + ')'

plt.annotate(results_str, xy=(0, 1), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points')
plt.legend(loc='upper right')
plt.title('Total value')

plt.savefig('figures/out-of-sample-value.png')


# ## Histograms of costs

plt.figure()

plt.hist(cost_m[:, -1], 90, label='Markovitz', density=True, alpha=0.5)
plt.hist(cost_rl[:, -1], 90, label='RL', density=True, alpha=0.5)
plt.hist(cost_opt[:, -1], 90, label='Optimal', density=True, alpha=0.5)

results_str = 'Markovitz (mean, std) = (' +\
    '{:.2f}'.format(np.mean(cost_m[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(cost_m[:, -1])) + ') \n' +\
    'RL (mean, std) = (' +\
    '{:.2f}'.format(np.mean(cost_rl[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(cost_rl[:, -1])) + ')\n' +\
    'Optimal (mean, std) = (' +\
    '{:.2f}'.format(np.mean(cost_opt[:, -1])).format('.2f') + ', ' +\
    '{:.2f}'.format(np.std(cost_opt[:, -1])) + ')'

plt.annotate(results_str, xy=(0, 1), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points')
plt.legend(loc='upper right')
plt.title('Total cost')

plt.savefig('figures/out-of-sample-cost.png')
