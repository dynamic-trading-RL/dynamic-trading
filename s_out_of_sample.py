# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:32:40 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from joblib import load
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt
from dt_functions import (compute_markovitz,
                          compute_GP,
                          compute_rl,
                          compute_wealth,
                          get_dynamics_params,
                          perform_ttest,
                          perform_linear_regression)
from market import get_Sigma
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


if __name__ == '__main__':

    # ------------------------------------- Parameters ------------------------

    j_oos = 10000
    t_ = 50

    riskDriverDynamicsType = load('data_tmp/riskDriverDynamicsType.joblib')
    factorDynamicsType = load('data_tmp/factorDynamicsType.joblib')

    # ------- Implied parameters

    calibration_parameters = pd.read_excel('data_tmp/calibration_parameters.xlsx',
                                           index_col=0)
    start_price = calibration_parameters.loc['start_price',
                                            'calibration-parameters']

    n_batches = load('data_tmp/n_batches.joblib')
    optimizers = load('data_tmp/optimizers.joblib')
    optimizer = load('data_tmp/optimizer.joblib')
    lam = load('data_tmp/lam.joblib')
    gamma = load('data_tmp/gamma.joblib')
    rho = load('data_tmp/rho.joblib')
    factorType = load('data_tmp/factorType.joblib')
    flag_qaverage = load('data_tmp/flag_qaverage.joblib')
    bound = load('data_tmp/bound.joblib')
    rescale_n_a = load('data_tmp/rescale_n_a.joblib')
    return_is_pnl = load('data_tmp/return_is_pnl.joblib')
    parallel_computing = load('data_tmp/parallel_computing.joblib')
    n_cores = load('data_tmp/n_cores.joblib')

    # ------------------------------------- Simulations -----------------------

    # Instantiate market
    market = instantiate_market(riskDriverDynamicsType, factorDynamicsType,
                                start_price, return_is_pnl)

    Sigma = market.get_sig()
    Lambda = lam*Sigma

    B, mu_r, Phi, mu_f = get_dynamics_params(market)

    # Simulations
    price, pnl, f = market.extract_simulations(j_episodes=j_oos, n_batches=1, t_=t_)  # ??? works differently now

    price = price.squeeze()
    pnl = pnl.squeeze()
    f = f.squeeze()

    # ------------------------------------- Markowitz -------------------------

    Markowitz = compute_markovitz(f, gamma, B, Sigma, price, mu_r,
                                  return_is_pnl)

    wealth_M, value_M, cost_M =\
        compute_wealth(pnl, Markowitz, gamma, Lambda, rho, Sigma, price,
                       return_is_pnl)

    # ------------------------------------- GP --------------------------------

    GP = compute_GP(f, gamma, lam, rho, B, Sigma, Phi, price, mu_r,
                    return_is_pnl)

    wealth_GP, value_GP, cost_GP =\
        compute_wealth(pnl, GP, gamma, Lambda, rho, Sigma, price,
                       return_is_pnl)

    # ------------------------------------- RL --------------------------------

    qb_list = []
    for b in range(n_batches):
        qb_list.append(load('supervised_regressors/q%d.joblib' % b))

    RL = np.zeros((j_oos, t_))

    if parallel_computing:

        compute_rl_part = partial(compute_rl, f=f, qb_list=qb_list,
                                  factorType=factorType, optimizers=optimizers,
                                  optimizer=optimizer, bound=bound,
                                  rescale_n_a=rescale_n_a)

        p = mp.Pool(n_cores)
        shares = p.map(compute_rl_part, range(j_oos))
        p.close()
        p.join()
        RL = np.array(shares)

    else:

        for j in range(j_oos):

            RL[j] = compute_rl(j, f=f, qb_list=qb_list,
                               factorType=factorType, optimizers=optimizers,
                               optimizer=optimizer,
                               bound=bound, rescale_n_a=rescale_n_a)

    wealth_RL, value_RL, cost_RL =\
        compute_wealth(pnl, RL, gamma, Lambda, rho, Sigma, price,
                       return_is_pnl)

    # ------------------------------------- Tests -----------------------------

    final_wealth_GP = wealth_GP[:, -1]
    final_wealth_M = wealth_M[:, -1]
    final_wealth_RL = wealth_RL[:, -1]

    final_value_GP = value_GP[:, -1]
    final_value_M = value_M[:, -1]
    final_value_RL = value_RL[:, -1]

    final_cost_GP = cost_GP[:, -1]
    final_cost_M = cost_M[:, -1]
    final_cost_RL = cost_RL[:, -1]


    # ---------------------------------------------------------- Welch's t-test

    perform_ttest(final_wealth_GP, final_wealth_M, final_wealth_RL,
                  final_value_GP, final_value_M, final_value_RL,
                  final_cost_GP, final_cost_M, final_cost_RL)

    # ------------------------------------------------------- linear regression

    linreg_GP_RL = perform_linear_regression(final_wealth_RL, final_wealth_GP)


    # ------------------------------------- Plots -----------------------------

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
    plt.plot(np.diff(GP[0, :]), color='g', label='GP', alpha=0.5)
    plt.plot(np.diff(RL[0, :]), color='r', label='RL', alpha=0.5)
    for j in range(1, min(50, j_oos)):
        plt.plot(np.diff(GP[j, :]), color='g', alpha=0.5)
        plt.plot(np.diff(RL[j, :]), color='r', alpha=0.5)
    plt.legend()
    plt.title('out-of-sample-trades')
    plt.savefig('figures/out-of-sample-trades.png')

    plt.figure()
    plt.scatter(np.diff(GP, axis=1).flatten(),
                np.diff(RL, axis=1).flatten(),
                s = 1)
    xx = np.array([min(np.diff(GP, axis=1).flatten().min(), np.diff(RL, axis=1).flatten().min()),
                   max(np.diff(GP, axis=1).flatten().max(), np.diff(RL, axis=1).flatten().max())])
    plt.plot(xx, xx, color='r', label='45° line')
    plt.legend()
    plt.xlabel('GP trades')
    plt.ylabel('RL trades')
    plt.title('out-of-sample trades')
    plt.savefig('figures/out-of-sample-trades-scatter.png')

    for j in range(min(7, j_oos)):
        plt.figure()
        plt.plot(np.diff(GP[j, :]), color='g', label='GP')
        plt.plot(np.diff(RL[j, :]), color='r', label='RL')
        plt.title('out-of-sample-trades %d' % j)
        plt.legend()
        plt.savefig('figures/out-of-sample-trades-%d.png' % j)

    plt.figure()
    plt.hist(np.diff(RL, axis=1).flatten(),
             color='r', density=True,
             alpha=0.5, label='RL', bins='auto')
    plt.hist(np.diff(GP, axis=1).flatten(),
             color='g', density=True,
             alpha=0.5, label='GP', bins='auto')
    plt.title('trade')
    plt.legend()
    plt.savefig('figures/trades-histogram.png')

    plt.figure()
    plt.plot(wealth_M[0], color='m', label='Markowitz', alpha=0.5)
    plt.plot(wealth_GP[0], color='g', label='GP', alpha=0.5)
    plt.plot(wealth_RL[0], color='r', label='RL', alpha=0.5)
    for j in range(1, min(50, j_oos)):
        plt.plot(wealth_M[j], color='m', alpha=0.5)
        plt.plot(wealth_GP[j], color='g', alpha=0.5)
        plt.plot(wealth_RL[j], color='r', alpha=0.5)
    plt.title('wealth')
    plt.legend()
    plt.savefig('figures/wealth.png')

    plt.figure()
    plt.hist(wealth_M[:, -1], color='m', density=True,
             alpha=0.5, label='Markowitz', bins='auto')
    plt.hist(wealth_GP[:, -1], color='g', density=True,
             alpha=0.5, label='GP', bins='auto')
    plt.hist(wealth_RL[:, -1], color='r', density=True,
             alpha=0.5, label='RL', bins='auto')
    plt.title('final-wealth')
    results_str = 'Markovitz (mean, std) = (' +\
        '{:.2f}'.format(np.mean(wealth_M[:, -1])).format('.2f') + ', ' +\
        '{:.2f}'.format(np.std(wealth_M[:, -1])) + ') \n' +\
        'RL (mean, std) = (' +\
        '{:.2f}'.format(np.mean(wealth_RL[:, -1])).format('.2f') + ', ' +\
        '{:.2f}'.format(np.std(wealth_RL[:, -1])) + ')\n' +\
        'GP (mean, std) = (' +\
        '{:.2f}'.format(np.mean(wealth_GP[:, -1])).format('.2f') + ', ' +\
        '{:.2f}'.format(np.std(wealth_GP[:, -1])) + ')'
    plt.annotate(results_str, xy=(0, 1), xytext=(12, -12), va='top',
                 xycoords='axes fraction', textcoords='offset points')
    plt.legend(loc='upper right')
    plt.savefig('figures/final-wealth.png')

    plt.figure()
    plt.plot(cost_M[0], color='m', label='Markowitz', alpha=0.5)
    plt.plot(cost_GP[0], color='g', label='GP', alpha=0.5)
    plt.plot(cost_RL[0], color='r', label='RL', alpha=0.5)
    for j in range(1, min(50, j_oos)):
        plt.plot(cost_M[j], color='m', alpha=0.5)
        plt.plot(cost_GP[j], color='g', alpha=0.5)
        plt.plot(cost_RL[j], color='r', alpha=0.5)
    plt.title('cost')
    plt.legend()
    plt.savefig('figures/cost.png')

    plt.figure()
    plt.plot(value_M[0], color='m', label='Markowitz', alpha=0.5)
    plt.plot(value_GP[0], color='g', label='GP', alpha=0.5)
    plt.plot(value_RL[0], color='r', label='RL', alpha=0.5)
    for j in range(1, min(50, j_oos)):
        plt.plot(value_M[j], color='m', alpha=0.5)
        plt.plot(value_GP[j], color='g', alpha=0.5)
        plt.plot(value_RL[j], color='r', alpha=0.5)
    plt.title('value')
    plt.legend()
    plt.savefig('figures/value.png')

    plt.figure()
    plt.scatter(wealth_GP[:, -1], wealth_RL[:, -1], s=1)
    xx = np.array([min(wealth_GP[:, -1].min(), wealth_RL[:, -1].min()),
                   max(wealth_GP[:, -1].max(), wealth_RL[:, -1].max())])
    plt.plot(xx, xx, color='r', label='45° line')
    plt.plot(xx, linreg_GP_RL.params[0] + linreg_GP_RL.params[1]*xx,
             label='%.2f + %.2f x' % (linreg_GP_RL.params[0],
                                      linreg_GP_RL.params[1]))

    xlim = [min(np.quantile(wealth_GP[:, -1], 0.05),
                np.quantile(wealth_RL[:, -1], 0.05)),
            max(np.quantile(wealth_GP[:, -1], 0.95),
                np.quantile(wealth_RL[:, -1], 0.95))]
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.title('Final wealth: GP vs RL')
    plt.xlabel('GP')
    plt.ylabel('RL')
    plt.legend()
    plt.savefig('figures/scatter_GPvsRL.png')

    print('#### END')
