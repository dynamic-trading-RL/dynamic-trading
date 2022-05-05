# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:32:40 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
from joblib import load
from functools import partial
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
import multiprocessing as mp
import matplotlib.pyplot as plt
from dt_functions import (instantiate_market,
                          get_Sigma,
                          simulate_market,
                          compute_markovitz,
                          compute_GP,
                          compute_rl,
                          compute_wealth,
                          get_dynamics_params,
                          ReturnDynamics,
                          FactorDynamics,
                          MarketDynamics,
                          Market)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


if __name__ == '__main__':

    # ------------------------------------- Parameters ------------------------

    j_oos = 10000
    t_ = 50

    returnDynamicsType = load('data/returnDynamicsType.joblib')
    factorDynamicsType = load('data/factorDynamicsType.joblib')

    # ------- Implied parameters

    calibration_parameters = pd.read_excel('data/calibration_parameters.xlsx',
                                           index_col=0)
    startPrice = calibration_parameters.loc['startPrice',
                                            'calibration-parameters']

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
    parallel_computing = load('data/parallel_computing.joblib')
    n_cores = load('data/n_cores.joblib')



    # -------------------------- REINFORCEMENT LEARNING -----------------------
    # Instantiate market
    # Instantiate dynamics
    returnDynamics = ReturnDynamics(returnDynamicsType)
    factorDynamics = FactorDynamics(factorDynamicsType)

    # Read calibrated parameters
    return_parameters = pd.read_excel('data/return_calibrations.xlsx',
                                      sheet_name='linear',
                                      index_col=0)['param'].to_dict()
    factor_parameters = pd.read_excel('data/factor_r_calibrations.xlsx',
                                      sheet_name='AR',
                                      index_col=0)['param'].to_dict()

    # Set dynamics
    returnDynamics.set_parameters(return_parameters)
    factorDynamics.set_parameters(factor_parameters)
    marketDynamics = MarketDynamics(returnDynamics=returnDynamics,
                                    factorDynamics=factorDynamics)
    market = Market(marketDynamics, startPrice, return_is_pnl=False)

    Sigma = get_Sigma(market)
    Lambda = lam*Sigma

    B, mu_r, Phi, mu_f = get_dynamics_params(market)

    # Simulations
    price, pnl, f = simulate_market(market, j_episodes=j_oos, n_batches=1,
                                    t_=t_)

    price = price.squeeze()
    pnl = pnl.squeeze()
    f = f.squeeze()

    # Strategy
    qb_list = []
    for b in range(n_batches):
        qb_list.append(load('models/q%d.joblib' % b))

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
                       return_is_pnl=False)

    # -------------------------- GP & MARKOWITZ -------------------------------
    lam_perc = 10**-1

    # Instantiate market
    # Instantiate dynamics
    returnDynamics = ReturnDynamics(returnDynamicsType)
    factorDynamics = FactorDynamics(factorDynamicsType)

    # Read calibrated parameters
    return_parameters = pd.read_excel('data/pnl_calibrations.xlsx',
                                      sheet_name='linear',
                                      index_col=0)['param'].to_dict()
    factor_parameters = pd.read_excel('data/factor_pnl_calibrations.xlsx',
                                      sheet_name='AR',
                                      index_col=0)['param'].to_dict()

    # Set dynamics
    returnDynamics.set_parameters(return_parameters)
    factorDynamics.set_parameters(factor_parameters)
    marketDynamics = MarketDynamics(returnDynamics=returnDynamics,
                                    factorDynamics=factorDynamics)
    market = Market(marketDynamics, startPrice, return_is_pnl=True)

    Sigma = get_Sigma(market)
    lam = lam_perc * 2 / Sigma
    Lambda = lam*Sigma

    B, mu_r, Phi, mu_f = get_dynamics_params(market)

    # Simulations
    price, pnl, f = simulate_market(market, j_episodes=j_oos, n_batches=1,
                                    t_=t_)

    price = price.squeeze()
    pnl = pnl.squeeze()
    f = f.squeeze()

    # Strategy
    Markowitz = compute_markovitz(f, gamma, B, Sigma, price, mu_r,
                                  return_is_pnl=True)

    wealth_M, value_M, cost_M =\
        compute_wealth(pnl, Markowitz, gamma, Lambda, rho, Sigma, price,
                       return_is_pnl=True)

    GP = compute_GP(f, gamma, lam, rho, B, Sigma, Phi, price, mu_r,
                    return_is_pnl=True)

    wealth_GP, value_GP, cost_GP =\
        compute_wealth(pnl, GP, gamma, Lambda, rho, Sigma, price,
                       return_is_pnl=True)


    # ------------------------------------- Tests -----------------------------

    # ---------------------------------------------------------- Welch's t-test

    final_wealth_GP = wealth_GP[:, -1]
    final_wealth_M = wealth_M[:, -1]
    final_wealth_RL = wealth_RL[:, -1]

    final_value_GP = value_GP[:, -1]
    final_value_M = value_M[:, -1]
    final_value_RL = value_RL[:, -1]

    final_cost_GP = cost_GP[:, -1]
    final_cost_M = cost_M[:, -1]
    final_cost_RL = cost_RL[:, -1]

    t_wealth_GP_M = ttest_ind(final_wealth_GP,
                              final_wealth_M,
                              usevar='unequal',
                              alternative='larger')

    t_wealth_RL_M = ttest_ind(final_wealth_RL,
                              final_wealth_M,
                              usevar='unequal',
                              alternative='larger')

    t_wealth_RL_GP = ttest_ind(final_wealth_RL,
                               final_wealth_GP,
                               usevar='unequal',
                               alternative='larger')

    print('\n\n\nWelch\'s tests (absolute):\n')
    print('    H0: GPw=Mw, H1: GPw>Mw. t: %.4f, p-value: %.4f' %
          (t_wealth_GP_M[0], t_wealth_GP_M[1]))
    print('    H0: RLw=Mw, H1: RLw>Mw. t: %.4f, p-value: %.4f' %
          (t_wealth_RL_M[0], t_wealth_RL_M[1]))
    print('    H0: RLw=GPw, H1: RLw>GPw. t: %.4f, p-value: %.4f' %
          (t_wealth_RL_GP[0], t_wealth_RL_GP[1]))

    t_value_GP_M = ttest_ind(final_value_GP,
                             final_value_M,
                             usevar='unequal',
                             alternative='larger')

    t_value_RL_M = ttest_ind(final_value_RL,
                             final_value_M,
                             usevar='unequal',
                             alternative='larger')

    t_value_RL_GP = ttest_ind(final_value_RL,
                              final_value_GP,
                              usevar='unequal',
                              alternative='larger')

    print('\n    H0: GPv=Mv, H1: GPv>Mv. t: %.4f, p-value: %.4f' %
          (t_value_GP_M[0], t_value_GP_M[1]))
    print('    H0: RLv=Mv, H1: RLv>Mv. t: %.4f, p-value: %.4f' %
          (t_value_RL_M[0], t_value_RL_M[1]))
    print('    H0: RLv=GPv, H1: RLv>GPv. t: %.4f, p-value: %.4f' %
          (t_value_RL_GP[0], t_value_RL_GP[1]))

    t_cost_GP_M = ttest_ind(final_cost_GP,
                            final_cost_M,
                            usevar='unequal',
                            alternative='smaller')

    t_cost_RL_M = ttest_ind(final_cost_RL,
                            final_cost_M,
                            usevar='unequal',
                            alternative='smaller')

    t_cost_RL_GP = ttest_ind(final_cost_RL,
                             final_cost_GP,
                             usevar='unequal',
                             alternative='smaller')

    print('\n    H0: GPc=Mc, H1: GPc<Mc. t: %.4f, p-value: %.4f' %
          (t_cost_GP_M[0], t_cost_GP_M[1]))
    print('    H0: RLc=Mc, H1: RLc<Mc. t: %.4f, p-value: %.4f' %
          (t_cost_RL_M[0], t_cost_RL_M[1]))
    print('    H0: RLc=GPc, H1: RLc<GPc. t: %.4f, p-value: %.4f' %
          (t_cost_RL_GP[0], t_cost_RL_GP[1]))

    # ------------------------------------------------------- linear regression

    linreg_GP_RL = OLS(final_wealth_RL, add_constant(final_wealth_GP)).fit()

    print('\n\n\n Linear regression of X=GP vs Y=RL:\n')
    print(linreg_GP_RL.summary())
    print('\n\n H0: beta=1; H1: beta!=1')
    print(linreg_GP_RL.t_test(([0., 1.], 1.)))

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
