# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 10:26:40 2022

@author: feder
"""

import numpy as np
import pandas as pd
from joblib import load
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt
from dt_functions import (instantiate_market,
                          get_Sigma,
                          compute_markovitz,
                          compute_GP,
                          compute_rl,
                          compute_wealth,
                          get_dynamics_params)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


if __name__ == '__main__':

    # ------------------------------------- Parameters ------------------------

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
    return_is_pnl = load('data/return_is_pnl.joblib')
    parallel_computing = load('data/parallel_computing.joblib')
    n_cores = load('data/n_cores.joblib')

    # ------------------------------------- Simulations -----------------------

    # Instantiate market
    market = instantiate_market(returnDynamicsType, factorDynamicsType,
                                startPrice, return_is_pnl)

    Sigma = get_Sigma(market)
    Lambda = lam*Sigma

    B, mu_r, Phi, mu_f = get_dynamics_params(market)

    # Time series
    df = load('data/df.joblib')
    ticker = load('data/ticker.joblib')

    price = df[ticker][-t_:]
    pnl = df[ticker].diff()[-t_:]
    f = df['f'][-t_:]

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
        qb_list.append(load('models/q%d.joblib' % b))

    RL = np.zeros((1, t_))

    if parallel_computing:

        compute_rl_part = partial(compute_rl, f=f, qb_list=qb_list,
                                  factorType=factorType, optimizers=optimizers,
                                  optimizer=optimizer, bound=bound,
                                  rescale_n_a=rescale_n_a)

        p = mp.Pool(n_cores)
        shares = p.map(compute_rl_part, range(1))
        p.close()
        p.join()
        RL = np.array(shares)

    else:

        for j in range(1):

            RL[j] = compute_rl(j, f=f, qb_list=qb_list,
                               factorType=factorType, optimizers=optimizers,
                               optimizer=optimizer,
                               bound=bound, rescale_n_a=rescale_n_a)

    RL = RL.squeeze()

    wealth_RL, value_RL, cost_RL =\
        compute_wealth(pnl, RL, gamma, Lambda, rho, Sigma, price,
                       return_is_pnl)

    # ------------------------------------- Plots -----------------------------

    plt.figure()
    plt.plot(df[ticker][-t_:])
    plt.title('price')
    plt.savefig('figures/price.png')

    plt.figure()
    plt.plot(Markowitz*price, color='m', label='Markowitz')
    plt.plot(GP*price, color='g', label='GP')
    plt.plot(RL*price, color='r', label='RL')
    plt.legend()
    plt.title('in-sample-shares')
    plt.savefig('figures/in-sample-shares.png')

    plt.figure()
    plt.plot(wealth_M, color='m', label='Markowitz')
    plt.plot(wealth_GP, color='g', label='GP')
    plt.plot(wealth_RL, color='r', label='RL')
    plt.legend()
    plt.title('in-sample-wealth')
    plt.savefig('figures/in-sample-wealth.png')

    plt.figure()
    plt.plot(value_M, color='m', label='Markowitz')
    plt.plot(value_GP, color='g', label='GP')
    plt.plot(value_RL, color='r', label='RL')
    plt.legend()
    plt.title('in-sample-value')
    plt.savefig('figures/in-sample-value.png')

    plt.figure()
    plt.plot(cost_M, color='m', label='Markowitz')
    plt.plot(cost_GP, color='g', label='GP')
    plt.plot(cost_RL, color='r', label='RL')
    plt.legend()
    plt.title('in-sample-cost')
    plt.savefig('figures/in-sample-cost.png')

    print('#### END')
