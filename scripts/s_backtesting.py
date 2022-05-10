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
from dt_functions import (compute_markovitz,
                          compute_GP,
                          compute_rl,
                          compute_wealth,
                          get_dynamics_params)
from market import instantiate_market, get_Sigma
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


if __name__ == '__main__':

    # ------------------------------------- Parameters ------------------------

    t_ = 50

    riskDriverDynamicsType = load('../data/data_tmp/riskDriverDynamicsType.joblib')
    factorDynamicsType = load('../data/data_tmp/factorDynamicsType.joblib')

    # ------- Implied parameters

    calibration_parameters = pd.read_excel('../data/data_tmp/calibration_parameters.xlsx',
                                           index_col=0)
    start_price = calibration_parameters.loc['start_price',
                                            'calibration-parameters']

    n_batches = load('../data/data_tmp/n_batches.joblib')
    optimizers = load('../data/data_tmp/optimizers.joblib')
    optimizer = load('../data/data_tmp/optimizer.joblib')
    lam = load('../data/data_tmp/lam.joblib')
    gamma = load('../data/data_tmp/gamma.joblib')
    rho = load('../data/data_tmp/rho.joblib')
    factorType = load('../data/data_tmp/factorType.joblib')
    flag_qaverage = load('../data/data_tmp/flag_qaverage.joblib')
    bound = load('../data/data_tmp/bound.joblib')
    rescale_n_a = load('../data/data_tmp/rescale_n_a.joblib')
    return_is_pnl = load('../data/data_tmp/return_is_pnl.joblib')

    # ------------------------------------- Simulations -----------------------

    # Instantiate market
    market = instantiate_market(riskDriverDynamicsType, factorDynamicsType,
                                start_price, return_is_pnl)

    Sigma = market.get_sig()
    Lambda = lam*Sigma

    B, mu_r, Phi, mu_f = get_dynamics_params(market)

    # Time series
    df = pd.read_csv('../data/data_tmp/df.csv', index_col=0, parse_dates=True)
    ticker = load('../data/data_tmp/ticker.joblib')

    price = df[ticker][-2*t_:-t_]
    pnl = df[ticker].diff()[-2*t_:-t_]
    f = df['factor'][-t_:]

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

    RL = compute_rl(0, f=f, qb_list=qb_list,
                    factorType=factorType, optimizers=optimizers,
                    optimizer=optimizer,
                    bound=bound, rescale_n_a=rescale_n_a)

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
