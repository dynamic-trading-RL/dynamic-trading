# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:41:52 2021

@author: Giorgi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump
from functools import partial
import multiprocessing as mp
from dt_functions import (ReturnDynamicsType, FactorDynamicsType,
                          FactorType,
                          instantiate_market,
                          get_Sigma,
                          simulate_market,
                          generate_episode,
                          Optimizers,
                          set_regressor_parameters_ann,
                          set_regressor_parameters_tree,
                          set_regressor_parameters_gb,
                          compute_markovitz,
                          compute_GP)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


if __name__ == '__main__':

    # ------------------------------------- Parameters ------------------------

    # RL parameters
    j_episodes = 1000
    n_batches = 5
    t_ = 1000

    parallel_computing = True
    n_cores_max = 50
    alpha = 1.
    eps = 0.1
    # None, 'differential_evolution', 'shgo', 'dual_annealing', 'best',
    # 'brute', 'local'
    optimizer = 'dual_annealing'
    # random_forest, gradient_boosting, ann_deep, ann_fast, ann_small
    sup_model = 'ann_deep'

    flag_qaverage = True
    predict_r = True

    resc_by_M = True

    dyn_update_q_value = True

    standardize_Y = True
    rescale_n_a = True

    if dyn_update_q_value:
        random_act_batch0 = False
    else:
        random_act_batch0 = True

    make_plots = True
    dump_XY = False

    # Market parameters
    returnDynamicsType = ReturnDynamicsType.Linear
    factorDynamicsType = FactorDynamicsType.AR
    gamma = 10**-4  # 2*10**-5  # risk aversion
    lam_perc = 10**-6  # costs: percentage of unit trade value
    rho = 1 - np.exp(-.02/252)  # discount
    factorType = FactorType.Observable

    # ------------------------------------- Reinforcement learning ------------
    calibration_parameters = pd.read_excel('data/calibration_parameters.xlsx',
                                           index_col=0)
    startPrice = calibration_parameters.loc['startPrice',
                                            'calibration-parameters']
    qb_list = []  # list to store models
    optimizers = Optimizers()

    if sup_model in ('ann_fast', 'ann_deep', 'ann_small'):
        hidden_layer_sizes, max_iter, n_iter_no_change, alpha_ann =\
            set_regressor_parameters_ann(sup_model)
    elif sup_model == 'random_forest':
        n_estimators, min_samples_split, max_samples, warm_start, verbose =\
            set_regressor_parameters_tree()
    elif sup_model == 'gradient_boosting':
        learning_rate, n_estimators, subsample, min_samples_split,\
            warm_start, n_iter_no_change, verbose =\
            set_regressor_parameters_gb()
    else:
        raise NameError('Invalid sup_model: ' + str(sup_model))

    if parallel_computing:
        print('Number of cores available: %d' % mp.cpu_count())
        n_cores = min(mp.cpu_count(), n_cores_max)
        print('Number of cores used: %d' % n_cores)

    # Instantiate market
    market = instantiate_market(returnDynamicsType, factorDynamicsType,
                                startPrice)

    # Simulations
    price, pnl, f = simulate_market(market, j_episodes, n_batches, t_)
    Sigma_r = get_Sigma(market)
    lam = lam_perc * 2 / Sigma_r
    Lambda_r = lam*Sigma_r

    # Get dynamics
    if (market._marketDynamics._returnDynamics._returnDynamicsType
            == ReturnDynamicsType.Linear):
        B = market._marketDynamics._returnDynamics._parameters['B']
        mu_r = market._marketDynamics._returnDynamics._parameters['mu']
    else:
        B_0 = market._marketDynamics._returnDynamics._parameters['B_0']
        B_1 = market._marketDynamics._returnDynamics._parameters['B_1']
        B = 0.5*(B_0 + B_1)
        mu_0 = market._marketDynamics._returnDynamics._parameters['mu_0']
        mu_1 = market._marketDynamics._returnDynamics._parameters['mu_1']
        mu_r = 0.5*(mu_0 + mu_1)
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

    if resc_by_M:
        Markowitz = compute_markovitz(f.flatten(), gamma, B, Sigma_r,
                                      price.flatten(), mu_r)
        bound = np.percentile(np.abs(Markowitz), 95)
    else:
        GP = compute_GP(f.flatten(), gamma, lam, rho, B, Sigma_r, Phi,
                        price.flatten(), mu_r)
        bound = np.percentile(np.abs(GP), 95)

    reward = np.zeros((n_batches, j_episodes))
    cost = np.zeros((n_batches, j_episodes))

    if make_plots:
        Y_plot = np.zeros((n_batches, j_episodes*(t_-1)))
        if factorType == FactorType.Observable:
            X_plot = np.zeros((n_batches, j_episodes*(t_-1), 3))
        elif factorType == FactorType.Latent:
            X_plot = np.zeros((n_batches, j_episodes*(t_-1), 2))
        else:
            raise NameError('Invalid factorType: ' + factorType.value)

    for b in range(n_batches):  # loop on batches

        print('Creating batch %d of %d; eps=%f' % (b+1, n_batches, eps))
        X = []  # simulations
        Y = []
        j_sort = []
        reward_sort = []
        cost_sort = []

        # generate episodes
        # create alias for generate_episode that fixes all the parameters but j
        # this way we can iterate it via multiprocessing.Pool.map()

        gen_ep_part = partial(generate_episode,
                              # market simulations
                              price=price[:, b, ], pnl=pnl[:, b, :],
                              f=f[:, b, :], market=market,
                              factorType=factorType,
                              # reward/cost parameters
                              rho=rho, gamma=gamma, Sigma_r=Sigma_r,
                              Lambda_r=Lambda_r,
                              # RL parameters
                              eps=eps, qb_list=qb_list,
                              flag_qaverage=flag_qaverage, alpha=alpha,
                              optimizers=optimizers, optimizer=optimizer,
                              b=b, bound=bound, predict_r=predict_r,
                              dyn_update_q_value=dyn_update_q_value,
                              random_act_batch0=random_act_batch0,
                              rescale_n_a=rescale_n_a)

        if parallel_computing:

            p = mp.Pool(n_cores)
            episodes = p.map(gen_ep_part, range(j_episodes))
            p.close()
            p.join()

            # unpack episodes into arrays
            for j in range(len(episodes)):
                X.append(episodes[j][0])
                Y.append(episodes[j][1])
                j_sort.append(episodes[j][2])
                reward_sort.append(episodes[j][3])
                cost_sort.append(episodes[j][4])

        else:
            for j in range(j_episodes):
                print('Computing episode '+str(j+1)+' on '+str(j_episodes))
                episodes = gen_ep_part(j)
                X.append(episodes[0])
                Y.append(episodes[1])
                j_sort.append(episodes[2])
                reward_sort.append(episodes[3])
                cost_sort.append(episodes[4])

        if factorType == FactorType.Observable:
            X = np.array(X).reshape((j_episodes*(t_-1), 3))
        elif factorType == FactorType.Latent:
            X = np.array(X).reshape((j_episodes*(t_-1), 2))
        else:
            raise NameError('Invalid factorType: ' + factorType.value)

        Y = np.array(Y).reshape((j_episodes*(t_-1)))

        if standardize_Y:
            Y = (Y - np.mean(Y)) / np.std(Y)

        ind_sort = np.argsort(j_sort)
        j_sort = np.sort(j_sort)
        reward[b, :] = np.array(reward_sort)[ind_sort]
        cost[b, :] = np.array(cost_sort)[ind_sort]

        print('Fitting model %d of %d' % (b+1, n_batches))

        if sup_model == 'random_forest':

            model = RandomForestRegressor(n_estimators=n_estimators,
                                          min_samples_split=min_samples_split,
                                          max_samples=max_samples,
                                          warm_start=warm_start,
                                          verbose=verbose).fit(X, Y)

        elif (sup_model == 'ann_fast'
              or sup_model == 'ann_deep'
              or sup_model == 'ann_small'):

            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                 alpha=alpha_ann,
                                 max_iter=max_iter,
                                 n_iter_no_change=n_iter_no_change,
                                 activation='relu').fit(X, Y)

        elif sup_model == 'gradient_boosting':

            model =\
                GradientBoostingRegressor(learning_rate=learning_rate,
                                          subsample=subsample,
                                          n_estimators=n_estimators,
                                          min_samples_split=min_samples_split,
                                          warm_start=warm_start,
                                          n_iter_no_change=n_iter_no_change,
                                          verbose=verbose).fit(X, Y)

        else:

            raise NameError('Invalid sup_model: ' + str(sup_model))

        qb_list.append(model)
        dump(model, 'models/q%d.joblib' % b)  # export regressor
        if dump_XY:
            dump(X, 'data/X%d.joblib' % b)
            dump(Y, 'data/Y%d.joblib' % b)
        if make_plots:
            X_plot[b] = X
            Y_plot[b] = Y
        print('    Score: %.3f' % model.score(X, Y))
        print('    Average reward: %.3f' % np.mean(reward[b, :]))
        print('    Average cost: %.3f' % np.mean(cost[b, :]))

        eps = max(eps/3, 0.00001)  # update epsilon

    # ------------------------------------- Dump data -------------------------
    print(optimizers)
    dump(n_batches, 'data/n_batches.joblib')
    dump(optimizers, 'data/optimizers.joblib')
    dump(optimizer, 'data/optimizer.joblib')
    dump(lam, 'data/lam.joblib')
    dump(gamma, 'data/gamma.joblib')
    dump(rho, 'data/rho.joblib')
    dump(factorType, 'data/factorType.joblib')
    dump(flag_qaverage, 'data/flag_qaverage.joblib')
    dump(bound, 'data/bound.joblib')
    dump(rescale_n_a, 'data/rescale_n_a.joblib')

    # ------------------------------------- Plots -----------------------------

    if make_plots:

        color = cm.Greens(np.linspace(0.3, 1, n_batches))

        for b in range(n_batches):
            plt.figure()
            plt.plot(X_plot[b, :, 0], Y_plot[b, :], '.', color='b',
                     label='actual')
            plt.plot(X_plot[b, :, 0], qb_list[b].predict(X_plot[b]), '.',
                     color='r', label='predicted')
            plt.legend()
            plt.title('state_0 vs q; batch=%d' % b)
            plt.savefig('figures/state_0 vs q; batch=%d.png' % b)

        for b in range(n_batches):
            plt.figure()
            plt.plot(X_plot[b, :, -1], Y_plot[b, :], '.', color='b',
                     label='actual')
            plt.plot(X_plot[b, :, -1], qb_list[b].predict(X_plot[b]), '.',
                     color='r', label='predicted')
            plt.legend()
            plt.title('action vs q; batch=%d' % b)
            plt.savefig('figures/action vs q; batch=%d.png' % b)

        if factorType == FactorType.Observable:
            for b in range(n_batches):
                plt.figure()
                plt.plot(X_plot[b, :, 1], Y_plot[b, :], '.', color='b',
                         label='actual')
                plt.plot(X_plot[b, :, 1], qb_list[b].predict(X_plot[b]), '.',
                         color='r', label='predicted')
                plt.legend()
                plt.title('state_1 vs q; batch=%d' % b)
                plt.savefig('figures/state_1 vs q; batch=%d.png' % b)


        plt.figure()
        for b in range(n_batches):
            plt.plot(reward[b, :], label='Batch: %d' % b, alpha=0.5,
                     color=color[b])
        plt.legend()
        plt.title('reward')
        plt.savefig('figures/reward.png')

        plt.figure()
        for b in range(n_batches):
            plt.plot(np.cumsum(reward[b, :]), label='Batch: %d' % b, alpha=0.5,
                     color=color[b])
        plt.legend()
        plt.title('cum-reward')
        plt.savefig('figures/cum-reward.png')

        plt.figure()
        plt.plot(np.cumsum(reward[-1, :]), alpha=0.5)
        plt.title('cum-reward-final')
        plt.savefig('figures/cum-reward-final.png')

        plt.figure()
        for b in range(n_batches):
            plt.plot(cost[b, :], label='Batch: %d' % b, alpha=0.5, color=color[b])
        plt.legend()
        plt.title('cost')
        plt.savefig('figures/cost.png')

        plt.figure()
        for b in range(n_batches):
            plt.plot(np.cumsum(cost[b, :]), label='Batch: %d' % b, alpha=0.5,
                     color=color[b])
        plt.legend()
        plt.title('cum-cost')
        plt.savefig('figures/cum-cost.png')

        plt.figure()
        plt.plot(np.cumsum(cost[-1, :]), alpha=0.5)
        plt.title('cum-cost-final')
        plt.savefig('figures/cum-cost-final.png')
