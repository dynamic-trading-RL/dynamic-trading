import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial import Polynomial
from scipy.stats import truncnorm

from enums import InitialQvalueEstimateType, OptimizerType, SupervisedRegressorType

available_ann_architectures = [(64,),
                               (64, 32),
                               (64, 32, 8),
                               (64, 32, 16, 8),
                               (64, 32, 16, 8, 4)]


def read_ticker():

    filename = os.path.dirname(os.path.dirname(__file__)) + '/data/data_source/settings/settings.csv'
    df_trad_params = pd.read_csv(filename, index_col=0)
    ticker = str(df_trad_params.loc['ticker'][0])

    return ticker


def get_available_futures_tickers():
    lst = ['cocoa', 'coffee', 'copper', 'WTI', 'WTI-spot', 'gold', 'lead', 'nat-gas-rngc1d', 'nat-gas-reuter',
           'nickel', 'silver', 'sugar', 'unleaded', 'zinc']  # 'tin', 'gasoil'

    return lst


def instantiate_polynomialFeatures(degree):

    poly = PolynomialFeatures(degree=degree,
                              interaction_only=False,
                              include_bias=True)

    return poly


def find_polynomial_minimum(coef, bounds):

    x_optim_when_error = truncnorm.rvs(a=bounds[0], b=bounds[1], loc=0., scale=0.01 * (bounds[1] - bounds[0]))
    flag_error = False

    if len(coef) < 2:
        raise NameError('Polynomial must be of degree >= 2')

    p = Polynomial(coef)
    dp = p.deriv(m=1)
    dp2 = p.deriv(m=2)

    stationary_points = dp.roots()

    # exclude complex roots
    stationary_points = np.real(stationary_points[np.isreal(stationary_points)])

    if len(stationary_points) == 0:
        x_optim = x_optim_when_error
        flag_error = True

    else:
        stationary_points = stationary_points[stationary_points >= bounds[0]]
        stationary_points = stationary_points[stationary_points <= bounds[1]]

        stationary_points_hessian = np.zeros(len(stationary_points))

        for i in range(len(stationary_points)):
            stationary_points_hessian[i] = dp2(stationary_points[i])

        minimal_points = stationary_points[stationary_points_hessian > 0]

        if len(minimal_points) > 0:
            x_optim = minimal_points[0]
            for i in range(len(minimal_points)):
                if p(minimal_points[i]) >= p(x_optim):
                    x_optim = minimal_points[i]
        else:
            x_optim = x_optim_when_error
            flag_error = True

    eps_plots = np.random.rand()
    if eps_plots < 10**-3:
        _make_plot_once_in_a_while(p, dp, dp2, bounds, x_optim, eps_plots)

    return x_optim, flag_error


def _make_plot_once_in_a_while(p, dp, dp2, bounds, x_optim, eps_plots):

    xx = np.linspace(bounds[0], bounds[1], 20)
    yy = p(xx)
    dyy = dp(xx)
    ddyy = dp2(xx)

    dpi = plt.rcParams['figure.dpi']
    plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
    plt.plot(xx, yy, label='Polinomial')
    plt.plot(xx, dyy, label='Polynomial 1st derivative')
    plt.plot(xx, ddyy, label='Polynomial 2nd derivative')
    plt.plot(xx, 0 * np.ones(len(xx)), label='Zero line', color='k')
    plt.vlines(x_optim, min(0, p(x_optim)), max(0, p(x_optim)), label=f'Optimum = {x_optim: .2f}')
    plt.legend()
    plt.xlim(bounds)

    filename = os.path.dirname(os.path.dirname(__file__)) + f'/figures/polynomial/polynomial{int(eps_plots*10**5)}.png'

    plt.savefig(filename)


def read_trading_parameters_training():
    filename = os.path.dirname(os.path.dirname(__file__)) + \
               '/data/data_source/settings/settings.csv'
    df_trad_params = pd.read_csv(filename, index_col=0)

    shares_scale = float(load(os.path.dirname(os.path.dirname(__file__)) + '/data/data_tmp/shares_scale.joblib'))

    j_episodes = int(df_trad_params.loc['j_episodes'][0])
    n_batches = int(df_trad_params.loc['n_batches'][0])
    t_ = int(df_trad_params.loc['t_'][0])

    if df_trad_params.loc['parallel_computing_train'][0] == 'Yes':
        parallel_computing_train = True
        n_cores = min(int(df_trad_params.loc['n_cores'][0]), mp.cpu_count())
    elif df_trad_params.loc['parallel_computing_train'][0] == 'No':
        parallel_computing_train = False
        n_cores = None
    else:
        raise NameError('Invalid value for parameter parallel_computing_train in settings.csv')

    if df_trad_params.loc['parallel_computing_sim'][0] == 'Yes':
        parallel_computing_sim = True
        n_cores = min(int(df_trad_params.loc['n_cores'][0]), mp.cpu_count())
    elif df_trad_params.loc['parallel_computing_sim'][0] == 'No':
        parallel_computing_sim = False
        n_cores = None
    else:
        raise NameError('Invalid value for parameter parallel_computing_sim in settings.csv')

    # if zero, the initial estimate of the qvalue function is 0; if random, it is N(0,1)
    initialQvalueEstimateType = InitialQvalueEstimateType(df_trad_params.loc['initialQvalueEstimateType'][0])

    # if True, the agent uses the model to predict the next step pnl and sig2 for the reward; else, uses the realized
    if df_trad_params.loc['predict_pnl_for_reward'][0] == 'Yes':
        predict_pnl_for_reward = True
    elif df_trad_params.loc['predict_pnl_for_reward'][0] == 'No':
        predict_pnl_for_reward = False
    else:
        raise NameError('Invalid value for parameter predict_pnl_for_reward in settings.csv')

    # if True, the agent averages across supervised regressors in its definition of q_value; else, uses the last one
    if df_trad_params.loc['average_across_models'][0] == 'Yes':
        average_across_models = True
    elif df_trad_params.loc['average_across_models'][0] == 'No':
        average_across_models = False
    else:
        raise NameError('Invalid value for parameter average_across_models in settings.csv')

    # if True, then the agent considers the supervised regressors only up to n<=n_batches, where n is the batch that
    # provided the best reward in the training phase
    if df_trad_params.loc['use_best_n_batch'][0] == 'Yes':
        use_best_n_batch = True
    elif df_trad_params.loc['use_best_n_batch'][0] == 'No':
        use_best_n_batch = False
    else:
        raise NameError('Invalid value for parameter use_best_n_batch in settings.csv')

    use_best_n_batch_mode = df_trad_params.loc['use_best_n_batch_mode'][0]

    # if True, the agent observes the reward GP would obtain and forces its strategy to be GP's if such reward is higher
    # than the one learned automatically
    if df_trad_params.loc['train_benchmarking_GP_reward'][0] == 'Yes':
        train_benchmarking_GP_reward = True
    elif df_trad_params.loc['train_benchmarking_GP_reward'][0] == 'No':
        train_benchmarking_GP_reward = False
    else:
        raise NameError('Invalid value for parameter train_benchmarking_GP_reward in settings.csv')

    # which optimizer to use in greedy policy
    optimizerType = OptimizerType(df_trad_params.loc['optimizerType'][0])

    # choose which model to use for supervised regression
    supervisedRegressorType = SupervisedRegressorType(df_trad_params.loc['supervisedRegressorType'][0])

    # initial epsilon for eps-greedy policy: at each batch iteration, we do eps <- eps/3
    eps_start = float(df_trad_params.loc['eps_start'][0])

    max_ann_depth = int(df_trad_params.loc['max_ann_depth'][0])

    if df_trad_params.loc['early_stopping'][0] == 'Yes':
        early_stopping = True
    elif df_trad_params.loc['early_stopping'][0] == 'No':
        early_stopping = False
    else:
        raise NameError('Invalid value for parameter early_stopping in settings.csv')

    max_iter = int(df_trad_params.loc['max_iter'][0])

    n_iter_no_change = int(df_trad_params.loc['n_iter_no_change'][0])

    activation = str(df_trad_params.loc['activation'][0])

    alpha_sarsa = float(df_trad_params.loc['alpha_sarsa'][0])

    if df_trad_params.loc['decrease_eps'][0] == 'Yes':
        decrease_eps = True
    elif df_trad_params.loc['decrease_eps'][0] == 'No':
        decrease_eps = False
    else:
        raise NameError('Invalid value for parameter decrease_eps in settings.csv')

    if df_trad_params.loc['random_initial_state'][0] == 'Yes':
        random_initial_state = True
    elif df_trad_params.loc['random_initial_state'][0] == 'No':
        random_initial_state = False
    else:
        raise NameError('Invalid value for parameter random_initial_state in settings.csv')

    max_polynomial_regression_degree = int(df_trad_params.loc['max_polynomial_regression_degree'][0])

    if df_trad_params.loc['max_complexity_no_gridsearch'][0] == 'Yes':
        max_complexity_no_gridsearch = True
    elif df_trad_params.loc['max_complexity_no_gridsearch'][0] == 'No':
        max_complexity_no_gridsearch = False
    else:
        raise NameError('Invalid value for parameter max_complexity_no_gridsearch in settings.csv')

    alpha_ewma = float(df_trad_params.loc['alpha_ewma'][0])

    return (shares_scale, j_episodes, n_batches, t_, parallel_computing_train, n_cores, initialQvalueEstimateType,
            predict_pnl_for_reward, average_across_models, use_best_n_batch, train_benchmarking_GP_reward,
            optimizerType, supervisedRegressorType, eps_start, max_ann_depth, early_stopping, max_iter,
            n_iter_no_change, activation, alpha_sarsa, decrease_eps, random_initial_state,
            max_polynomial_regression_degree, max_complexity_no_gridsearch, alpha_ewma, parallel_computing_sim,
            use_best_n_batch_mode)
