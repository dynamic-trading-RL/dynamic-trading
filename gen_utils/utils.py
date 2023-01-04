import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial import Polynomial
from scipy.stats import truncnorm


def read_ticker():

    filename = os.path.dirname(os.path.dirname(__file__)) + '/data/data_source/settings/settings.csv'
    df_trad_params = pd.read_csv(filename, index_col=0)
    ticker = str(df_trad_params.loc['ticker'][0])

    return ticker


def get_available_futures_tickers():
    lst = ['cocoa', 'coffee', 'copper', 'WTI', 'WTI-spot', 'gasoil', 'gold', 'lead', 'nat-gas-rngc1d', 'nat-gas-reuter',
           'nickel', 'silver', 'sugar', 'tin', 'unleaded', 'zinc']

    return lst


def instantiate_polynomialFeatures(degree):

    poly = PolynomialFeatures(degree=degree,
                              interaction_only=False,
                              include_bias=True)

    return poly


def find_polynomial_minimum(coef, bounds):

    x_optim_when_error = truncnorm.rvs(a=bounds[0], b=bounds[1], loc=0., scale=0.12)
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
    fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
    plt.plot(xx, yy, label='Polinomial')
    plt.plot(xx, dyy, label='Polynomial 1st derivative')
    plt.plot(xx, ddyy, label='Polynomial 2nd derivative')
    plt.plot(xx, 0 * np.ones(len(xx)), label='Zero line', color='k')
    plt.vlines(x_optim, min(0, p(x_optim)), max(0, p(x_optim)), label=f'Optimum = {x_optim: .2f}')
    plt.legend()
    plt.xlim(bounds)

    filename = os.path.dirname(os.path.dirname(__file__)) + f'/figures/polynomial/polynomial{int(eps_plots*10**5)}.png'

    plt.savefig(filename)
