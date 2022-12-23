import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial import Polynomial

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

    x_optim_when_error = 0.

    if len(coef) < 2:
        raise NameError('Polynomial must be of degree >= 2')

    p = Polynomial(coef)
    dp = p.deriv(m=1)
    dp2 = p.deriv(m=2)

    stationary_points = dp.roots()
    if np.iscomplex(stationary_points).any():
        return x_optim_when_error

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

    return x_optim
