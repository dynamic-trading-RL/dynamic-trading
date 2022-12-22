import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


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