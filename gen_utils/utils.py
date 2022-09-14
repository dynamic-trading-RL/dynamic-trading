import os
import pandas as pd


def read_ticker():

    filename = os.path.dirname(os.path.dirname(__file__)) + '/data/data_source/settings/settings.csv'
    df_trad_params = pd.read_csv(filename, index_col=0)
    ticker = str(df_trad_params.loc['ticker'][0])

    return ticker
