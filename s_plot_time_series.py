
import os
import pandas as pd
import matplotlib.pyplot as plt

from gen_utils.utils import get_available_futures_tickers

if __name__ == '__main__':

    mkt_data_path = f'{os.path.dirname(__file__)}/data/data_source/market_data/'

    assets_data_path = f'{mkt_data_path}assets_data.xlsx'

    # Plot assets
    for ticker in get_available_futures_tickers():

        time_series = pd.read_excel(assets_data_path, sheet_name=ticker, index_col=0, parse_dates=True)
        time_series.index.name = 'date'
        time_series.rename(columns={'VALUE': ticker})

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
        plt.plot(time_series)
        plt.title(ticker)
        plt.ylabel('Value [$]')
        plt.xlabel('date')
        filename = f'{os.path.dirname(__file__)}/figures/time_series/asset_{ticker}_time_series.png'
        plt.savefig(filename)
        plt.close()

    for factor_ticker in ('SP500', 'VIX', 'RV5'):

        time_series = pd.read_excel(f'{mkt_data_path}{factor_ticker}.xlsx', index_col=0, parse_dates=True)
        time_series.index.name = 'date'
        time_series.rename(columns={'VALUE': factor_ticker})

        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
        plt.plot(time_series)
        plt.title(factor_ticker)
        plt.ylabel('Value')
        plt.xlabel('date')
        filename = f'{os.path.dirname(__file__)}/figures/time_series/factor_{factor_ticker}_time_series.png'
        plt.savefig(filename)
        plt.close()
