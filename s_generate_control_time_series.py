import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


if __name__ == '__main__':

    np.random.seed(789)

    today = datetime.strptime('2022-12-31', '%Y-%m-%d')

    t_ = 8000

    # factor parameters
    mu_f = 0.
    B_f = 0.8
    sig2_f = 0.2

    # asset parameters
    mu_x = 0.
    B_x = 0.9
    sig2_x = 0.1
    v_min = 100.

    # generate f
    f = np.zeros(t_)
    f[0] = mu_f + np.sqrt(sig2_f) * np.random.randn()
    for t in range(1, t_):
        f[t] = mu_f + B_f * f[t-1] + np.sqrt(sig2_f) * np.random.randn()

    # generate x
    x = np.zeros(t_)
    x[0] = mu_x + np.sqrt(sig2_x) * np.random.randn()
    x[1:] = mu_x + B_x * f[:-1] + np.sqrt(sig2_x) * np.random.randn(t_ - 1)

    # generate v
    v = np.cumsum(x)
    v += v_min - v.min()

    dates = pd.date_range(today - timedelta(days=t_-1), today, freq='D')

    df_f = pd.DataFrame(data=np.cumsum(f), index=dates, columns=['VALUE'])
    df_f.index.name = 'date'
    filename = os.path.dirname(__file__) + '/data/data_source/market_data/fake_factor.xlsx'
    df_f.to_excel(filename, sheet_name='fake_factor')

    df = pd.DataFrame(data=v, index=dates, columns=['VALUE'])
    df.index.name = 'DATE'
    filename = os.path.dirname(__file__) + '/data/data_source/market_data/fake_asset_data.xlsx'
    df.to_excel(filename, sheet_name='fake_asset')
