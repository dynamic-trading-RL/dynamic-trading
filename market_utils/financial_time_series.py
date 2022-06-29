import pandas as pd
import os
import numpy as np

from enums import FactorDefinitionType, RiskDriverType, FactorSourceType


class FinancialTimeSeries:

    def __init__(self, ticker: str, window: int = None):

        self.ticker = ticker
        self._set_time_series(window)

    def _set_time_series(self, window: int):
        # if window is None, then it is taken from financial_time_series_setting.csv

        self._set_settings_from_file(window)

        self.riskDriverType = RiskDriverType(self.info.loc['riskDriverType'][0])
        self.factorDefinitionType = FactorDefinitionType(self.info.loc['factorDefinitionType'][0])
        self.window = int(self.info.loc['window'][0])
        self._set_asset_time_series(int(self.info.loc['max_len'][0]), float(self.info.loc['in_sample_proportion'][0]))
        self._set_risk_driver_time_series()
        self._set_factor()
        self.time_series.dropna(inplace=True)
        self._set_info()
        self._print_info()

    def _print_info(self):

        filename = os.path.dirname(os.path.dirname(__file__)) + \
                   '/data/financial_time_series_data/financial_time_series_info/' + self.ticker + '-info.csv'
        self.info.to_csv(filename)

    def _set_settings_from_file(self, window):

        filename = \
            os.path.dirname(os.path.dirname(__file__)) + \
            '/data/data_source/trading_settings/financial_time_series_settings.csv'
        self.info = pd.read_csv(filename, index_col=0)

        if window is not None:
            self.info.loc['window'] = window
        else:
            if 'window' not in self.info.index:
                raise NameError('financialTimeSeries was instantiated without window, therefore window must be '
                                + 'written in time-series-info.csv')

        if self.info.loc['factorSourceType'] == FactorSourceType.Exogenous.value:

            if self.info.loc['factor_ticker'] == '':
                raise NameError('User specified factorSourceType = Exogenous but has not provided a factor_ticker')

    def _set_asset_time_series(self, max_len: int, in_sample_proportion: float):

        if self.ticker in get_available_futures_tickers():

            time_series = pd.read_excel(os.path.dirname(os.path.dirname(__file__))
                                        + '/data/data_source/market_data/futures_data.xlsx',
                                        sheet_name=self.ticker, index_col=0).fillna(method='pad')

        else:

            import yfinance as yf

            end_date = '2022-05-23'
            time_series = yf.download(self.ticker, end=end_date)['Adj Close'].to_frame()

        time_series.index = pd.to_datetime(time_series.index)
        first_valid_loc = time_series.first_valid_index()
        last_valid_loc = time_series.last_valid_index()
        date_range = pd.date_range(first_valid_loc, last_valid_loc)
        time_series = time_series.reindex(date_range, method='pad')

        time_series.columns = [self.ticker]

        if max_len > len(time_series):
            max_len = len(time_series)

        self._time_series_len = max_len
        self._in_sample_proportion_len = int(self._time_series_len * in_sample_proportion)
        self._out_of_sample_proportion_len = max_len - self._in_sample_proportion_len

        self.time_series = time_series.iloc[-max_len: -max_len + self._in_sample_proportion_len]
        self.time_series.insert(len(self.time_series.columns), 'pnl', np.array(self.time_series[self.ticker].diff()))

    def _set_risk_driver_time_series(self):

        if self.riskDriverType == RiskDriverType.PnL:
            self.time_series['risk-driver'] = self.time_series[self.ticker].diff()
        elif self.riskDriverType == RiskDriverType.Return:
            self.time_series['risk-driver'] = self.time_series[self.ticker].pct_change()
        else:
            raise NameError('Invalid riskDriverType: ' + self.riskDriverType.value)

    def _set_factor(self):

        if self.info.loc['factorSourceType'] == FactorSourceType.Constructed.value:

            x = self.time_series['risk-driver']

            if self.factorDefinitionType == FactorDefinitionType.MovingAverage:
                self.time_series['factor'] = x.rolling(self.window).mean()

            elif self.factorDefinitionType == FactorDefinitionType.StdMovingAverage:
                all_stds = x.rolling(self.window).std()
                lower_bound = all_stds.quantile(0.1)
                den = x.rolling(self.window).std()
                den[den < lower_bound] = lower_bound
                self.time_series['factor'] = x.rolling(self.window).mean() / den

        elif self.info.loc['factorSourceType'] == FactorSourceType.Exogenous.value:

            pass

        else:
            raise NameError('factorSourceType not correctly specified')

    def _set_info(self):

        self.info = pd.DataFrame(index=['ticker',
                                        'end_date',
                                        'start_price',
                                        'len',
                                        'out_of_sample_proportion_len',
                                        'window',
                                        'riskDriverType',
                                        'factorDefinitionType'],
                                 data=[self.ticker,
                                       self.time_series.index[-1],
                                       self.time_series[self.ticker].iloc[-1],
                                       len(self.time_series),
                                       self._out_of_sample_proportion_len,
                                       self.window,
                                       self.riskDriverType.value,
                                       self.factorDefinitionType.value],
                                 columns=['info'])


def get_available_futures_tickers():
    lst = ['cocoa', 'coffee', 'copper', 'WTI', 'gasoil', 'gold', 'lead', 'nat-gas-rngc1d', 'nat-gas-reuter', 'nickel',
           'silver', 'sugar', 'tin', 'unleaded', 'zinc']

    return lst


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':
    financialTimeSeries = FinancialTimeSeries('WTI', window=10)
