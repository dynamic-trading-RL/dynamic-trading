import pandas as pd
import os
import numpy as np

from enums import FactorComputationType, RiskDriverType, FactorSourceType, ModeType, FactorTransformationType
from gen_utils.utils import get_available_futures_tickers


class FinancialTimeSeries:

    def __init__(self, ticker: str, window: int = None, modeType: ModeType = ModeType.InSample):

        self.ticker = ticker
        self._set_time_series(window, modeType)

    def _set_time_series(self, window: int, modeType: ModeType):

        self.modeType = modeType

        # if window is None, then it is taken from financial_time_series_setting.csv
        self._set_settings_from_file(window)
        self._set_asset_time_series(start_date=self.info.loc['start_date'][0],
                                    end_date=self.info.loc['end_date'][0],
                                    in_sample_proportion=float(self.info.loc['in_sample_proportion'][0]))
        self._set_risk_driver_time_series()
        self._set_factor_time_series()
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
            '/data/data_source/settings/settings.csv'
        self.info = pd.read_csv(filename, index_col=0)

        if window is not None:
            self.info.loc['window'] = window
        else:
            if 'window' not in self.info.index:
                raise NameError('financialTimeSeries was instantiated without window, therefore window must be '
                                + 'written in time-series-info.csv')

        if type(self.info.loc['factor_ticker'].item()) == str:

            self.factor_ticker = self.info.loc['factor_ticker'].item()
            self.factorSourceType = FactorSourceType.Exogenous

        else:

            self.factorSourceType = FactorSourceType.Constructed


        self.riskDriverType = RiskDriverType(self.info.loc['riskDriverType'][0])
        self.factorComputationType = FactorComputationType(self.info.loc['factorComputationType'][0])
        self.window = int(self.info.loc['window'][0])
        self.factorTransformationType = FactorTransformationType(self.info.loc['factorTransformationType'].item())

    def _set_asset_time_series(self, start_date: str, end_date: str, in_sample_proportion: float):

        if self.ticker in get_available_futures_tickers():

            time_series = pd.read_excel(os.path.dirname(os.path.dirname(__file__))
                                        + '/data/data_source/market_data/assets_data.xlsx',
                                        sheet_name=self.ticker, index_col=0)

        elif self.ticker == 'fake_asset':

            time_series = pd.read_excel(os.path.dirname(os.path.dirname(__file__))
                                        + '/data/data_source/market_data/fake_asset_data.xlsx',
                                        sheet_name=self.ticker, index_col=0)

        else:

            import yfinance as yf

            end_date = '2022-05-23'
            time_series = yf.download(self.ticker, end=end_date)['Adj Close'].to_frame()

        if pd.isna(start_date):
            start_date = time_series.index[0]
        else:
            start_date = pd.to_datetime(start_date)
        time_series = time_series.loc[time_series.index >= start_date]

        if pd.isna(end_date):
            end_date = time_series.index[-1]
        else:
            end_date = pd.to_datetime(end_date)
        time_series = time_series.loc[time_series.index <= end_date]

        time_series.index = pd.to_datetime(time_series.index)

        first_valid_loc = time_series.first_valid_index()
        last_valid_loc = time_series.last_valid_index()
        date_range = pd.date_range(first_valid_loc, last_valid_loc)
        time_series = time_series.reindex(date_range)
        time_series.dropna(inplace=True)
        self._start_date = time_series.index[0]

        time_series.columns = [self.ticker]

        self._time_series_len = len(time_series)
        self._in_sample_proportion_len = int(self._time_series_len * in_sample_proportion)
        self._out_of_sample_proportion_len = self._time_series_len - self._in_sample_proportion_len

        if self.modeType == ModeType.InSample:
            self.time_series =\
                time_series.iloc[-self._time_series_len: -self._time_series_len + self._in_sample_proportion_len]
        elif self.modeType == ModeType.OutOfSample:
            self.time_series =\
                time_series.iloc[-self._out_of_sample_proportion_len:]
        else:
            raise NameError(f'Invalid modeType: {self.modeType.value}')

        self.time_series.insert(len(self.time_series.columns), 'pnl', np.array(self.time_series[self.ticker].diff()))
        self.time_series.insert(len(self.time_series.columns), 'average_past_pnl', self.time_series['pnl'].rolling(window=self.window, min_periods=1).mean())

    def _set_risk_driver_time_series(self):

        if self.riskDriverType == RiskDriverType.PnL:
            self.time_series['risk-driver'] = self.time_series[self.ticker].diff()

        elif self.riskDriverType == RiskDriverType.Return:
            self.time_series['risk-driver'] = self.time_series[self.ticker].pct_change()

        else:
            raise NameError(f'Invalid riskDriverType: {self.riskDriverType.value}')

    def _set_factor_time_series(self):

        if self.factorSourceType == FactorSourceType.Constructed:
            v = self.time_series[self.ticker]

        elif self.factorSourceType == FactorSourceType.Exogenous:
            filename = os.path.dirname(os.path.dirname(__file__)) +\
                       '/data/data_source/market_data/' + self.factor_ticker + '.xlsx'
            v = pd.read_excel(filename, index_col=0, parse_dates=True)
        else:
            raise NameError('factorSourceType not correctly specified')

        if self.factorTransformationType == FactorTransformationType.Diff:
            pass
        elif self.factorTransformationType == FactorTransformationType.LogDiff:
            v = np.log(v)
        else:
            raise NameError('factorTransformationType not correctly specified')

        x = v.diff()

        self._get_factor_time_series_from_x(x)

    def _get_factor_time_series_from_x(self, x):

        if self.factorComputationType == FactorComputationType.MovingAverage:
            self.time_series['factor'] = x.rolling(self.window).mean()

        elif self.factorComputationType == FactorComputationType.StdMovingAverage:
            lower_bound = x.rolling(self.window).std().squeeze().quantile(0.1)
            den = x.squeeze().rolling(self.window).std()
            den[(den < lower_bound) & (~pd.isna(den))] = lower_bound
            self.time_series['factor'] = x.squeeze().rolling(self.window).mean() / den

    def _set_info(self):

        self.info = pd.DataFrame(index=['ticker',
                                        'end_date',
                                        'start_price',
                                        'len',
                                        'out_of_sample_proportion_len',
                                        'window',
                                        'riskDriverType',
                                        'factorComputationType'],
                                 data=[self.ticker,
                                       self.time_series.index[-1],
                                       self.time_series[self.ticker].iloc[-1],
                                       len(self.time_series),
                                       self._out_of_sample_proportion_len,
                                       self.window,
                                       self.riskDriverType.value,
                                       self.factorComputationType.value],
                                 columns=['info'])


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':
    financialTimeSeries = FinancialTimeSeries('WTI')
