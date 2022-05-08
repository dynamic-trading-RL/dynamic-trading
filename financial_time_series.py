import pandas as pd
import yfinance as yf

from enums import FactorDefinitionType


class FinancialTimeSeries:

    def __init__(self, ticker, ):

        self.ticker = ticker
        self.timeSeries = None

    def set_time_series(self,
                        t_past=8000,
                        use_pnl=True,
                        factorDefinitionType=FactorDefinitionType.MovingAverage,
                        window=5):

        self.use_pnl = use_pnl
        self.factorDefinitionType = factorDefinitionType
        self.window = window
        self._set_asset_time_series(t_past)
        self._set_pnl_and_return_time_series()
        self._set_factor()
        self.time_series.dropna(inplace=True)
        self._set_info()

    def _set_asset_time_series(self, t_past):

        if self.ticker == 'WTI':

            time_series = pd.read_csv('data_source/DCOILWTICO.csv', index_col=0,
                                      na_values='.').fillna(method='pad')

        else:

            end_date = '2021-12-31'
            time_series = yf.download(self.ticker, end=end_date)['Adj Close'].to_frame()

        first_valid_loc = time_series.first_valid_index()
        time_series = time_series.loc[first_valid_loc:]

        time_series.index = pd.to_datetime(time_series.index)
        time_series.columns = [self.ticker]

        if t_past > len(time_series):
            t_past = len(time_series)

        self.time_series = time_series.iloc[-t_past:]

    def _set_pnl_and_return_time_series(self):

        self.time_series['pnl'] = self.time_series[self.ticker].diff()
        self.time_series['return'] = self.time_series[self.ticker].pct_change()

    def _set_factor(self):

        if self.use_pnl:
            x = self.time_series['pnl']
        else:
            x = self.time_series['return']

        if self.factorDefinitionType == FactorDefinitionType.MovingAverage:
            self.time_series['f'] = x.rolling(self.window).mean()

        elif self.factorDefinitionType == FactorDefinitionType.StdMovingAverage:
            self.time_series['f'] = x.rolling(self.window).mean() / x.rolling(self.window).std()

    def _set_info(self):

        self.info = pd.DataFrame(index=['ticker',
                                        'end_date',
                                        'start_price',
                                        't_past',
                                        'window',
                                        'use_pnl'],
                                 data=[self.ticker,
                                       self.time_series.index[-1],
                                       self.time_series[self.ticker].iloc[-1],
                                       len(self.time_series),
                                       self.window,
                                       str(self.use_pnl)],
                                 columns=['info'])


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':

    WTI_timeSeries = FinancialTimeSeries('WTI')
    WTI_timeSeries.set_time_series()
    print(WTI_timeSeries.time_series.tail())
