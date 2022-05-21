import pandas as pd
# import yfinance as yf

from enums import FactorDefinitionType, RiskDriverType


class FinancialTimeSeries:

    def __init__(self, ticker: str):

        self.ticker = ticker

    def set_time_series(self):

        self.riskDriverType = RiskDriverType(self.info.loc['riskDriverType'][0])
        self.factorDefinitionType = FactorDefinitionType(self.info.loc['factorDefinitionType'][0])
        self.window = int(self.info.loc['window'][0])
        self._set_asset_time_series(int(self.info.loc['t_past'][0]))
        self._set_risk_driver_time_series()
        self._set_factor()
        self.time_series.dropna(inplace=True)
        self._set_info()

    def print_info(self):

        filename = '../data/data_tmp/' + self.ticker + '-info.csv'
        self.info.to_csv(filename)

    def set_info_from_file(self):

        filename = '../data/data_tmp/' + self.ticker + '-info.csv'
        self.info = pd.read_csv(filename, index_col=0)

    def _set_asset_time_series(self, t_past: int):

        if self.ticker == 'WTI':

            time_series = pd.read_csv('../data/data_source/market_data/DCOILWTICO.csv', index_col=0,
                                      na_values='.').fillna(method='pad')

        else:

            import yfinance as yf

            end_date = '2021-12-31'
            time_series = yf.download(self.ticker, end=end_date)['Adj Close'].to_frame()

        first_valid_loc = time_series.first_valid_index()
        time_series = time_series.loc[first_valid_loc:]

        time_series.index = pd.to_datetime(time_series.index)
        time_series.columns = [self.ticker]

        if t_past > len(time_series):
            t_past = len(time_series)

        self.time_series = time_series.iloc[-t_past:]

    def _set_risk_driver_time_series(self):

        if self.riskDriverType == RiskDriverType.PnL:
            self.time_series['risk-driver'] = self.time_series[self.ticker].diff()
        elif self.riskDriverType == RiskDriverType.Return:
            self.time_series['risk-driver'] = self.time_series[self.ticker].pct_change()
        else:
            raise NameError('Invalid riskDriverType: ' + self.riskDriverType.value)

    def _set_factor(self):

        x = self.time_series['risk-driver']

        if self.factorDefinitionType == FactorDefinitionType.MovingAverage:
            self.time_series['factor'] = x.rolling(self.window).mean()

        elif self.factorDefinitionType == FactorDefinitionType.StdMovingAverage:
            self.time_series['factor'] = x.rolling(self.window).mean() / x.rolling(self.window).std()

    def _set_info(self):

        self.info = pd.DataFrame(index=['ticker',
                                        'end_date',
                                        'start_price',
                                        't_past',
                                        'window',
                                        'riskDriverType',
                                        'factorDefinitionType'],
                                 data=[self.ticker,
                                       self.time_series.index[-1],
                                       self.time_series[self.ticker].iloc[-1],
                                       len(self.time_series),
                                       self.window,
                                       self.riskDriverType.value,
                                       self.factorDefinitionType.value],
                                 columns=['info'])


# ------------------------------ TESTS ---------------------------------------------------------------------------------

if __name__ == '__main__':

    financialTimeSeries = FinancialTimeSeries('WTI')
    financialTimeSeries.set_time_series()
    financialTimeSeries.print_info()
