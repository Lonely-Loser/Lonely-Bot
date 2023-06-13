from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import mplfinance as mpf
from stocktrends import Renko
import matplotlib.pyplot as plt
import os


# It's recommended to import this module as ta

# /////////////////////////////////  Functions  /////////////////////////////////


class Technical_Analysia:
    def ticker(self, assets=[], interval='1d', period='1y', export_csv=False):
        self.assets = assets
        if self.assets == []:
            raise ValueError('--->>> "assets" CAN NOT LEAVE EMPTY!!! <<<---')
        self.asset_name = ''
        self.data = pd.DataFrame()
        for a in self.assets:
            self.data[f'{a} Open'] = yf.download(tickers=a, interval=interval, period=period)['Open']
            self.data[f'{a} High'] = yf.download(tickers=a, interval=interval, period=period)['High']
            self.data[f'{a} Low'] = yf.download(tickers=a, interval=interval, period=period)['Low']
            self.data[f'{a} Close'] = yf.download(tickers=a, interval=interval, period=period)['Close']
            self.data[f'{a} Adj Close'] = yf.download(tickers=a, interval=interval, period=period)['Adj Close']
            self.data[f'{a} Volume'] = yf.download(tickers=a, interval=interval, period=period)['Volume']
            self.asset_name += f'({a})'
        # Download CSV file
        if export_csv == False:
            pass
        elif export_csv == True:
            os.makedirs('Inventory', exist_ok=True)
            if len(self.assets) == 1:
                self.data.to_csv(f'Inventory/({a}) (Interval; {interval}) (Period; {period}).csv')
            else:
                self.data.to_csv(f'Inventory/{self.asset_name} (Interval; {interval}) (Period; {period}).csv')
                self.asset_name = ''
        else:
            os.makedirs('Inventory', exist_ok=True)
            self.data.to_csv(f'Inventory/{export_csv}.csv')
        self.cols = self.data.columns.tolist()
        self.sym = self.cols[1][0:-5]
        return self.data, self.sym

    def import_csv(self, name):
        self.data = pd.read_csv(f'Inventory/{name}.csv')
        # self.data.set_index('Date', inplace=True)
        self.cols = self.data.columns.tolist()
        self.sym = self.cols[1][0:-5]
        self.data.index.name = 'Date'
        # self.data.rename(columns={f'{self.sym} Date': 'Date', f'{self.sym} High': 'High', f'{self.sym} Low': 'Low',
        #                           f'{self.sym} Open': 'Open', f'{self.sym} Close': 'Close',
        #                           f'{self.sym} Adj Close': 'Adj Close', f'{self.sym} Volume': 'Volume'}, inplace=True)
        # mpf.plot(self.data, type='renko', style='yahoo', figsize=(50, 20), title='RENKO CHART', volume=True, renko_params=dict(atr_length=14))
        return self.data, self.sym

    def MACD(self, fast_period=12, slow_period=26, signal_period=9, situation='Close'):
        self.data_MACD = pd.DataFrame()
        self.data_MACD['Fast'] = self.data[f'{self.sym} {situation}'].ewm(span=fast_period,
                                                                          min_periods=fast_period).mean()
        self.data_MACD['Slow'] = self.data[f'{self.sym} {situation}'].ewm(span=slow_period,
                                                                          min_periods=slow_period).mean()
        self.data_MACD['MACD'] = self.data_MACD['Fast'] - self.data_MACD['Slow']
        self.data_MACD['Signal'] = self.data_MACD['MACD'].ewm(span=signal_period, min_periods=signal_period).mean()
        self.data_MACD.dropna(inplace=True)
        return self.data_MACD

    def ATR(self, n=14):
        self.data_ATR = pd.DataFrame()
        self.data_ATR['H-L'] = self.data[f'{self.sym} High'] - self.data[f'{self.sym} Low']
        self.data_ATR['H-PC'] = abs(
            self.data[f'{self.sym} High'] - self.data[f'{self.sym} Close'].shift(1))  # PC = Previous Close
        self.data_ATR['L-PC'] = abs(self.data[f'{self.sym} Low'] - self.data[f'{self.sym} Close'].shift(1))
        self.data_ATR['TR'] = self.data_ATR[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
        self.data_ATR['ATR'] = self.data_ATR['TR'].rolling(n).mean()
        self.data_ATR = self.data_ATR.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
        return self.data_ATR

    def BB(self, n=20, situation='Close'):
        self.data_BB = pd.DataFrame()
        self.sum = self.data[f'{self.sym} High'] + self.data[f'{self.sym} Low'] + self.data[f'{self.sym} {situation}']
        self.TP = self.sum / 3  # TP = Typical Price (Avg of HLC)
        self.std = self.TP.rolling(n).std()
        self.data_BB['Mid Line'] = self.TP.rolling(n).mean()
        self.data_BB['Lower'] = self.data_BB['Mid Line'] - 2 * self.std
        self.data_BB['Upper'] = self.data_BB['Mid Line'] + 2 * self.std
        self.data_BB.dropna(inplace=True)
        return self.data_BB

    def RSI(self, n=14, situation='Close'):
        self.data_RSI = pd.DataFrame()
        self.delta = self.data[f'{self.sym} {situation}'].diff(1).dropna()
        self.up = self.delta.clip(lower=0)
        self.down = self.delta.clip(upper=0)
        self.avg_gain = self.up.rolling(n).mean()
        self.avg_loss = self.down.rolling(n).mean().abs()
        self.RS = self.avg_gain / self.avg_loss
        self.data_RSI['RSI'] = 100 - (100 / (1.0 + self.RS)).dropna()
        return self.data_RSI

    def ADX(self, n=14):
        self.data_ADX = pd.DataFrame()
        self.data_ADX['TR'] = self.ATR(n)['TR']
        # DM variables
        self.delta_high = self.data[f'{self.sym} High'] - self.data[f'{self.sym} High'].shift(1)
        self.delta_low = self.data[f'{self.sym} Low'].shift(1) - self.data[f'{self.sym} Low']
        # data_+DM
        self.data_ADX['+DM'] = np.where(self.delta_high > self.delta_low, self.delta_high, 0)
        self.data_ADX['+DM'] = self.data_ADX['+DM'].clip(lower=0)
        # data_-DM
        self.data_ADX['-DM'] = np.where(self.delta_low > self.delta_high, self.delta_low, 0)
        self.data_ADX['-DM'] = self.data_ADX['-DM'].clip(lower=0)
        # Smoothed_DM
        self.TRn, self.DM_plus_n, self.DM_minus_n = [], [], []
        self.TR = self.data_ADX['TR'].tolist()
        self.dm_plus = self.data_ADX['+DM'].tolist()
        self.dm_minus = self.data_ADX['-DM'].tolist()
        for i in range(len(self.data_ADX)):
            if i < n:
                self.TRn.append(np.NaN)
                self.DM_plus_n.append(np.NaN)
                self.DM_minus_n.append(np.NaN)
            elif i == n:
                self.TRn.append(self.data_ADX['TR'].rolling(n).sum().tolist()[n])
                self.DM_plus_n.append(self.data_ADX['+DM'].rolling(n).sum().tolist()[n])
                self.DM_minus_n.append(self.data_ADX['-DM'].rolling(n).sum().tolist()[n])
            else:
                self.TRn.append(self.TRn[i - 1] - (self.TRn[i - 1] / n) + self.TR[i])
                self.DM_plus_n.append(self.DM_plus_n[i - 1] - (self.DM_plus_n[i - 1] / n) + self.dm_plus[i])
                self.DM_minus_n.append(self.DM_minus_n[i - 1] - (self.DM_minus_n[i - 1] / n) + self.dm_minus[i])
        self.data_ADX['Smoothed TR'] = np.array(self.TRn)
        self.data_ADX['Smoothed +DM'] = np.array(self.DM_plus_n)
        self.data_ADX['Smoothed -DM'] = np.array(self.DM_minus_n)
        # DI
        self.data_ADX['+DI'] = 100 * (self.data_ADX['Smoothed +DM'] / self.data_ADX['Smoothed TR'])
        self.data_ADX['-DI'] = 100 * (self.data_ADX['Smoothed -DM'] / self.data_ADX['Smoothed TR'])
        # DX
        self.data_ADX['DI_diff'] = abs(self.data_ADX['+DI'] - self.data_ADX['-DI'])
        self.data_ADX['DI_sum'] = abs(self.data_ADX['+DI'] + self.data_ADX['-DI'])
        self.data_ADX['DX'] = 100 * (self.data_ADX['DI_diff'] / self.data_ADX['DI_sum'])
        # ADX
        self.li_ADX = []
        self.DX = self.data_ADX['DX'].tolist()
        for j in range(len(self.data_ADX)):
            if j < 2 * n - 1:
                self.li_ADX.append(np.NaN)
            elif j == 2 * n - 1:
                self.li_ADX.append(self.data_ADX['DX'][j - n + 1:j + 1].mean())
            else:
                self.li_ADX.append(((n - 1) * self.li_ADX[j - 1] + self.DX[j]) / n)
        self.data_ADX['ADX'] = np.array(self.li_ADX)
        self.data_ADX = self.data_ADX.drop(
            ['TR', '+DM', '-DM', 'Smoothed TR', 'Smoothed +DM', 'Smoothed -DM', '+DI', '-DI', 'DI_diff', 'DI_sum'],
            axis=1)
        return self.data_ADX

    def OBV(self):
        self.data_OBV = pd.DataFrame()
        self.data_OBV['ret'] = self.data[f'{self.sym} Close'].pct_change()
        self.data_OBV['dir'] = np.where(self.data_OBV['ret'] > 0, 1, -1)
        self.data_OBV['dir'][0] = 0
        self.data_OBV['adj_volume'] = self.data[f'{self.sym} Volume'] * self.data_OBV['dir']
        self.data_OBV['OBV'] = self.data_OBV['adj_volume'].cumsum()
        self.data_OBV = self.data_OBV.drop(['ret', 'dir', 'adj_volume'], axis=1)
        return self.data_OBV

    def DEMA(self, short=20, long=50, situation='Close'):
        self.data_DEMA = pd.DataFrame()
        self.ema_short = self.data[f'{self.sym} {situation}'].ewm(span=short, adjust=False).mean()
        self.ema_long = self.data[f'{self.sym} {situation}'].ewm(span=long, adjust=False).mean()
        self.data_DEMA['Short'] = 2 * self.ema_short - self.ema_short.ewm(span=short, adjust=False).mean()
        self.data_DEMA['Long'] = 2 * self.ema_long - self.ema_long.ewm(span=long, adjust=False).mean()
        return self.data_DEMA

    def DEMA_Signal(self, short=20, long=50, situation='Close'):
        self.data_DEMA_signal = pd.DataFrame()
        self.buy_list, self.sell_list = [], []
        self.flag = False
        self.DEMA(short, long)
        for i in range(0, len(self.data)):
            if self.data_DEMA['Short'][i] > self.data_DEMA['Long'][i] and self.flag == False:
                self.buy_list.append(self.data[f'{self.sym} {situation}'][i])
                self.sell_list.append(np.NaN)
            elif self.data_DEMA['Short'][i] < self.data_DEMA['Long'][i] and self.flag == True:
                self.buy_list.append(np.NaN)
                self.sell_list.append(self.data[f'{self.sym} {situation}'][i])
            else:
                self.buy_list.append(np.NaN)
                self.sell_list.append(np.NaN)
        self.data_DEMA_signal['Buy'] = self.buy_list
        self.data_DEMA_signal['Sell'] = self.sell_list
        return self.data_DEMA_signal

    def slope(self, n=5, situation='Close'):
        self.data_slope = self.data[f'{self.sym} {situation}']
        self.slopes = [i * 0 for i in range(n - 1)]

        for i in range(n, len(self.data_slope) + 1):
            self.y = self.data_slope[i - n:i]
            self.x = np.array(range(n))
            self.y_scaled = (self.y - self.y.min()) / (self.y.max() - self.y.min())
            self.x_scaled = self.x / 4  # x.min() = 0 and x.max() = 4 ==> x_scaled = (x - x.min()) / (x.max() - x.min()) = x / 4
            self.x_scaled = sm.add_constant(self.x_scaled)
            self.model = sm.OLS(self.y_scaled, self.x_scaled)
            self.results = self.model.fit()
            self.slopes.append(self.results.params[-1])

        self.slope_angle = (np.rad2deg(np.arctan(np.array(self.slopes))))
        self.data['slope'] = np.array(self.slope_angle)
        # data.iloc[:, [5, 6]].plot(subplots=True, layout=(2, 1), figsize=(16, 8))
        # plt.show()
        return self.data_slope

    def renko(self):
        self.data_renko_0 = self.data.copy()
        # self.data_renko_0.reset_index(inplace=True)
        # self.data_renko_0 = self.data_renko_0.iloc[:, [0, 1, 2, 3, 4, 5, 6]]
        self.data_renko_0.rename(
            columns={f'{self.sym} Date': 'date', f'{self.sym} High': 'high', f'{self.sym} Low': 'low',
                     f'{self.sym} Open': 'open', f'{self.sym} Adj Close': 'close', f'{self.sym} Volume': 'volume'},
            inplace=True)
        print(self.data_renko_0)
        self.data_renko = Renko(self.data_renko_0)
        self.data_renko.brick_size = round(self.ATR(120)['ATR'][-1], 0)
        # self.renko_df = self.data_renko.get_ohlc_data()
        # mpf.plot(self.data_renko_0, type='renko', renko_params=dict(brick_size=self.data_renko, atr_length=14),
        #          style='yahoo', figsize=(18, 7), title='RENKO CHART')
        # return self.renko_df

# ////////////////////////////////////  tst  ////////////////////////////////////
# fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(16, 8))
# data.iloc[:, 3].plot(ax=ax0)
# ax0.set(ylabel='Price')
# data_MACD.iloc[:, [2, 3]].plot(ax=ax0)
# ax1.set(xlabel='Date', ylabel='MACD')
# fig.suptitle('MACD Indicator')
# plt.show()
