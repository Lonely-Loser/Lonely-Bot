from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import mplfinance as mplf
from stocktrends import Renko
import matplotlib.pyplot as plt
import os

# It's recommended to import this module as ta

# /////////////////////////////////  Functions  /////////////////////////////////
global data_slope, data_renko
assets = ['BTC-USD']
data, data_MACD, data_ATR, data_BB, data_RSI, data_ADX, data_OBV, data_DEMA, data_DEMA_signal = pd.DataFrame()


def ticker(assets=assets, interval='1d', period='1y', export_csv=False):
    global data
    asset_name = ''
    # data = pd.DataFrame()
    for a in assets:
        data[f'{a} Open'] = yf.download(tickers=a, interval=interval, period=period)['Open']
        data[f'{a} High'] = yf.download(tickers=a, interval=interval, period=period)['High']
        data[f'{a} Low'] = yf.download(tickers=a, interval=interval, period=period)['Low']
        data[f'{a} Close'] = yf.download(tickers=a, interval=interval, period=period)['Close']
        data[f'{a} Adj Close'] = yf.download(tickers=a, interval=interval, period=period)['Adj Close']
        data[f'{a} Volume'] = yf.download(tickers=a, interval=interval, period=period)['Volume']
        asset_name += f'({a})'
    # Download CSV file
    if export_csv == False:
        pass
    elif export_csv == True:
        os.makedirs('Inventory', exist_ok=True)
        if len(assets) == 1:
            data.to_csv(f'Inventory/({a}) (Interval; {interval}) (Period; {period}).csv')
        else:
            data.to_csv(f'Inventory/{asset_name} (Interval; {interval}) (Period; {period}).csv')
            asset_name = ''
    else:
        os.makedirs('Inventory', exist_ok=True)
        data.to_csv(f'Inventory/{export_csv}.csv')
    # return data


def import_csv(name):
    global data
    data = pd.read_csv(f'Inventory/{name}.csv')
    data.set_index('Datetime', inplace=True)
    return data


def MACD(fast_period=12, slow_period=26, signal_period=9):
    global data, data_MACD
    # data_MACD = pd.DataFrame()
    data_MACD['Fast'] = data['Close'].ewm(span=fast_period, min_periods=fast_period).mean()
    data_MACD['Slow'] = data['Close'].ewm(span=slow_period, min_periods=slow_period).mean()
    data_MACD['MACD'] = data_MACD['Fast'] - data_MACD['Slow']
    data_MACD['Signal'] = data_MACD['MACD'].ewm(span=signal_period, min_periods=signal_period).mean()
    data_MACD.dropna(inplace=True)
    return data_MACD


def ATR(n=14):
    global data, data_ATR
    # data_ATR = pd.DataFrame()
    data_ATR['H-L'] = data['High'] - data['Low']
    data_ATR['H-PC'] = abs(data['High'] - data['Close'].shift(1))  # PC = Previous Close
    data_ATR['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    data_ATR['TR'] = data_ATR[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    data_ATR['ATR'] = data_ATR['TR'].rolling(n).mean()
    data_ATR = data_ATR.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    return data_ATR


def BB(n=20):
    global data, data_BB
    # data_BB = pd.DataFrame()
    TP = (data.High + data.Low + data.Close) / 3  # TP = Typical Price (Avg of HLC)
    std = TP.rolling(n).std()
    data_BB['Mid Line'] = TP.rolling(n).mean()
    data_BB['Upper'] = data_BB['Mid Line'] + 2 * std
    data_BB['Lower'] = data_BB['Mid Line'] - 2 * std
    data_BB.dropna(inplace=True)
    return data_BB


def RSI(n=14):
    global data, data_RSI
    # data_RSI = pd.DataFrame()
    delta = data['Close'].diff(1).dropna()
    up = delta.clip(lower=0)
    down = delta.clip(upper=0)
    avg_gain = up.rolling(n).mean()
    avg_loss = down.rolling(n).mean().abs()
    RS = avg_gain / avg_loss
    data_RSI['RSI'] = 100 - (100 / (1.0 + RS)).dropna()
    return data_RSI


def ADX(n=14):
    global data, data_ATR, data_ADX
    # data_ADX = pd.DataFrame()
    data_ADX['TR'] = ATR(n)['TR']
    # DM variables
    delta_high = data['High'] - data['High'].shift(1)
    delta_low = data['Low'].shift(1) - data['Low']
    # data_+DM
    data_ADX['+DM'] = np.where(delta_high > delta_low, delta_high, 0)
    data_ADX['+DM'] = data_ADX['+DM'].clip(lower=0)
    # data_-DM
    data_ADX['-DM'] = np.where(delta_low > delta_high, delta_low, 0)
    data_ADX['-DM'] = data_ADX['-DM'].clip(lower=0)
    # Smoothed_DM
    TRn, DM_plus_n, DM_minus_n = [], [], []
    TR = data_ADX['TR'].tolist()
    dm_plus = data_ADX['+DM'].tolist()
    dm_minus = data_ADX['-DM'].tolist()
    for i in range(len(data_ADX)):
        if i < n:
            TRn.append(np.NaN)
            DM_plus_n.append(np.NaN)
            DM_minus_n.append(np.NaN)
        elif i == n:
            TRn.append(data_ADX['TR'].rolling(n).sum().tolist()[n])
            DM_plus_n.append(data_ADX['+DM'].rolling(n).sum().tolist()[n])
            DM_minus_n.append(data_ADX['-DM'].rolling(n).sum().tolist()[n])
        else:
            TRn.append(TRn[i - 1] - (TRn[i - 1] / n) + TR[i])
            DM_plus_n.append(DM_plus_n[i - 1] - (DM_plus_n[i - 1] / n) + dm_plus[i])
            DM_minus_n.append(DM_minus_n[i - 1] - (DM_minus_n[i - 1] / n) + dm_minus[i])
    data_ADX['Smoothed TR'] = np.array(TRn)
    data_ADX['Smoothed +DM'] = np.array(DM_plus_n)
    data_ADX['Smoothed -DM'] = np.array(DM_minus_n)
    # DI
    data_ADX['+DI'] = 100 * (data_ADX['Smoothed +DM'] / data_ADX['Smoothed TR'])
    data_ADX['-DI'] = 100 * (data_ADX['Smoothed -DM'] / data_ADX['Smoothed TR'])
    # DX
    data_ADX['DI_diff'] = abs(data_ADX['+DI'] - data_ADX['-DI'])
    data_ADX['DI_sum'] = abs(data_ADX['+DI'] + data_ADX['-DI'])
    data_ADX['DX'] = 100 * (data_ADX['DI_diff'] / data_ADX['DI_sum'])
    # ADX
    li_ADX = []
    DX = data_ADX['DX'].tolist()
    for j in range(len(data_ADX)):
        if j < 2 * n - 1:
            li_ADX.append(np.NaN)
        elif j == 2 * n - 1:
            li_ADX.append(data_ADX['DX'][j - n + 1:j + 1].mean())
        else:
            li_ADX.append(((n - 1) * li_ADX[j - 1] + DX[j]) / n)
    data_ADX['ADX'] = np.array(li_ADX)
    data_ADX = data_ADX.drop(
        ['TR', '+DM', '-DM', 'Smoothed TR', 'Smoothed +DM', 'Smoothed -DM', '+DI', '-DI', 'DI_diff', 'DI_sum'], axis=1)
    return data_ADX


def OBV():
    global data, data_OBV
    # data_OBV = pd.DataFrame()
    # data_OBV['ret'] = data['Close'].pct_change()
    # data_OBV['dir'] = np.where(data_OBV['ret'] > 0, 1, -1)
    data_OBV['dir'] = np.where(data['Change %'] > 0, 1, -1)
    data_OBV['dir'][0] = 0
    data_OBV['adj_volume'] = data['Volume'] * data_OBV['dir']
    data_OBV['OBV'] = data_OBV['adj_volume'].cumsum()
    data_OBV = data_OBV.drop(['ret', 'dir', 'adj_volume'], axis=1)
    return data_OBV


def DEMA(short=20, long=50):
    global data, data_DEMA
    # data_DEMA = pd.DataFrame()
    ema_short = data['Close'].ewm(span=short, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long, adjust=False).mean()
    data_DEMA['Short'] = 2 * ema_short - ema_short.ewm(span=short, adjust=False).mean()
    data_DEMA['Long'] = 2 * ema_long - ema_long.ewm(span=long, adjust=False).mean()
    return data_DEMA


def DEMA_Signal(short, long):
    global data, data_DEMA, data_DEMA_signal
    # data_DEMA_signal = pd.DataFrame()
    buy_list, sell_list = [], []
    flag = False

    DEMA(short, long)
    for i in range(0, len(data)):
        if data_DEMA['Short'][i] > data_DEMA['Long'][i] and flag == False:
            buy_list.append(data['Close'][i])
            sell_list.append(np.NaN)
        elif data_DEMA['Short'][i] < data_DEMA['Long'][i] and flag == True:
            buy_list.append(np.NaN)
            sell_list.append(data['Close'][i])
        else:
            buy_list.append(np.NaN)
            sell_list.append(np.NaN)

    data_DEMA_signal['Buy'] = buy_list
    data_DEMA_signal['Sell'] = sell_list
    return data_DEMA_signal


def slope(n=5):
    global data, data_slope
    data_slope = data['Close']
    slopes = [i * 0 for i in range(n - 1)]

    for i in range(n, len(data_slope) + 1):
        y = data_slope[i - n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min()) / (y.max() - y.min())
        x_scaled = x / 4  # x.min() = 0 and x.max() = 4 ==> x_scaled = (x - x.min()) / (x.max() - x.min()) = x / 4
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])

    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    data['slope'] = np.array(slope_angle)
    # data.iloc[:, [5, 6]].plot(subplots=True, layout=(2, 1), figsize=(16, 8))
    # plt.show()
    return data_slope


def renko():
    global data, data_renko_0
    data_renko_0 = data.copy()
    data_renko_0.reset_index(inplace=True)
    data_renko_0 = data_renko_0.iloc[:, [0, 1, 2, 3, 4, 5, 6]]
    data_renko_0.rename(columns={'Date': 'date', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Adj Close': 'close',
                                 'Volume': 'volume'}, inplace=True)
    data_renko = Renko(data_renko_0)
    data_renko.brick_size = round(ATR(120)['ATR'][-1], 0)
    # renko_df = data_renko.get_ohlc_data()
    # return renko_df

# ////////////////////////////////////  tst  ////////////////////////////////////
# fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(16, 8))
# data.iloc[:, 3].plot(ax=ax0)
# ax0.set(ylabel='Price')
# data_MACD.iloc[:, [2, 3]].plot(ax=ax0)
# ax1.set(xlabel='Date', ylabel='MACD')
# fig.suptitle('MACD Indicator')
# plt.show()
