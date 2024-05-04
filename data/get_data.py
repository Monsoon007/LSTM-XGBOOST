"""
返回通用的、没有针对模型进行特定处理的数据
最后一列为未来T日的平均日收益率
"""

from __future__ import print_function, absolute_import

import datetime

from gm.api import *
import numpy as np
import pandas as pd
from pandas import to_datetime


def add_return_column(data, T=3):
    """
    向DataFrame添加未来T日的平均日收益率。

    参数:
    - data: 包含交易数据的DataFrame。
    - T: 未来的天数，默认为3。

    返回值:
    - 该函数没有返回值，但会修改传入的DataFrame，为每个交易日添加一个新列，表示未来T日的收益率。
    """
    # (1+avg)^T = (1+y) = close(T)/close
    # 取对数，T ln(1+avg) = ln(1+y)
    # 对于较小量，ln(1+x) ≈ x
    # 所以 avg = y/T =(close(T)/close-1)/T
    data['avg_daily_return_' + str(T)] = (data['close'].shift(-T) / data['close'] - 1) / T

def add_trend_column(data, T=3,threshold=0.001):
    """
    向DataFrame添加未来T日的平均日收益率对应的标的趋势: 如果收益率大于阈值，则为2，如果收益率小于阈值，则为0，否则为1。从0开始是为了适应torch的交叉熵计算要求
    """
    data[str(T)+'_return_to_trend'] = ((data['close'].shift(-T) / data['close'] - 1) / T).apply(lambda x: 2 if x>threshold else 0 if x<-threshold else 1)

def add_ma_columns(data, ma_periods):
    """
    向DataFrame添加移动平均列。

    参数:
    - data: 包含交易数据的DataFrame。
    - ma_periods: 一个包含要计算的移动平均周期的列表，每个元素是一个整数，表示天数。

    返回值:
    - 该函数没有返回值，但会修改传入的DataFrame，为每个指定的周期添加一个新列。
    """
    for period in ma_periods:
        ma_column_name = f'MA_{period}'  # 根据周期生成列名称，如'MA_5'表示5天移动平均
        data[ma_column_name] = data['close'].rolling(window=period).mean()


def add_ema_columns(data, ema_periods):
    """
    向DataFrame添加指数移动平均列。

    参数:
    - data: 包含交易数据的DataFrame。
    - ema_periods: 一个包含要计算的指数移动平均周期的列表，每个元素是一个整数，表示天数。

    返回值:
    - 该函数没有返回值，但会修改传入的DataFrame，为每个指定的周期添加一个新列。
    """
    for period in ema_periods:
        ema_column_name = f'EMA_{period}'  # 根据周期生成列名称，如'EMA_5'表示5天指数移动平均
        data[ema_column_name] = data['close'].ewm(span=period).mean()


def add_rsi_factor(data, period=14):
    """
    向DataFrame添加相对强弱指数（RSI）因子。

    参数:
    - data: 包含交易数据的DataFrame。
    - period: 计算RSI的周期，默认为14天。
    """
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))


def add_atr_factor(data, period=14):
    """
    向DataFrame添加平均真实范围（ATR）因子。

    参数:
    - data: 包含交易数据的DataFrame。
    - period: 计算ATR的周期，默认为14天。
    """
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR'] = true_range.rolling(window=period).mean()


def add_bollinger_band_width_factor(data, period=20, num_std=2):
    """
    向DataFrame添加布林带宽度因子。

    参数:
    - data: 包含交易数据的DataFrame。
    - period: 计算布林带的周期，默认为20天。
    - num_std: 布林带的标准差倍数，默认为2。
    """
    ma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    data['Bollinger_Width'] = upper_band - lower_band


def add_vwap_factor(data):
    """
    向DataFrame添加成交量加权平均价格（VWAP）因子。

    参数:
    - data: 包含交易数据的DataFrame。
    """
    data['VWAP'] = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()


def calculate_ar(data, period=26):
    """
    计算AR（人气指标）。

    参数:
    - data: 包含'high', 'low', 'open'列的DataFrame。
    - period: 计算指标的周期，默认为26天。

    返回:
    - ar: DataFrame，包含计算周期内的AR值。
    """
    ar_numerator = (data['high'] - data['open']).rolling(window=period).sum()
    ar_denominator = (data['open'] - data['low']).rolling(window=period).sum()
    ar = (ar_numerator / ar_denominator) * 100
    return ar


def calculate_br(data, period=26):
    """
    计算BR（意愿指标）。

    参数:
    - data: 包含'high', 'low', 'close'列的DataFrame。
    - period: 计算指标的周期，默认为26天。

    返回:
    - br: DataFrame，包含计算周期内的BR值。
    """
    br_numerator = (data['high'] - data['close'].shift()).rolling(window=period).sum()
    br_denominator = (data['close'].shift() - data['low']).rolling(window=period).sum()
    br = (br_numerator / br_denominator) * 100
    return br


def my_get_previous_n_trading_date(date, counts=1, exchanges='SHSE'):
    """
    获取date前移N个交易日的日期,end_date为datetime格式，包括date日期
    :param date：目标日期
    :param counts：历史回溯天数，默认为1，即前一天
    """
    return get_previous_n_trading_dates(exchanges, date, counts)[0]

def my_get_next_n_trading_date(date, counts=1, exchange='SHSE'):
    """
    获取date后移N个交易日的日期,end_date为datetime格式，包括date日期
    :param date：目标日期
    :param counts：未来预测天数，默认为1，即后一天
    """
    return get_next_n_trading_dates(exchange,date,counts)[-1]


def get_common_data(symbol, start_date, end_date, T,threshold=0.001):
    # 设置token
    set_token('9c0950e38c59552734328ad13ad93b6cc44ee271')
    macro_data = pd.read_excel('../data/macro.xlsx')
    # date作为索引
    macro_data.set_index('date', inplace=True)
    # macro_data的数据范围
    macro_start_date = macro_data.index[0]
    macro_end_date = macro_data.index[-1]
    max_lead = 120
    # 由于MA_120,EMA_120,avg_daily_return_120,AR,BR需要120天的数据，所以需要提前120天
    leadStart = my_get_previous_n_trading_date(start_date, counts=max_lead)

    # 由于需要计算未来T日的平均日收益率，所以end_date需要推迟T天
    forwardEnd = my_get_next_n_trading_date(end_date, counts=T)

    # 检查leadStart和end_date是否在数据范围内
    if to_datetime(leadStart) < macro_start_date:
        raise ValueError('leadStart is out of macro range')
    if to_datetime(end_date) > macro_end_date:
        raise ValueError('end_date is out of macro range')
    # if to_datetime(leadStart)<to_datetime('2012-05-01'):
    #     raise ValueError('leadStart 不能早于 2012-05-01，因为trade_data最早2012-05，加上MA提前量')
    trade_data = history(symbol, frequency='1d', start_time=leadStart, end_time=forwardEnd, fill_missing='last',
                         df=True)
    # print((trade_data.info()))
    # 去除 'symbol', 'eob', 'frequency','position' 列
    trade_data.drop(['symbol', 'eob', 'frequency', 'position'], axis=1, inplace=True)
    # 将'bob'去时区化后作为索引
    trade_data.set_index('bob', inplace=True)
    # 将data的索引设置为tz-naive
    trade_data.index = trade_data.index.tz_localize(None)
    data = trade_data
    # add_return_column(data, T)
    add_trend_column(data,T,threshold)
    add_ma_columns(data, [5, 10, 20, 60, max_lead])
    add_ema_columns(data, [5, 10, 20, 60, max_lead])
    add_rsi_factor(data)
    add_atr_factor(data)
    add_bollinger_band_width_factor(data)
    add_vwap_factor(data)
    data['AR'] = calculate_ar(data)
    data['BR'] = calculate_br(data)
    # print(data.info())
    # data数据范围修正为start_date和end_date之间
    data = data[start_date:end_date]
    # print(data.info())
    # 将macro_data根据日期与data合并
    data = data.join(macro_data, how='left')
    # 去除有空值的行
    data.dropna(inplace=True)
    # 将监督变量列放在最后
    cols = list(data.columns)
    supervised_col_name = str(T)+'_return_to_trend'
    cols.remove(supervised_col_name)
    cols.append(supervised_col_name)
    data = data[cols]
    return data


if __name__ == '__main__':
    # 获取数据
    data = get_common_data('SHSE.510300', '2019-03-01', '2020-03-31', 3)

    print(data.info())
    print(data.columns)
