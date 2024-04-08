# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import datetime
import numpy as np
from gm.api import *
import pandas as pd
import sys
from hmmlearn.hmm import GaussianHMM # 选择的HMM模型
import logging
import itertools
from scipy.stats import boxcox #正态变换
from sklearn.cluster import KMeans

def init(context):
    # 股票标的
    # context.symbol = 'SHSE.600519'
    context.symbol = 'SHSE.510300'

    # # 历史窗口长度
    # context.history_len = 10

    # # 预测窗口长度
    # context.forecast_len = 1

    # 训练样本长度
    context.training_len = 550 # 150近似最优

    # 止盈幅度
    context.earn_rate = 0.10

    # 最小涨幅卖出幅度
    context.sell_rate = -0.10

    context.commission_ratio = 0.0001

    context.T = 7
    
    # 订阅行情
    subscribe(symbols=context.symbol, frequency='60s')

    context.predictions = pd.DataFrame(columns=["date","prediction"])

    context.most_probable = []

    # # 生成last T days的收益率可能组合
    # with open('D:\Academy\HMM\code\outcomes.npy', 'rb') as f:
    #     context.outcomes = np.load(f)
    # print(context.outcomes)

    # 生成last T days的收益率可能组合
    # 获取目标股票的daily历史行情


    macro_data = pd.read_csv('E:\All Codes\HMM\macro.csv')
    macro_data = macro_data.set_index('date').drop(macro_data.columns[[0,18,19,-1,-2,-4,-5,-6,-7,-8,-9,-10,-11]],axis=1)
    macro_data.index = pd.to_datetime(macro_data.index) 
    for column in macro_data.columns:
        if macro_data[column].min() <= 0:
            macro_data[column] = macro_data[column].apply(lambda x:x-macro_data[column].min()+np.exp(-10))
    context.macro_data = macro_data

def on_bar(context, bars):
    bar = bars[0]
    # print('>>>>>>>\n')
    # print(bars)
    # print('<<<<<<<\n')
    # print(bar)
    # 当前时间
    now = context.now
    

    # 每天开票进行预测
    if now.hour==9 and now.minute==31:
        # 获取当前时间的星期
        weekday = now.isoweekday()
        # 上一交易日
        last_date = get_previous_trading_date(exchange='SHSE', date=now)
        # print("------------\n")
        # print(str(now)+"\n")
        # print(str(last_date)+"\n")
        # N天前的交易日
        last_N_date = get_previous_N_trading_date(last_date,counts=context.training_len,exchanges='SHSE')
        # 获取持仓
        position = context.account().position(symbol=context.symbol, side=PositionSide_Long)
        print(">>>>>>>>>>>\n")
        print(str(now)+"\n")
        try:
            prediction, most_probable_outcome = hmm_predict(context,last_N_date,last_date)
            df = pd.DataFrame([[now,prediction,most_probable_outcome]],columns=["date","prediction","outcome"])
            # context.predictions.concat(df)
            context.predictions = pd.concat([context.predictions,df])
            print(str(prediction)+"\n")
            # 若预测值为上涨且空仓则买入
            if prediction == 1 and not position :
                order_target_percent(symbol=context.symbol, percent=1, order_type=OrderType_Market,position_side=PositionSide_Long)
            # 若预测下跌则清仓    
            if prediction == -1 and position :
                order_close_all()
        except:
            print("##### Error of HMM Prediction #######")
        
        # prediction, most_probable_outcome = hmm_predict(context,last_N_date,last_date)
        # df = pd.DataFrame([[now,prediction,most_probable_outcome]],columns=["date","prediction","outcome"])
        # # context.predictions.concat(df)
        # context.predictions = pd.concat([context.predictions,df])
        # print(str(prediction)+"\n")
        # # 若预测值为上涨且空仓则买入
        # if prediction == 1 and not position :
        #     order_target_percent(symbol=context.symbol, percent=1, order_type=OrderType_Market,position_side=PositionSide_Long)
        # # 若预测下跌则清仓    
        # if prediction == -1 and position :
        #     order_close_all()
        
        # print("流程正确")
        # 当涨幅大于10%,平掉所有仓位止盈    
        if position and bar.close/position['vwap'] >= 1+context.earn_rate:
            order_close_all()
            print("触发止盈")

        # 当跌幅大于10%时,平掉所有仓位止损
        if position and bar.close/position['vwap'] < 1+context.sell_rate :
            order_close_all()
            print("触发止损")


def on_order_status(context, order):  #用于打印交易信息
    # 标的代码
    symbol = order['symbol']
    # 委托价格
    price = order['price']
    # 委托数量
    volume = order['volume']
    # 目标仓位
    target_percent = order['target_percent']
    # 查看下单后的委托状态，等于3代表委托全部成交
    status = order['status']
    # 买卖方向，1为买入，2为卖出
    side = order['side']
    # 开平仓类型，1为开仓，2为平仓
    effect = order['position_effect']
    # 委托类型，1为限价委托，2为市价委托
    order_type = order['order_type']
    if status == 3:
        if effect == 1:
            if side == 1:
                side_effect = '开多仓'
            elif side == 2:
                side_effect = '开空仓'
        else:
            if side == 1:
                side_effect = '平空仓'
            elif side == 2:
                side_effect = '平多仓'
        order_type_word = '限价' if order_type==1 else '市价'
        print('{}:标的：{}，操作：以{}{}，委托价格：{}，委托数量：{}'.format(context.now,symbol,order_type_word,side_effect,price,volume))
       
def hmm_predict(context,start_date,end_date):
    """
    训练HMM模型
    :param start_date:训练样本开始时间
    :param end_date:训练样本结束时间
    """
    T = context.T
    # N = 20
    return_upper = 0.0002
    return_lower = -0.0002



    # 获取目标股票的daily历史行情
    trade_data = history(context.symbol, frequency='1d', start_time=start_date, end_time=end_date, fill_missing='last',df=True).set_index('eob')
    # trade_data['pctChg'] = trade_data['close']/trade_data['pre_close']-1
    trade_data.drop(columns=trade_data.columns[[0,1,-1,-2]], inplace=True) # 剔除行情数据中无用数据
    trade_data.index = pd.to_datetime(trade_data.index).date
    trade_macro_data = trade_data.merge(context.macro_data, how='left', left_index=True, right_index=True)
    # 计算 T日收益率（连续复利）
    trade_macro_data['return_'+str(T)] = np.log(trade_macro_data['close']/trade_macro_data['close'].shift(T))/T
    trade_macro_data.dropna(inplace=True)
    # trade_data = pd.DataFrame(trade_data['return_'+str(T)])
    # y, lambda0 = boxcox(x, lmbda=None, alpha=None)
    for column in trade_macro_data.columns:
        if column == 'return_'+str(T):
            continue
        # print(column)
        # print(trade_data[column])
        trade_macro_data[column],__ = boxcox(trade_macro_data[column])
    

    # except:
    #     return 2 
    # model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=10000) 
    model = GaussianHMM(n_components=3,covariance_type='diag', n_iter=10000) 

    model.fit(trade_macro_data)
    states_pct = model.means_[:,-1]

    hidden_states = model.predict(trade_macro_data[-3:]) #解码最近三天对应的状态
    
    outcome = states_pct[hidden_states]
    if (outcome[0]* outcome[1] <0 
      and outcome[1]* outcome[2])<0:
        return 0,outcome #震荡
    elif outcome[-1] > return_upper :
        return 1,outcome  #上涨
    elif np.all(outcome <return_lower)  :
        return -1,outcome#下跌
    else :
        return 0,outcome #震荡


def get_previous_N_trading_date(date,counts=1,exchanges='SHSE'):
    """
    获取end_date前N个交易日,end_date为datetime格式，包括date日期
    :param date：目标日期
    :param counts：历史回溯天数，默认为1，即前一天
    """
    if isinstance(date,str) and len(date)>10:
        date = datetime.datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
    if isinstance(date,str) and len(date)==10:
        date = datetime.datetime.strptime(date,'%Y-%m-%d')
    previous_N_trading_date = get_trading_dates(exchange=exchanges, start_date=date-datetime.timedelta(days=max(counts+30,counts*2)), end_date=date)[-counts]
    return previous_N_trading_date


def on_backtest_finished(context, indicator):
    real_now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    filename = f'{str(real_now)}.csv'
    # filename = 'predictions.csv'
    context.predictions.to_csv(filename)
    # pd.DataFrame(context.most_probable).to_csv('most_outcomes.csv')
    print( '*' *50)
    print('回测已完成，请通过右上角“回测历史”功能查询详情。')


if __name__ == '__main__':
    '''
        strategy_id策略ID, 由系统生成
        filename文件名, 请与本文件名保持一致
        mode运行模式, 实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID, 可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式, 不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        '''
    run(strategy_id='7b16e7ef-6a08-11ed-a326-88aedd1eed07',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='',
        backtest_start_time='2019-02-28 08:00:00',
        backtest_end_time='2020-09-10 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)

