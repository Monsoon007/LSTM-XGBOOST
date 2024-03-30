# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import datetime
import numpy as np
from gm.api import *
import pandas as pd
import sys
#from hmmlearn.hmm import GaussianHMM # 选择的HMM模型
import logging
import itertools
#from scipy.stats import boxcox #正态变换
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
import random
# random.seed(1)


def init(context):
    # 股票标的
    # context.symbol = 'SHSE.600519'
    context.symbol = 'SHSE.510300'
    random.seed(1)
    # # 历史窗口长度
    # context.history_len = 10

    # # 预测窗口长度
    # context.forecast_len = 1

    # 训练样本长度
    context.training_len = 200

    # 止盈幅度
    context.earn_rate = 0.10

    # 最小涨幅卖出幅度
    context.sell_rate = -0.10

    context.commission_ratio = 0.0001

    context.T = 10
    
    context.percent=0
    
    # 订阅行情
    # subscribe(symbols=context.symbol, frequency='60s')

    context.predictions = pd.DataFrame(columns=["date",'return_pre',"return_pre7","outcome",'limit_return'])

    context.most_probable = []

    # # 生成last T days的收益率可能组合
    # with open('D:\Academy\HMM\code\outcomes.npy', 'rb') as f:
    #     context.outcomes = np.load(f)
    # print(context.outcomes)

    # 生成last T days的收益率可能组合
    # 获取目标股票的daily历史行情

    # trade_data = history(context.symbol, frequency='1d', start_time='2016-01-01', end_time='2019-02-01', fill_missing='last',df=True).set_index('eob')
    # # trade_data['pctChg'] = trade_data['close']/trade_data['pre_close']-1
    # trade_data.drop(columns=trade_data.columns[[0,1,-1,-2]], inplace=True) # 剔除行情数据中无用数据



    # 计算 T日收益率（连续复利）
    #trade_data['return_'+str(context.T)] = np.log(trade_data['close'].shift(-context.T)/trade_data['close'])/context.T
    #return_T = pd.array(trade_data[:-context.T]['return_'+str(context.T)])
    #return_T.reshape(-1,1)
    #kmeans = KMeans(n_clusters=20, random_state=0).fit(return_T.reshape(-1,1))
    #returns = kmeans.cluster_centers_
    #returns = returns.reshape(20)
    #context.outcomes = np.array(list(itertools.product(returns,repeat=context.T))) #遍历组合

    path = "E:\中国移动同步盘\大四\毕业设计\数据"
    # "E:\中国移动同步盘\大四\毕业设计\数据\macro.csv"
    macroPath = path + "\macro.csv"
    sentPath = path + "\sentiment1.7.xlsx"
    techPath = path + "\\tech.xlsx"

    macro_data = pd.read_csv(macroPath)
    sent_data = pd.read_excel(sentPath)
    tech_data = pd.read_excel(techPath)


    #macro_data = macro_data.set_index('date').drop(macro_data.columns[[0,18,19]],axis=1)
    macro_data = macro_data.set_index('date').drop(macro_data.columns[[0,18,19,-1,-2,-4,-5,-6,-7,-8,-9,-10,-11]],axis=1)
    macro_data.index = pd.to_datetime(macro_data.index) 
    # for column in macro_data.columns:
    #     if macro_data[column].min() <= 0:
    #         macro_data[column] = macro_data[column].apply(lambda x:x-macro_data[column].min()+np.exp(-10))
    context.macro_data = macro_data

    #macro_data = macro_data.set_index('date').drop(macro_data.columns[[0,18,19]],axis=1)
    sent_data = sent_data.set_index('date')
    sent_data.index = pd.to_datetime(sent_data.index) 
    context.sent_data = sent_data

    #macro_data = macro_data.set_index('date').drop(macro_data.columns[[0,18,19]],axis=1)
    tech_data = tech_data.set_index('date')
    tech_data.index = pd.to_datetime(tech_data.index) 
    context.tech_data = tech_data
    
    #技术指标
    # tech_data = pd.read_excel('tech.xls').set_index('date')
    # tech_data.index = pd.to_datetime(tech_data.index) 

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
        #try:
        prediction,outcome,limit_return= LSTM_predict(context,last_N_date,last_date)
        df = pd.DataFrame([[now,prediction,outcome[-1],outcome,limit_return]],columns=["date","prediction",'return_pre',"return_pre7",'limit_return'])
        context.predictions = pd.concat([context.predictions,df])

        print('prediction:'+ str(prediction)+"\n")
        print('recent_7day_return:') 
        print(outcome)
        # 若预测值为上涨且空仓则买入
        if prediction == 1:
            context.percent=max(min(200*np.mean(outcome)/limit_return,1),0)
            order_target_percent(symbol=context.symbol, percent=context.percent, order_type=OrderType_Market,position_side=PositionSide_Long)
            print("多仓percent")
            print(context.percent)
            print("多仓volume")
            #print(position.volume)
        # 若预测下跌则清仓    
        if prediction == -1 and position :
            context.percent=min(np.max(context.percent-np.mean(outcome)/limit_return,0),1)
            order_target_percent(symbol=context.symbol, percent=context.percent, order_type=OrderType_Market,position_side=PositionSide_Long)
            print("空仓percent")
            print(context.percent)
            print("空仓volume")
            #print(position.volume)
            # order_close_all()
        #except:
            #print("##### Error of LSTM Prediction #######")
        
        # prediction, predict_recent_return = hmm_predict(context,last_N_date,last_date)
        # df = pd.DataFrame([[now,prediction,predict_recent_return]],columns=["date","prediction","outcome"])
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
       
def LSTM_predict(context,start_date,end_date):

    return_upper = 0.002
    return_lower = -0.002
    #取过去N天的数据作训练输入
    traning_days = 40
    T = context.T
    #获得数据
    trade_data = history(context.symbol, frequency='1d', start_time=start_date, end_time=end_date, fill_missing='last',df=True).set_index('eob')
    # trade_data = history(context.symbol, frequency='1d', start_time=start_date, end_time=end_date, fill_missing='last',df=True).set_index('eob')
    # get_history_symbol(symbol=None, start_date=None, end_date=None, df=False)

    trade_data.drop(columns=trade_data.columns[[0,1,-1,-2]], inplace=True) # 剔除行情数据中无用数据
    #在这里修改数据，加入宏观数据
    trade_data.index = pd.to_datetime(trade_data.index).date
    # trade_data = trade_data.merge(context.sent_data, how='left', left_index=True, right_index=True)   
    trade_data = trade_data.merge(context.macro_data, how='left', left_index=True, right_index=True)   
    # trade_data = trade_data.merge(context.tech_data, how='left', left_index=True, right_index=True)   
    #技术指标
    # trade_data.index = pd.to_datetime(trade_data.index).date
    # trade_data = trade_data.merge(context.tech_data, how='left', left_index=True, right_index=True)   

    #收益率
    return_T = pd.array(np.log(trade_data['close'].shift(-T)/trade_data['close'])/T)
    trade_data.insert(loc=len(trade_data.columns), column='return'+str(T), value=return_T)
    # test_data = trade_data.iloc[-T:]
    #归一化
    min_td = np.min(trade_data['return'+str(T)])
    max_td = np.max(trade_data['return'+str(T)])
    trade_data = trade_data.apply(lambda x:(x-min(x))/(max(x)-min(x)+np.exp(-10)))
    #用60天的数据来预测下一天的数据
    #举个例子 x[0]是0~59天的股价 y[0]是第60天的股价
    X = []
    Y = []
    for i in range(trade_data.shape[0]-traning_days):
        # 全部columns作为特征，除了最后一列收益率
        X.append(np.array(trade_data.iloc[i:(i+traning_days),:-1].values, dtype=np.float64))  
        # 选择return作为标签输出
        Y.append(np.array(trade_data.iloc[(i+traning_days),-1],dtype=np.float64))
    X = np.array(X)
    Y = np.array(Y)

    trade_data.dropna(inplace=True)

 # return_T = pd.array(trade_data['return_'+str(T)])
    kmeans = KMeans(n_clusters=20, random_state=0,algorithm="elkan").fit(trade_data['return'+str(T)].values.reshape(-1,1))
    returns = kmeans.cluster_centers_
    max_return = np.max(returns)
    min_return = np.min(returns)

   



    #搭建模型
    #3层LSTM，对于每一个LSTM，加入dropout防止过拟合，最后1层全连接用来输出
    #输入数据是60长度7维向量组，最后的输出为1个值
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X.shape[1],X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    #adam优化 最小二乘损失
    model.compile(optimizer='adam',loss='mean_squared_error')

    #进行训练
    epochs1=50
    model.fit(X[:-T],Y[:-T],epochs=epochs1,batch_size=16)

    #预测价格
    predict_recent_return = model.predict(X[-T:])
    #逆归一化
    predict_recent_return = predict_recent_return*(max_td-min_td+np.exp(-10)) + min_td

    # 判断买入卖出信号
    # if (predict_recent_return[0]* predict_recent_return[1] <0 
    #   and predict_recent_return[1]* predict_recent_return[2])<0:
    #     return 0,predict_recent_return,0 #震荡
    if min(predict_recent_return) > return_upper :
        return 1,predict_recent_return,max_return #上涨
    elif max(predict_recent_return) <return_lower  :
        return -1,predict_recent_return,min_return#下跌
    else :
        return 0,predict_recent_return,0#震荡
    
    # states_pctChg = pd.DataFrame(model.means_)[7]
    # hidden_states_pre = model.predict(trade_macro_data[-1:])
    # return states_pctChg[hidden_states_pre].values
    # if np.all(predict_recent_return >return_upper) 



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
    run(strategy_id='36bd1563-ed9a-11ee-b6d7-a00af654686a',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='',
        backtest_start_time='2019-02-28 08:00:00',
        backtest_end_time='2020-02-28 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)

