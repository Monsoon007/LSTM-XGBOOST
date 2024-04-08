# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import datetime
import numpy as np
from gm.api import *
import pandas as pd
# from hmmlearn.hmm import GaussianHMM # 选择的HMM模型
# from scipy.stats import boxcox #正态变换
from sklearn.cluster import KMeans
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing

from model.LSTM_base_model import best_lstm_model
from ..data.get_data import get_common_data, my_get_previous_n_trading_date  # 导入位于../data/get_data.py的get_common_data函数
from model.LSTM_data_handle import data_for_lstm

# random.seed(1)

def init(context):
    # 每天14:50 定时执行algo任务,
    # algo执行定时任务函数，只能传context参数
    # date_rule执行频率，目前暂时支持1d、1w、1m，其中1w、1m仅用于回测，实时模式1d以上的频率，需要在algo判断日期
    # time_rule执行时间， 注意多个定时任务设置同一个时间点，前面的定时任务会被后面的覆盖
    schedule(schedule_func=algo, date_rule='1d', time_rule='14:50:00')

    # 股票标的
    # context.symbol = 'SHSE.600519'
    context.symbol = 'SHSE.510300'
    random.seed(1)
    # # 历史窗口长度
    # context.history_len = 10

    # # 预测窗口长度
    # context.forecast_len = 1

    # # 训练样本长度
    # context.training_len = 200
    # context.T = 3  # 预测未来T天的收益率

    # 止盈幅度
    context.earn_rate = 0.10

    # 最小涨幅卖出幅度
    context.sell_rate = -0.10

    context.commission_ratio = 0.0001

    context.percent = 0

    # 订阅行情
    # subscribe(symbols=context.symbol, frequency='60s')

    context.predictions = pd.DataFrame(columns=["date", 'return_pre', "return_pre7", "outcome", 'limit_return'])

    context.most_probable = []

    path = "E:\中国移动同步盘\大四\毕业设计\数据"
    # "E:\中国移动同步盘\大四\毕业设计\数据\macro.csv"
    macroPath = path + "\macro.csv"
    sentPath = path + "\sentiment1.7.xlsx"
    techPath = path + "\\tech.xlsx"

    macro_data = pd.read_csv(macroPath)
    sent_data = pd.read_excel(sentPath)
    tech_data = pd.read_excel(techPath)

    # macro_data = macro_data.set_index('date').drop(macro_data.columns[[0,18,19]],axis=1)
    macro_data = macro_data.set_index('date').drop(
        macro_data.columns[[0, 18, 19, -1, -2, -4, -5, -6, -7, -8, -9, -10, -11]], axis=1)
    macro_data.index = pd.to_datetime(macro_data.index)
    # for column in macro_data.columns:
    #     if macro_data[column].min() <= 0:
    #         macro_data[column] = macro_data[column].apply(lambda x:x-macro_data[column].min()+np.exp(-10))
    context.macro_data = macro_data

    # macro_data = macro_data.set_index('date').drop(macro_data.columns[[0,18,19]],axis=1)
    sent_data = sent_data.set_index('date')
    sent_data.index = pd.to_datetime(sent_data.index)
    context.sent_data = sent_data

    # macro_data = macro_data.set_index('date').drop(macro_data.columns[[0,18,19]],axis=1)
    tech_data = tech_data.set_index('date')
    tech_data.index = pd.to_datetime(tech_data.index)
    context.tech_data = tech_data

    # 技术指标
    # tech_data = pd.read_excel('tech.xls').set_index('date')
    # tech_data.index = pd.to_datetime(tech_data.index)


def algo(context):
    now = context.now
    # 上一交易日
    last_date = get_previous_trading_date(exchange='SHSE', date=now)
    # print("------------\n")
    # print(str(now)+"\n")
    # print(str(last_date)+"\n")
    # N天前的交易日
    last_N_date = my_get_previous_n_trading_date(last_date, counts=context., exchanges='SHSE')
    # 获取持仓
    position = context.account().position(symbol=context.symbol, side=PositionSide_Long)

    print(">>>>>>>>>>>\n")
    print(str(now) + "\n")
    # try:
    prediction, outcome, limit_return = LSTM_predict(context, last_date)
    df = pd.DataFrame([[now, prediction, outcome[-1], outcome, limit_return]],
                      columns=["date", "prediction", 'return_pre', "return_pre7", 'limit_return'])
    context.predictions = pd.concat([context.predictions, df])

    print('prediction:' + str(prediction) + "\n")
    print('recent_'+str(context.T)+'day_return:')
    print(outcome)
    # 若预测值为上涨且空仓则买入
    if prediction == 1:
        context.percent = max(min(200 * np.mean(outcome) / limit_return, 1), 0)
        order_target_percent(symbol=context.symbol, percent=context.percent, order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print("多仓percent")
        print(context.percent)
        # print("多仓volume")
        # print(position.volume)
    # 若预测下跌则清仓
    if prediction == -1 and position:
        context.percent = min(np.max(context.percent - np.mean(outcome) / limit_return, 0), 1)
        order_target_percent(symbol=context.symbol, percent=context.percent, order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print("空仓percent")
        print(context.percent)
        print("空仓volume")




    todayClose = history(context.symbol, frequency='1d', start_time=now, end_time=now, fill_missing='last',
                         df=True).set_index('eob').close
    # print("流程正确")
    # print(todayClose)
    # # 暂停，等待确认
    # input("Press Enter to continue...")
    # 当涨幅大于10%,平掉所有仓位止盈
    if position and len(position) > 0 and todayClose.item() / position['vwap'] >= 1 + context.earn_rate:
        order_close_all()
        print("触发止盈")

    # 当跌幅大于10%时,平掉所有仓位止损
    if position and len(position) > 0 and todayClose.item() / position['vwap'] < 1 + context.sell_rate:
        order_close_all()
        print("触发止损")


def on_order_status(context, order):  # 用于打印交易信息
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
        order_type_word = '限价' if order_type == 1 else '市价'
        print('{}:标的：{}，操作：以{}{}，委托价格：{}，委托数量：{}'.format(context.now, symbol, order_type_word, side_effect,
                                                                      price, volume))


def LSTM_predict(context, last_date):

    return_upper = 0.002
    return_lower = -0.002

    # model和超参数已经在run_strategy中加载


    # 检查CUDA是否可用，并选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # 初始化隐藏状态和细胞状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device).requires_grad_()
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])  # 取LSTM最后一个时间步的输出
            return out

    # 定义模型参数
    input_dim = X.shape[2]
    hidden_dim = 50
    num_layers = 3
    output_dim = 1

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)  # 移动模型到GPU

    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 准备数据
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # 移动数据到GPU
    Y_tensor = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32).to(device)  # 移动数据到GPU
    train_data = TensorDataset(X_tensor[:-T], Y_tensor[:-T])
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    # 训练模型
    epochs = 100
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # 预测
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        X_test_tensor = torch.tensor(X[-T:], dtype=torch.float32).to(device)  # 确保测试数据也在GPU上
        predicted_returns = model(X_test_tensor)
        predicted_returns = predicted_returns.cpu().numpy()  # 将预测结果移回CPU


    # 判断买入卖出信号
    # if (predict_recent_return[0]* predict_recent_return[1] <0
    #   and predict_recent_return[1]* predict_recent_return[2])<0:
    #     return 0,predict_recent_return,0 #震荡
    if min(predicted_returns) > return_upper:
        return 1, predicted_returns, max_return  # 上涨
    elif max(predicted_returns) < return_lower:
        return -1, predicted_returns, min_return  # 下跌
    else:
        return 0, predicted_returns, 0  # 震荡







def on_backtest_finished(context, indicator):
    # 回测业绩指标数据
    data = [
        indicator['pnl_ratio'], indicator['pnl_ratio_annual'], indicator['sharp_ratio'],
        indicator['max_drawdown'], indicator['win_ratio'],
    ]
    print(data)

def run_strategy(training_len, T,withMacro, withTech, withSent):
    # 导入上下文
    from gm.model.storage import context

    context.model_path,context.T, context.window_size, context.hidden_dim, context.num_layers = best_lstm_model(T=None)

    base_data = get_common_data('SHSE.510300','2007-01-01', '2021-01-01',T)
    # base_data的最后一列为未来T日的平均日收益率

    # kmeans聚类，用于后续交易时的判断
    kmeans = KMeans(n_clusters=20, random_state=0, algorithm="elkan").fit(base_data[-1].values.reshape(-1, 1))
    returns = kmeans.cluster_centers_
    context.max_return = np.max(returns)
    context.min_return = np.min(returns)

    # # 初始化时添加相关的控制变量
    # context.result = [] # context.result用以存储超参·········

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
        backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
    '''
    run(strategy_id='f174ce32-eda8-11ee-b6d7-a00af654686a',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='9c0950e38c59552734328ad13ad93b6cc44ee271',
        backtest_start_time='2020-02-28 08:00:00',
        backtest_end_time='2020-03-28 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
    return context.result

if __name__ == '__main__':
    # 参数组合列表
    print('构建参数组：')
    paras_list = []
    # 循环输入参数数值回测
    for window_size in range(50,100,150,200,250):
        for T in range(1,3,5,7):
            paras_list.append([window_size, T])
    print(paras_list)

    # 多进程并行
    print('多进程并行运行参数优化...')
    processes_list = []
    pool = multiprocessing.Pool(processes=12,
                                maxtasksperchild=1)  # create 12 processes
    for i in range(len(paras_list)):
        processes_list.append(
            pool.apply_async(func=run_strategy,
                             args=(paras_list[i][0], paras_list[i][1], paras_list[i][2], paras_list[i][3], paras_list[i][4]))
    pool.close()
    pool.join()
    print('运行结束！')

    # 获取组合的回测结果,并导出
    info = [pro.get()[0] for pro in processes_list]
    info = pd.DataFrame(info,
                        columns=[
                            'pnl_ratio', 'pnl_ratio_annual', 'sharp_ratio',
                            'max_drawdown', 'short', 'long'
                        ])
    print(info)
    info.to_csv('不同参数运行结果.csv', index=False)



