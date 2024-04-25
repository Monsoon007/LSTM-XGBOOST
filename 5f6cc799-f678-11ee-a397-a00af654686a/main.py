# coding=utf-8
from __future__ import print_function, absolute_import

import ast
import multiprocessing
import os
from random import random

import numpy as np

import torch
from gm.api import *
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.get_data import get_common_data, my_get_previous_n_trading_date
from model.LSTM_base_model import best_lstm_model, prepare_data, LSTMModel, T_values, test_start_date, test_end_date, \
    config_id, val_start_date, val_end_date
import pandas as pd


def init(context):
    # 每天14:50 定时执行algo任务,
    # algo执行定时任务函数，只能传context参数
    # date_rule执行频率，目前暂时支持1d、1w、1m，其中1w、1m仅用于回测，实时模式1d以上的频率，需要在algo判断日期
    # time_rule执行时间， 注意多个定时任务设置同一个时间点，前面的定时任务会被后面的覆盖
    schedule(schedule_func=algo, date_rule='1d', time_rule='14:50:00')

    # 股票标的
    # context.symbol = 'SHSE.600519'
    context.symbol = 'SHSE.510300'
    # random.seed(1)

    # 导入上下文
    from gm.model.storage import context

    model_dict = best_lstm_model(T=context.T)
    context.model_path = model_dict['model_path']
    context.window_size = model_dict['window_size']
    context.hidden_dim = model_dict['hidden_dim']
    context.num_layers = model_dict['num_layers']
    # context.T = model_dict['T']
    context.model_path = os.path.join('../model/', context.model_path)
    context.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # base_data = get_common_data('SHSE.510300', '2008-01-01', '2020-01-01', context.T)
    # base_data的最后一列为未来T日的平均日收益率

    # # kmeans聚类，用于后续交易时的判断
    # kmeans = KMeans(n_clusters=20, random_state=0, algorithm="elkan").fit(base_data.iloc[:, -1].values.reshape(-1, 1))
    # returns = kmeans.cluster_centers_
    # context.max_return = np.max(returns)
    # context.min_return = np.min(returns)

    # 止盈幅度
    context.earn_rate = 2

    # 最小涨幅卖出幅度
    context.sell_rate = -0.10

    context.commission_ratio = 0.0001

    context.percent = 0


def algo(context):
    now = context.now
    # 上一交易日
    the_day_before = get_previous_trading_date(exchange='SHSE', date=now)
    # 获取持仓
    position = context.account().position(symbol=context.symbol, side=PositionSide_Long)
    # print(">>>>>>>>>>>\n")
    # print(str(now) + "\n")
    # try:
    prediction, outcome = LSTM_predict(context, the_day_before)

    # 若预测值为上涨则买入
    if prediction == 1:
        context.percent = 1
        order_target_percent(symbol=context.symbol, percent=context.percent, order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        # print("多仓percent")
        # print(context.percent)

    # 若预测下跌则清仓
    if prediction == -1 and position:
        context.percent = 0
        order_target_percent(symbol=context.symbol, percent=context.percent, order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        # print("空仓percent")
        # print(context.percent)
        # print("空仓volume")

    todayClose = history(context.symbol, frequency='1d', start_time=now, end_time=now, fill_missing='last',
                         df=True).set_index('eob').close

    # 当涨幅大于,平掉所有仓位止盈
    if position and len(position) > 0 and todayClose.item() / position['vwap'] >= 1 + context.earn_rate:
        order_close_all()
        # print("触发止盈")

    # 当跌幅大于,平掉所有仓位止损
    if position and len(position) > 0 and todayClose.item() / position['vwap'] < 1 + context.sell_rate:
        order_close_all()
        # print("触发止损")


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
        # print(f'{symbol} {side_effect} {volume}手, 价格{price}, 目标仓位{target_percent}, {order_type_word}委托, 成交')


def LSTM_predict(context, the_day_before):
    """
    last_date: 上一个交易日,
    """
    return_upper = context.threshold
    return_lower = -context.threshold

    # modelPath和超参数已经在run_strategy中加载

    # X_test, _ = prepare_data(context.symbol, my_get_previous_n_trading_date(last_date, context.window_size + 100),
    #                          my_get_previous_n_trading_date(last_date,1), context.T, context.window_size)
    X_test, _ = prepare_data(context.symbol, my_get_previous_n_trading_date(the_day_before, context.window_size + 100),
                             the_day_before, context.T, context.window_size) #已经是上一个交易日，避免了利用当天收盘价等“未来信息”
    # 加载模型
    model = LSTMModel(X_test.shape[2], int(context.hidden_dim), context.num_layers, 1, context.device).to(
        context.device)
    model.load_state_dict(torch.load(context.model_path))
    model.eval()  # 进入评估模式

    outputs = model(X_test.float().to(context.device))
    predicted_return = outputs.cpu().detach().numpy().flatten()
    predicted_return = predicted_return[-1]
    # 打印日期和预测值
    # print("预测日期：", last_date, f"预测未来{context.T}日的平均日收益率：", predicted_return)
    if predicted_return > return_upper:
        return 1, predicted_return
    elif predicted_return < return_lower:
        return -1, predicted_return  # 下跌
    else:
        return 0, predicted_return  # 震荡


def on_backtest_finished(context, indicator):
    """
    来自gm.api的内置函数，不允许修改入参
    """
    result = {
        'params': context.params,
        'T': context.T,
        'threshold': context.threshold,
        'pnl_ratio': indicator['pnl_ratio'],
        'sharp_ratio': indicator['sharp_ratio'],
        'max_drawdown': indicator['max_drawdown'],
        'pnl_ratio_annual': indicator['pnl_ratio_annual']

    }
    print(result)
    context.results.append(result)


def parameter_optimization(paras_list):
    pool = multiprocessing.Pool(processes=4)  # 根据系统性能调整进程数
    results = []
    pbar = tqdm(total=len(paras_list), desc="Optimizing")

    def update(*args):
        pbar.update()

    for params in paras_list:
        result = pool.apply_async(run_strategy, args=(params,), callback=update)
        results.append(result)
    pool.close()
    pool.join()
    pbar.close()
    return [result.get() for result in results]


def run_strategy(params):
    # 导入上下文
    from gm.model.storage import context

    context.params = params
    context.T = params['T']
    context.threshold = params['threshold']

    context.results = []

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
    run(strategy_id='5f6cc799-f678-11ee-a397-a00af654686a',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='9c0950e38c59552734328ad13ad93b6cc44ee271',
        backtest_start_time=val_start_date + ' 08:00:00',
        backtest_end_time=val_end_date + ' 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
    return context.results


def process_and_save_data(optimization_results, file_name):
    """
    处理一个包含字符串化字典的DataFrame，将其转换为包含原始数据的DataFrame，并保存到Excel文件中。
    """
    data = pd.DataFrame(optimization_results)

    # 检查第一行，如果是字符串则转换回字典，如果已经是字典则直接使用
    first_row_data = data.iloc[0, 0]
    if isinstance(first_row_data, str):
        dict_data = ast.literal_eval(first_row_data)
    else:
        dict_data = first_row_data  # 假定输入的就是字典格式

    keys = list(dict_data.keys())
    df = pd.DataFrame(columns=keys)

    for i in range(len(data)):
        row_data = data.iloc[i, 0]
        if isinstance(row_data, str):
            row_data = ast.literal_eval(row_data)
        temp_df = pd.DataFrame([row_data])
        df = pd.concat([df, temp_df], ignore_index=True)

    df.to_excel(file_name, index=False)


if __name__ == '__main__':
    # 参数列表，可以是从配置文件读取的
    paras_list = [
        {'T': T, 'threshold': threshold}
        for T in T_values
        # for T in [4]
        for threshold in np.arange(0.001, 0.002, 0.001)
    ]
    optimization_results = parameter_optimization(paras_list)

    # process_and_save_data(optimization_results, f'../results/no_kmeans/optimization_results_{config_id}.xlsx')
    process_and_save_data(optimization_results, f'../results/LSTM/lstm_val_backtest.xlsx')

    # process_and_save_data(optimization_results, f'../results/optimization_results_3.xlsx')
