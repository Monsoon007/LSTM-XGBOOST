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

from data.get_data import get_common_data, my_get_previous_n_trading_date, my_get_next_n_trading_date
from model.LSTM_base_model import best_lstm_model, prepare_data, LSTMModel, T_values, test_start_date, test_end_date, \
    config_id, val_start_date, val_end_date, lstm_predict
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

    context.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    predictied_trend = context.Predictied_trends['Y_pred'][context.i]
    # predictied_trend = 2
    context.i = context.i + 1

    # 若预测值为上涨则买入
    if predictied_trend == 2 and not position:
        context.percent = 1
        order_target_percent(symbol=context.symbol, percent=context.percent, order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        # print("多仓percent")
        # print(context.percent)

    # 若预测下跌则清仓
    if predictied_trend == 0 and position:
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


# def LSTM_predict(context, the_day_before):
#     """
#     last_date: 上一个交易日,
#     """
#     return_upper = context.threshold
#     return_lower = -context.threshold
#
#     result = lstm_predict(context.model_dict,the_day_before,the_day_before)
#
#     predicted_trend = result['Y_pred'][0]
#
#     return predicted_trend


def on_backtest_finished(context, indicator):
    """
    来自gm.api的内置函数，不允许修改入参
    """
    result = {
        'params': context.params,
        'set': context.set,
        'T': context.T,
        'threshold': context.threshold,
        'pnl_ratio': indicator['pnl_ratio'],
        'sharp_ratio': indicator['sharp_ratio'],
        'max_drawdown': indicator['max_drawdown'],
        'pnl_ratio_annual': indicator['pnl_ratio_annual'],

    }
    # print(result)
    context.results.append(result)


from concurrent.futures import ProcessPoolExecutor, as_completed
def parameter_optimization(paras_list, processes=4):
    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = {executor.submit(run_strategy, params): params for params in paras_list}
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"任务 {futures[future]} 完成，结果: {result}")
            except Exception as exc:
                print(f"任务 {futures[future]} 生成异常: {exc}")
        return results


def run_strategy(params):
    # 导入上下文
    from gm.model.storage import context

    context.params = params
    context.T = params['T']
    context.threshold = params['threshold']
    context.set = params['set']
    Predictied_trends_Path = f'../results/LSTM/lstm_{context.T}_{context.set}_prediction_{config_id}.xlsx'
    set = params['set']
    # 如果存在，直接读入
    if os.path.exists(Predictied_trends_Path):
        context.Predictied_trends = pd.read_excel(Predictied_trends_Path, index_col=0)
    else:
        context.Predictied_trends = lstm_predict(best_lstm_model(params['T']), eval(f'{set}_start_date'),
                                                 eval(f'{set}_end_date'))
        # 保存预测结果
        context.Predictied_trends.to_excel(Predictied_trends_Path)
    context.i = 0
    context.results = []

    # backtest_start_time = my_get_next_n_trading_date(date=eval(f'{set}_start_date'), counts=1) + ' 08:00:00'
    # backtest_end_time = my_get_next_n_trading_date(date=eval(f'{set}_end_date'), counts=1) + ' 16:00:00'
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
        backtest_start_time=my_get_next_n_trading_date(date=eval(f'{set}_start_date'), counts=1) + ' 08:00:00',
        backtest_end_time=my_get_next_n_trading_date(date=eval(f'{set}_end_date'), counts=1) + ' 16:00:00',
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
    sets = ['val']

    paras_list = [
        {'set': set, 'T': T, 'threshold': threshold}
        for set in sets
        # for T in T_values
        for T in [9,24]
        # for threshold in np.arange(0.0001, 0.0035, 0.0005)
        for threshold in np.arange(0.001, 0.002, 0.001)

    ]
    optimization_results = parameter_optimization(paras_list, processes=8)
    savePath = f'../results/LSTM/lstm_val_backtest_{config_id}_support.xlsx'
    # process_and_save_data(optimization_results, f'../results/no_kmeans/optimization_results_{config_id}.xlsx')
    process_and_save_data(optimization_results, savePath)
    print(f"所有策略已经运行完毕! 结果保存在{savePath}中！")
    # process_and_save_data(optimization_results, f'../results/optimization_results_3.xlsx')
