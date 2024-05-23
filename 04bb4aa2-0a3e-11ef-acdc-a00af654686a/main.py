# coding=utf-8
from __future__ import print_function, absolute_import

import ast
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import random

import numpy as np

import torch
from gm.api import *
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.get_data import get_common_data, my_get_previous_n_trading_date, my_get_next_n_trading_date
from model.LSTM_base_model import best_lstm_model, prepare_data, LSTMModel, T_values, test_start_date, test_end_date, \
    config_id, val_start_date, val_end_date, lstm_predict,train_start_date,train_end_date
from model.my_xgboost import lstm_to_xgboost, xgb_T_values, xgb_predict, modelDirectory, train_xgboost_classify
import pandas as pd
import xgboost as xgb


# from LSTM_Strategy.main import algo

def init(context):
    # 每天14:50 定时执行algo任务,
    # algo执行定时任务函数，只能传context参数
    # date_rule执行频率，目前暂时支持1d、1w、1m，其中1w、1m仅用于回测，实时模式1d以上的频率，需要在algo判断日期
    # time_rule执行时间， 注意多个定时任务设置同一个时间点，前面的定时任务会被后面的覆盖
    schedule(schedule_func=algo, date_rule='1d', time_rule='14:50:00')

    context.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 止盈幅度
    context.earn_rate = 2

    # 最小涨幅卖出幅度
    context.sell_rate = -0.10

    context.commission_ratio = 0.0001

    context.percent = 0


# aigo
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


# def combined_model_predict(context, the_day_before):
#
#
#     Y_true,Y_pred = xgb_predict(context.target_T,the_day_before,the_day_before )
#     predicted_trend = Y_pred[0]
#
#     return predicted_trend

def on_backtest_finished(context, indicator):
    """
    来自gm.api的内置函数，不允许修改入参
    """
    result = {
        # 'params': context.params,
        'model': context.model,
        'symbol': context.symbol,
        'set': context.set,
        'T': context.T,
        'threshold': context.threshold,
        'pnl_ratio': indicator['pnl_ratio'],
        'sharp_ratio': indicator['sharp_ratio'],
        'max_drawdown': indicator['max_drawdown'],
        'pnl_ratio_annual': indicator['pnl_ratio_annual'],
    }
    # print(result)
    context.result = result


def run_strategy(params):
    # 导入上下文
    from gm.model.storage import context
    # context.params = params
    context.model = params['model']
    context.symbol = params['symbol']
    context.set = params['set']
    context.T = params['T']
    context.threshold = params['threshold']
    if context.model == 'xgb':
        Predictied_trends_Path = f'../results/XGBoost_FLIXNet/Predictied_trends_{context.symbol}_{context.set}_{context.T}_{context.threshold}.xlsx'
        # 如果存在，直接读入
        if os.path.exists(Predictied_trends_Path):
            context.Predictied_trends = pd.read_excel(Predictied_trends_Path)
        else:
            context.Predictied_trends = xgb_predict(context.T,eval(f'{context.set}_start_date'),
                                                     eval(f'{context.set}_end_date'), context.symbol)
            # 保存预测结果
            context.Predictied_trends.to_excel(Predictied_trends_Path)
    elif context.model == 'lstm':
        Predictied_trends_Path = f'../results/LSTM/Predictied_trends_{context.symbol}_{context.set}_{context.T}_{context.threshold}.xlsx'
        # 如果存在，直接读入
        if os.path.exists(Predictied_trends_Path):
            context.Predictied_trends = pd.read_excel(Predictied_trends_Path, index_col=0)
        else:
            context.Predictied_trends = lstm_predict(best_lstm_model(params['T']), eval(f'{context.set}_start_date'),
                                                     eval(f'{context.set}_end_date'))
            # 保存预测结果
            context.Predictied_trends.to_excel(Predictied_trends_Path)
    elif context.model == 'true_testfor_threshold':
        Predictied_trends_Path = f'../results/common/Predictied_trends_{context.symbol}_{context.set}_{context.T}_{context.threshold}.xlsx'
        # 如果存在，直接读入
        if os.path.exists(Predictied_trends_Path):
            context.Predictied_trends = pd.read_excel(Predictied_trends_Path, index_col=0)
        else:
            context.Predictied_trends = pd.DataFrame(get_common_data(context.symbol, eval(f'{context.set}_start_date'),
                                                        eval(f'{context.set}_end_date'), context.T, context.threshold).iloc[:,-1].rename('Y_pred')) # 实际上是将真实值作为预测值，用来测试threshold如何取值收益最高
            # 保存预测结果
            context.Predictied_trends.to_excel(Predictied_trends_Path)

    context.i = 0
    context.result = dict()
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
    run(strategy_id='04bb4aa2-0a3e-11ef-acdc-a00af654686a',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='9c0950e38c59552734328ad13ad93b6cc44ee271',
        backtest_start_time=my_get_next_n_trading_date(date=eval(f'{context.set}_start_date'), counts=1) + ' 08:00:00',
        backtest_end_time=my_get_next_n_trading_date(date=eval(f'{context.set}_end_date'), counts=1) + ' 16:00:00',
        # backtest_start_time='2023-09-14 08:00:00',
        # backtest_end_time='2024-02-01 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
    return context.result


def process_and_save_data(optimization_results, file_name):
    """
    处理一个包含字符串化字典的DataFrame，将其转换为包含原始数据的DataFrame，并保存到Excel文件中。
    """
    data = pd.DataFrame(optimization_results)

    data.to_excel(file_name, index=False)
    return data


def parameter_optimization(paras_list, processes=2):
    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = {executor.submit(run_strategy, params): params for params in paras_list}
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                if isinstance(result, dict) and 'sharp_ratio' in result:
                    results.append(result)
                else:
                    raise ValueError("Result is not a dictionary with a 'sharp_ratio' key")
            except Exception as exc:
                print(f"任务产生异常: {exc}")
        return results


# 函数：执行参数优化
def optimize_and_save(paras_list, save_path, processes=8):
    results = parameter_optimization(paras_list, processes)
    process_and_save_data(results, save_path)


# 函数：重新运行优化对于空的或为0的sharp_ratio
def rerun_optimization_for_empty_or_zero_sharp_ratio(file_path):
    data = pd.read_excel(file_path)
    mask = data['sharp_ratio'].isnull() | (data['sharp_ratio'] == 0)
    fail_count = 0
    retry_limit = 1000
    while not data.loc[mask].empty:
        if fail_count >= retry_limit:
            raise Exception(f"连续{retry_limit}尝试后，仍有sharp_ratio为空或为0的行，中断运行。")

        paras_list = data.loc[mask].to_dict('records')
        new_results = parameter_optimization(paras_list, processes=4)

        # 确保我们处理的是字典列表
        if all(isinstance(result, dict) for result in new_results):
            for idx, result in enumerate(new_results):
                # 确保每个结果包含 'sharp_ratio'
                if 'sharp_ratio' in result:
                    # 将新结果填充到原始数据中,整行都要覆盖
                    data.loc[data.index[mask][idx], :] = pd.Series(result)
                else:
                    raise KeyError("Expected 'sharp_ratio' in the result dictionaries")
        else:
            raise TypeError("Expected a list of dictionaries from parameter_optimization")

        mask = data['sharp_ratio'].isnull() | (data['sharp_ratio'] == 0)
        fail_count += 1

    data.to_excel(file_path, index=False)


def backtest(models, symbols, sets, T_values, threshold_values, save_path):
    # 参数列表，可以是从配置文件读取的
    paras_list = [
        {'model': model, 'symbol': symbol, 'set': set, 'T': T, 'threshold': threshold}
        for model in models
        for symbol in symbols
        for set in sets
        for T in T_values
        for threshold in threshold_values
    ]

    # 第一次运行优化并保存
    optimize_and_save(paras_list, save_path, 4)

    # 检查并重新运行空的或为0的sharp_ratio
    rerun_optimization_for_empty_or_zero_sharp_ratio(save_path)
    print(f"所有策略已经运行完毕! 结果保存在{save_path}中！")


if __name__ == '__main__':
    # symbols = ['SHSE.600519', 'SZSE.300750', 'SHSE.601318', 'SHSE.600036', 'SZSE.000858', 'SHSE.601012', 'SZSE.000333','SHSE.600900', 'SHSE.601166', 'SZSE.002594']
    symbols = ['SHSE.510300']
    # sets = [ 'val','test']
    sets = ['val']

    # models = ['lstm', 'xgb']

    models = ['true_testfor_threshold']
    thresholds = np.arange(0.001, 0.021, 0.001)
    # thresholds = [0.001]
    backtest_T_values = [1,3,7,15]
    # backtest_T_values = T_values
    backtest(models, symbols, sets,backtest_T_values , thresholds,
             f'../results/common/optimization_thresholds_results_{config_id}.xlsx')

