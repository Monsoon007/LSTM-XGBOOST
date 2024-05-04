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

    # 股票标的
    # context.symbol = 'SHSE.600519'
    # context.symbol = 'SHSE.510300'
    # random.seed(1)

    # 导入上下文
    from gm.model.storage import context

    context.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xgb_path = f'{modelDirectory}/xgb_model_{context.target_T}.json'

    # 如果xgb_path不存在
    if not os.path.exists(xgb_path):
        train_xgboost_classify(context.target_T)

    context.bst = xgb.Booster()
    context.bst.load_model(xgb_path)

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
    predictied_trend = combined_model_predict(context, the_day_before)

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


def combined_model_predict(context, the_day_before):


    Y_true,Y_pred = xgb_predict(context.target_T,the_day_before,the_day_before )
    predicted_trend = Y_pred[0]

    return predicted_trend

def on_backtest_finished(context, indicator):
    """
    来自gm.api的内置函数，不允许修改入参
    """
    result = {
        'params': context.params,
        'pnl_ratio': indicator['pnl_ratio'],
        'sharp_ratio': indicator['sharp_ratio'],
        'max_drawdown': indicator['max_drawdown'],
        'pnl_ratio_annual': indicator['pnl_ratio_annual'],
        'T': context.target_T,
        'symbol': context.symbol,
    }
    # 将result转换为dataframe
    # df = pd.DataFrame([result])
    # # 保存结果
    # path = f'../results/combined_model/xgb_test_strategy_{config_id}_test.xlsx'
    # # 追加保存
    # if os.path.exists(path):
    #     df.to_excel(path, index=False, header=False, mode='a')
    # else:
    #     df.to_excel(path, index=False)
    print(result)
    context.results.append(result)


def run_strategy(params):
    # 导入上下文
    from gm.model.storage import context

    context.params = params
    context.symbol = params['symbol']
    context.target_T = int(params['target_T'])
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
    run(strategy_id='30047b77-f74e-11ee-8f7f-a00af654686a',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='9c0950e38c59552734328ad13ad93b6cc44ee271',
        backtest_start_time=val_start_date + ' 08:00:00',
        backtest_end_time=val_end_date + ' 16:00:00',
        # backtest_start_time='2023-09-14 08:00:00',
        # backtest_end_time='2024-02-01 16:00:00',
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


def parameter_optimization(paras_list, processes=2):
    pool = multiprocessing.Pool(processes)  # 根据系统性能调整进程数
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


if __name__ == '__main__':
    # 贵州茅台
    # 证券代码: 600519
    # 上市地: 上海证券交易所
    # 宁德时代
    # 证券代码: 300750
    # 上市地: 深圳证券交易所
    # 中国平安
    # 证券代码: 601318
    # 上市地: 上海证券交易所
    # 招商银行
    # 证券代码: 600036
    # 上市地: 上海证券交易所
    # 五粮液
    # 证券代码: 000858
    # 上市地: 深圳证券交易所
    # 隆基绿能
    # 证券代码: 601012
    # 上市地: 上海证券交易所
    # 美的集团
    # 证券代码: 000333
    # 上市地: 深圳证券交易所
    # 长江电力
    # 证券代码: 600900
    # 上市地: 上海证券交易所
    # 兴业银行
    # 证券代码: 601166
    # 上市地: 上海证券交易所
    # 比亚迪
    # 证券代码: 002594
    # 上市地: 深圳证券交易所

    # symbols = ['SHSE.600519', 'SZSE.300750', 'SHSE.601318', 'SHSE.600036', 'SZSE.000858', 'SHSE.601012', 'SZSE.000333','SHSE.600900', 'SHSE.601166', 'SZSE.002594']
    symbols = ['SHSE.510300']
    # 参数列表，可以是从配置文件读取的
    paras_list = [
        {'symbol':symbol,'target_T': target_T, 'threshold': threshold}
        # for target_T in xgb_T_values

        for symbol in symbols
        for target_T in [1,2,3,4,5]
        for threshold in np.arange(0.001, 0.002, 0.001)
    ]
    optimization_results = parameter_optimization(paras_list, processes=3)

    # process_and_save_data(optimization_results, f'../results/combined_model/xgb_test_strategy_{config_id}.xlsx')
    process_and_save_data(optimization_results, f'../results/XGBoost_FLIXNet/xgb_val_backtest_{config_id}.xlsx')
    print(f"所有策略已经运行完毕! 结果保存在'../results/XGBoost_FLIXNet/xgb_val_backtest_{config_id}.xlsx'")
