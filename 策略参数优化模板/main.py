# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from gm.api import *

import pandas as pd
import multiprocessing

'''
基本思想：设定所需优化的参数数值范围及步长，将参数数值循环输入进策略，进行遍历回测，
        记录每次回测结果和参数，根据某种规则将回测结果排序，找到最好的参数。
1、定义策略函数
2、多进程循环输入参数数值
3、获取回测报告，生成DataFrame格式
4、排序
本程序以双均线策略为例，优化两均线长短周期参数。
'''


# 原策略中的参数定义语句需要删除！
def init(context):
    # 行情订阅
    context.sec_id = 'SHSE.600519'
    subscribe(symbols=context.sec_id, frequency='1d', count=31)


def on_bar(context, bars):
    # 均线计算
    close = context.data(symbol=context.sec_id, frequency='1d', count=31, fields='close')['close']
    MA_short = close.rolling(context.short).mean().round(2).values
    MA_long = close.rolling(context.long).mean().round(2).values
    # 持仓获取
    position = list(filter(lambda x:x['symbol']==context.sec_id,get_position())) 
    # 逻辑判断并买入
    if not position:
        if MA_short[-1] > MA_long[-1] and MA_short[-2] < MA_long[-2]:
            order_target_percent(symbol=context.sec_id, percent=0.8, order_type=OrderType_Market, position_side=PositionSide_Long)
    elif position:
        if MA_short[-1] < MA_long[-1] and MA_short[-2] > MA_long[-2]:
            order_target_percent(symbol=context.sec_id, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)


# 获取每次回测的报告数据
def on_backtest_finished(context, indicator):
    # 回测业绩指标数据
    data = [
        indicator['pnl_ratio'], indicator['pnl_ratio_annual'],indicator['sharp_ratio'], 
        indicator['max_drawdown'], context.short,context.long
    ]
    # 将超参加入context.result
    context.result.append(data)


def run_strategy(short, long):
    # 导入上下文
    from gm.model.storage import context
    # 用context传入参数
    context.short = short
    context.long = long
    # context.result用以存储超参·········
    context.result = []
    '''
        strategy_id策略ID,由系统生成
        filename文件名,请与本文件名保持一致
        mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID,可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
    '''
    run(strategy_id='8cfb55fc-eff7-11ee-a07c-a00af654686a',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='9c0950e38c59552734328ad13ad93b6cc44ee271',
        backtest_start_time='2023-01-01 08:00:00',
        backtest_end_time='2024-01-01 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=500000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
    return context.result


if __name__ == '__main__':
    # 参数组合列表
    print('构建参数组：')
    paras_list = []
    # 循环输入参数数值回测
    for short in range(5, 10, 2):
        for long in range(10, 21, 5):
            paras_list.append([short, long])
    print(paras_list)

    # 多进程并行
    print('多进程并行运行参数优化...')
    processes_list = []
    pool = multiprocessing.Pool(processes=12,
                                maxtasksperchild=1)  # create 12 processes
    for i in range(len(paras_list)):
        processes_list.append(
            pool.apply_async(func=run_strategy,
                             args=(paras_list[i][0], paras_list[i][1])))
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
    info.to_csv('info.csv', index=False)