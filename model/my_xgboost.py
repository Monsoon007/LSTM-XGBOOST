import json
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from xgboost import plot_importance

from data.get_data import get_common_data
# from graphviz import Source

from model.LSTM_base_model import LSTMModel, prepare_data, symbol, train_start_date, \
    train_end_date, val_start_date, val_end_date, best_lstms, lstm_predict, test_start_date, test_end_date
from torch.utils.tensorboard import SummaryWriter


def lstm_to_xgboost(start_date, end_date, target_T):
    data = get_common_data(symbol, start_date, end_date, target_T)

    # 取Y为data的最后一列
    Y = data.iloc[:, -1]

    X = pd.DataFrame()
    for model_dict in best_lstms:
        Y_true_pred = lstm_predict(model_dict, start_date, end_date)
        # 将Y_true_pred['Y_pred']加入X，并加上列名'model_T_Y_pred'
        X['lstm_' + str(model_dict['T']) + '_Y_pred'] = Y_true_pred['Y_pred']
        # if model_dict['T'] == target_T:
        #     Y = Y_true_pred['Y_true']
    return X, Y


def plot_prediction(Y_true, Y_pred, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(Y_true, Y_pred, alpha=0.5)
    plt.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()


def plot_residuals(Y_true, Y_pred):
    residuals = Y_true - Y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(Y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=Y_pred.min(), xmax=Y_pred.max(), colors='red', linestyles='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()


def train_xgboost(target_T=7):
    # 准备数据
    X_train, Y_train = lstm_to_xgboost(train_start_date, train_end_date, target_T)
    X_val, Y_val = lstm_to_xgboost(val_start_date, val_end_date, target_T)

    # XGBOOST
    # 将数据转换为DMatrix对象，XGBoost专用的数据结构
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(X_val, label=Y_val)

    # XGBoost的参数，包括使用GPU的配置
    param = {
        'max_depth': 5,  # 树的最大深度
        'eta': 0.3,  # 学习率
        'objective': 'reg:squarederror',  # 回归任务的损失函数类型
        'eval_metric': 'rmse',  # 评估指标为均方根误差
        "device": "cuda",
        'tree_method': 'hist',  # 使用GPU加速的直方图优化算法
    }
    num_round = 100  # 训练迭代次数

    # 训练模型
    bst = xgb.train(param, dtrain, num_round, evals=[(dval, 'eval')])

    # 保存模型
    bst.save_model(f'../model/xgb_models/xgb_model_{target_T}.json')

    # # 生成Graphviz格式
    # graph = xgb.to_graphviz(bst, num_trees=0)
    # # 保存或显示图像
    # graph.render(filename='tree', format='png', cleanup=True)
    # 保存XGBoost模型的特征重要性图片
    plot_importance(bst)
    plt.savefig(f'xgb_model_{target_T}_importance.png')


def xgb_predict(target_T, start_date, end_date):
    """
    返回Y_true, Y_pred
    """
    # 准备数据
    X, Y_true = lstm_to_xgboost(start_date, end_date, target_T)
    # 加载模型
    bst = xgb.Booster()
    # bst.load_model(f'../model/xgb_models/xgb_model_{target_T}.json')
    if os.path.exists(f'../model/xgb_models/xgb_model_{target_T}.json'):
        print(f'Loading xgb_model_{target_T}.json')
        bst.load_model(f'../model/xgb_models/xgb_model_{target_T}.json')
    else:
        print(f'Training xgb_model_{target_T}.json')
        train_xgboost(target_T)
        bst.load_model(f'../model/xgb_models/xgb_model_{target_T}.json')
    # 将数据转换为DMatrix对象，XGBoost专用的数据结构
    dtest = xgb.DMatrix(X)
    # 预测
    Y_pred = bst.predict(dtest)
    return Y_true, Y_pred


def xgb_evaluate(target_T, set="val", only_r2=False):
    """
    mode为
    """
    if set == "val":
        start_date = val_start_date
        end_date = val_end_date
        pathStr = 'Val'
    elif set == "test":
        start_date = test_start_date
        end_date = test_end_date
        pathStr = 'Test'
    Y_true, Y_pred = xgb_predict(target_T, start_date, end_date)

    # 计算R2
    from sklearn.metrics import r2_score
    r2 = r2_score(Y_true, Y_pred)
    # print(f"R2: {r2:.4f}")
    if only_r2:
        return r2
    # 绘制折线图
    plt.figure(figsize=(15, 6))
    plt.plot(Y_true, label='True Value', color='#1f77b4', linewidth=2)
    plt.plot(Y_pred, label='Predicted Value', color='#ff7f0e', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'XGBoost_{target_T} Prediction VS True in {pathStr} set')
    # 在图的左下角添加R²值
    plt.text(0.1, 0.05, f'$R²: {r2:.4f}$', transform=plt.gca().transAxes, fontsize=16, color='green')
    plt.legend()
    # 如果路径不存在，则先创建
    if not os.path.exists(f'../results/XGBoost_{pathStr}_evaluate'):
        os.makedirs(f'../results/XGBoost_{pathStr}_evaluate')
    # 保存图片
    plt.savefig(f'../results/XGBoost_{pathStr}_evaluate/{pathStr}_xgboost_{target_T}_prediction_vs_true.png')
    return r2


def get_xgb_r2_dict(set='val', updated=False):
    # 定义文件路径
    file_path = f'../results/XGBoost_FLIXNet/XGBoost_{set}_r2_dict.csv'

    # 如果不需要更新且文件已存在，则直接从CSV文件加载数据
    if not updated and os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)  # 假设第一列是索引
        r2_dict = df['R2'].to_dict()
        print(f'Loaded XGBoost {set} r2 dict from {file_path}')
        return r2_dict
    print(f'Calculating XGBoost {set} r2 dict...')
    # 如果需要更新或文件不存在，则重新计算
    r2_dict = {}
    for T in tqdm(xgb_T_values, desc=f'xgb_evaluate in {set} set'):
        r2 = xgb_evaluate(T, set, only_r2=True)
        r2_dict[int(T)] = r2  # 确保键是Python的int类型

    # 将字典转换为DataFrame，并保存到CSV文件
    df = pd.DataFrame(list(r2_dict.items()), columns=['T', 'R2'])
    df.to_csv(file_path, index=False)

    return r2_dict

def get_xgb_r2_series(set='val', updated=False):
    r2_dict = get_xgb_r2_dict(set, updated)
    r2_series = pd.Series(r2_dict)
    # 索引名字为T，列名为R2
    r2_series.index.name = 'T'
    r2_series.name = 'R2'
    return r2_series

best_lstms = best_lstms(topK=30, T_values=list(np.arange(1, 31, 3)))
xgb_T_values = list(np.arange(1, 31))

if __name__ == '__main__':
    for target_T in tqdm(xgb_T_values, desc='Training XGBoost for different target_T values'):
        # 查看是否已经训练
        if os.path.exists(f'../model/xgb_models/xgb_model_{target_T}.json'):
            continue
        else:
            train_xgboost(target_T)
