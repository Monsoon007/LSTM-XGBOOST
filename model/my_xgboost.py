import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance

from model.LSTM_base_model import T_values, best_lstm_model, LSTMModel, prepare_data, symbol, train_start_date, \
    train_end_date, val_start_date, val_end_date
from torch.utils.tensorboard import SummaryWriter






def lstm_predict(model_dict, start_date, end_date):
    X_test, Y_true = prepare_data(symbol, start_date, end_date, model_dict['T'], model_dict['window_size'])
    # 检查如果GPU没有使用成功，raise error
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise ValueError('GPU not available')
    model = LSTMModel(X_test.shape[2], int(model_dict['hidden_dim']), model_dict['num_layers'], 1, device).to(device)
    model.load_state_dict(torch.load(os.path.join('../model/', model_dict['model_path'])))
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.float().to(device))
        Y_pred = outputs.cpu().detach().numpy().flatten()
    # 将Y_test处理后，与Y_pred合并为一个DataFrame

    Y_true = Y_true.flatten()
    Y_true = pd.DataFrame(Y_true, columns=['Y_true'])
    Y_pred = pd.DataFrame(Y_pred, columns=['Y_pred'])
    result = pd.concat([Y_true, Y_pred], axis=1)
    result.index = pd.date_range(start=start_date, periods=len(result), freq='B')
    return result

def lstm_to_xgboost(start_date, end_date, target_T):
    X = pd.DataFrame()
    for model_dict in best_lstms:
        Y_true_pred = lstm_predict(model_dict, start_date, end_date)
        # 将Y_true_pred['Y_pred']加入X，并加上列名'model_T_Y_pred'
        X['model_'+str(model_dict['T'])+'_Y_pred'] = Y_true_pred['Y_pred']
        if model_dict['T'] == target_T:
            Y = Y_true_pred['Y_true']
    return X,Y


# 选出各个T下最优的模型，加入列表
best_lstms = []
for T in T_values:
    best_lstms.append(best_lstm_model(T))


target_T = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# XGBoost训练过程的评估指标收集
evals_result = {}  # 用于存储评估指标
if __name__ == '__main__':

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
        # 'predictor': 'gpu_predictor'  # 预测时使用的GPU算法
    }
    num_round = 100  # 训练迭代次数

    # 训练模型
    bst = xgb.train(param, dtrain, num_round, evals=[(dval, 'eval')], early_stopping_rounds=10,evals_result=evals_result)

    # 保存模型
    bst.save_model('xgb_model.json')

    # 绘制评估指标（例如：RMSE）随训练迭代的变化
    epochs = len(evals_result['eval']['rmse'])
    x_axis = range(0, epochs)
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, evals_result['eval']['rmse'], label='Test RMSE')
    plt.legend()
    plt.ylabel('RMSE')
    plt.title('XGBoost RMSE')
    plt.show()

    # 展示XGBoost模型的特征重要性
    plot_importance(bst)
    plt.show()