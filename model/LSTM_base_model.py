# import itertools
import inspect
import json
import os
import subprocess
from gm.api import *

import pandas as pd
from tqdm import tqdm
from tqdm.contrib import itertools

from model.LSTM_data_handle import data_for_lstm
from data.get_data import get_common_data, my_get_previous_n_trading_date
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def prepare_data(symbol, start_date, end_date, T, window_size):
    """
    数据会返回一个元组，第一个元素是X，第二个元素是Y。
    X的形状为(样本数量, 窗口大小, 特征数量)，Y的形状为(样本数量,)。
    数据会自动前移window_size天，使给出的Y能覆盖要求的start_date到end_date的数据。
    """
    # print(
    #     f"Preparing data for symbol: {symbol}, start_date: {start_date}, end_date: {end_date}, T: {T}, window_size: {window_size}")
    # 设置token
    set_token('9c0950e38c59552734328ad13ad93b6cc44ee271')
    data = get_common_data(symbol, my_get_previous_n_trading_date(start_date, counts=window_size), end_date, T)
    # print(f"Data shape: {data.shape}")
    X, Y = data_for_lstm(data, window_size)
    # print(f"Prepared data X: {X}, Y: {Y}")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)


def train(model, device, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 每n个批次记录一次训练损失
        if batch_idx % 100 == 0:
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')


def validate(model, device, val_loader, criterion, writer, epoch):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    writer.add_scalar('Validation Loss', val_loss, epoch)

    return val_loss


def main():
    output_dim = 1
    epochs = 100
    batch_size = 16
    lr = 0.01  # 学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 用于存储所有最佳模型结果的列表
    best_models_results = []

    # 检查模型保存目录是否存在，如果不存在，则创建它
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for T, window_size, hidden_dim, num_layers in itertools.product(T_values, window_sizes, hidden_dims,
                                                                    num_layers_list):
        # 检查该种参数模型是否已经训练过
        model_path = os.path.join(model_save_dir,
                                  f'best_lstm_model_T{T}_window{window_size}_hidden{hidden_dim}_layers{num_layers}.pth')

        if os.path.exists(model_path):
            X_val, Y_val = prepare_data(symbol, val_start_date, val_end_date, T, window_size)

            val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size, shuffle=False)

            # 加载模型
            model = LSTMModel(input_dim=X_val.shape[2], hidden_dim=hidden_dim, num_layers=num_layers,
                              output_dim=output_dim, device=device)
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

            # 设置损失函数
            criterion = nn.MSELoss()

            # 计算在验证集上的MSE loss
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # 将该模型的结果记录到best_models_results中
            best_models_results.append({
                'T': T,
                'window_size': window_size,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'best_val_loss': val_loss,
                'best_model_path': model_path
            })
            continue

        # 创建tensorboard writer
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = f'runs_{config_id}/lstm_experiment_{current_time}_T{T}_window{window_size}_hidden{hidden_dim}_layers{num_layers}'
        writer = SummaryWriter(log_dir)

        # 准备数据
        X_train, Y_train = prepare_data(symbol, train_start_date, train_end_date, T, window_size)
        X_val, Y_val = prepare_data(symbol, val_start_date, val_end_date, T, window_size)
        X_test, Y_test = prepare_data(symbol, test_start_date, test_end_date, T, window_size)

        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size, shuffle=False)

        # 检查val_loader是否为空
        if len(val_loader) == 0:
            print(len(X_val), len(Y_val), val_start_date, val_end_date)
        else:
            print(f"Validation loader has {len(val_loader)} batches.")

        model = LSTMModel(X_train.shape[2], hidden_dim, num_layers, output_dim, device).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        best_val_loss = float("inf")
        best_model_path = ""  # 验证集上最佳模型的保存路径
        for epoch in range(epochs):
            train(model, device, train_loader, criterion, optimizer, epoch, writer)
            val_loss = validate(model, device, val_loader, criterion, writer, epoch)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存验证集上最佳模型
                best_model_path = os.path.join(model_save_dir,
                                               f'best_lstm_model_T{T}_window{window_size}_hidden{hidden_dim}_layers{num_layers}.pth')
                torch.save(model.state_dict(), best_model_path)
        # 保存每个超参数组合和其最佳验证损失的记录
        best_models_results.append({
            'T': T,
            'window_size': window_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'best_val_loss': best_val_loss,
            'best_model_path': best_model_path
        })

        # 加载最佳模型
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        # 打印best_val_loss
        print(
            f'Best Validation Loss for T {T}, window_size {window_size}, hidden_dim {hidden_dim}, num_layers {num_layers}: {best_val_loss}')
        test(model, device, test_loader, criterion)

        writer.close()

    # 所有训练完成后，将最佳模型的结果保存到Excel文件中
    results_df = pd.DataFrame(best_models_results)
    results_df.to_excel(results_path, index=False)


def get_lstm_model(T, window_size, hidden_dim, num_layers):
    """
    暴露给外部的接口，用于获取指定超参数的LSTM模型地址
    """
    # 检查模型保存目录是否存在，如果不存在，则创建它
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 检查该种参数模型是否已经训练过，如果训练过，则返回该模型地址
    model_path = os.path.join(model_save_dir,
                              f'best_lstm_model_T{T}_window{window_size}_hidden{hidden_dim}_layers{num_layers}.pth')
    if os.path.exists(model_path):
        return model_path
    else:
        raise ValueError('Model not found!')


def best_lstm_model(T=None):
    """
    获取最佳LSTM模型的地址及其超参数
    """
    # 读取最佳模型的结果
    best_models_results = pd.read_excel(results_path)

    # 如果T==None，则返回所有模型中最佳的模型
    if T is None:
        best_model = best_models_results.loc[best_models_results['best_val_loss'].idxmin()]
    else:
        # 否则返回T等于指定值的最佳模型
        best_model = best_models_results.loc[
            best_models_results[best_models_results['T'] == T]['best_val_loss'].idxmin()]

    # 返回一个字典，包含最佳模型的地址及其超参数
    return {
        'T': best_model['T'],
        'window_size': best_model['window_size'],
        'hidden_dim': best_model['hidden_dim'],
        'num_layers': best_model['num_layers'],
        'model_path': best_model['best_model_path'],
        'val_r2': lstm_evaluate(best_model, set="val", only_r2=True),
    }


def best_lstms(topK=10):
    """
    获取所有T条件下的最佳LSTM模型的地址及其超参数、val_r2
    可以设置topK参数，返回val_r2最高的前topK个模型，默认为10
    """
    best_lstms = []
    for T in tqdm(T_values, desc=f'Running {inspect.currentframe().f_code.co_name}'):
        best_model = best_lstm_model(T)
        best_lstms.append(best_model)
    # 按照val_r2的第二个位置的r2进行降序
    best_lstms.sort(key=lambda x: x['val_r2'][1], reverse=True)
    # 取前topK个
    best_lstms = best_lstms[:topK]
    return best_lstms


def lstm_predict(model_dict, start_date, end_date):
    """
    model_dict: 包含模型超参数的字典
    返回结果为包含模型预测值和真实值的DataFrame
    """
    X_test, Y_true = prepare_data(symbol, start_date, end_date, model_dict['T'], model_dict['window_size'])
    # 检查如果GPU没有使用成功，raise error
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise ValueError('GPU not available')
    try:
        model_path = model_dict['model_path']
    except KeyError:
        model_path = model_dict['best_model_path']
    model = LSTMModel(X_test.shape[2], int(model_dict['hidden_dim']), model_dict['num_layers'], 1, device).to(device)
    model.load_state_dict(torch.load(os.path.join('../model/', model_path)))
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


def lstm_evaluate(model_dict, set="val", only_r2=False):
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
    df = lstm_predict(model_dict, start_date, end_date)

    # 计算R2
    from sklearn.metrics import r2_score
    from matplotlib import pyplot as plt

    r2 = r2_score(df['Y_true'], df['Y_pred'])
    # print(f"R2: {r2:.4f}")

    if only_r2:
        return model_dict["T"], r2

    # 绘制折线图
    plt.figure(figsize=(15, 6))
    plt.plot(df['Y_true'], label='True Value', color='#1f77b4', linewidth=2)
    plt.plot(df['Y_pred'], label='Predicted Value', color='#ff7f0e', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'LSTM_{model_dict["T"]} Prediction VS True in {pathStr} set')
    # 在图的左下角添加R²值
    plt.text(0.1, 0.05, f'$R²: {r2:.4f}$', transform=plt.gca().transAxes, fontsize=16, color='green')
    plt.legend()
    # 保存图片
    plt.savefig(f'../results/LSTM_{pathStr}_evaluate/{pathStr}_lstm_{model_dict["T"]}_prediction_vs_true.png')
    return model_dict["T"], r2


def get_lstm_r2_dict(set='val', updated=False):
    # 定义文件路径
    file_path = f'../results/LSTM_{set}_r2_dict.csv'

    # 如果不需要更新且文件已存在，则直接从CSV文件加载数据
    if not updated and os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)  # 假设第一列是索引
        r2_dict = df['R2'].to_dict()
        print(f'Loaded LSTM {set} r2 dict from {file_path}')
        return r2_dict
    print(f'Calculating LSTM {set} r2 dict...')
    # 如果需要更新或文件不存在，则重新计算
    r2_dict = {}
    for T in tqdm(T_values, desc=f'lstm_evaluate in {set} set'):
        _, r2 = lstm_evaluate(best_lstm_model(T), set, only_r2=True)
        r2_dict[int(T)] = r2  # 确保键是Python的int类型

    # 将字典转换为DataFrame，并保存到CSV文件
    df = pd.DataFrame(list(r2_dict.items()), columns=['Temperature', 'R2'])
    df.to_csv(file_path, index=False)

    return r2_dict


symbol = 'SHSE.510300'

date_config = {
    'train_start_date': '2013-07-01',
    'train_end_date': '2020-07-01',
    'val_start_date': '2020-07-09',
    'val_end_date': '2022-01-24',
    'test_start_date': '2022-11-02',
    'test_end_date': '2023-09-13'
}

train_start_date = date_config['train_start_date']
train_end_date = date_config['train_end_date']
val_start_date = date_config['val_start_date']
val_end_date = date_config['val_end_date']
test_start_date = date_config['test_start_date']
test_end_date = date_config['test_end_date']

param_config = {
    'T_values': list(np.arange(1, 31)),
    'window_sizes': [20, 40, 60, 120],
    'hidden_dims': [30, 50, 70],
    'num_layers_list': [3, 5, 9]
}

T_values = param_config['T_values']
window_sizes = param_config['window_sizes']
hidden_dims = param_config['hidden_dims']
num_layers_list = param_config['num_layers_list']

config_id = 6

# 将config追加保存到configs.xlsx文件中
configs = {
    'symbol': symbol,
    'date_config': str(date_config),
    'param_config': str(param_config),
    'config_id': config_id,
}

configs_df = pd.DataFrame(configs, index=[0])

excel_path = '../results/configs.xlsx'

# if not os.path.exists(excel_path):
#     configs_df.to_excel(excel_path, index=False)
# else:
#     # 读取现有的 Excel 文件
#     existing_df = pd.read_excel(excel_path)
#     # 将新数据追加到现有数据之后
#     updated_df = pd.concat([existing_df, configs_df], ignore_index=True)
#     # 将更新后的 DataFrame 保存回 Excel 文件
#     updated_df.to_excel(excel_path, index=False)

# 定义模型保存目录
model_save_dir = 'lstm_models_' + str(config_id)

results_path = f'../results/best_models_results/best_models_results_{config_id}.xlsx'

if __name__ == "__main__":
    main()
