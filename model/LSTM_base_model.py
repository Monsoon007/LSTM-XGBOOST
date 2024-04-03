# import itertools
import os
import subprocess

import pandas as pd
from tqdm.contrib import itertools

from model.LSTM_data_handle import data_for_lstm
from data.get_data import get_common_data
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
    data = get_common_data(symbol, start_date, end_date, T)
    X, Y = data_for_lstm(data, window_size)
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
    symbol = 'SHSE.510300'
    train_start_date = '2008-01-01'
    train_end_date = '2020-01-01'
    val_start_date = '2020-01-01'
    val_end_date = '2021-01-01'
    test_start_date = '2021-01-01'
    test_end_date = '2022-01-01'

    output_dim = 1
    epochs = 100
    batch_size = 16
    lr = 0.01  # 学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义超参数搜索空间
    T_values = [1, 3, 7, 15, 30]
    window_sizes = [20, 40, 60, 120]
    hidden_dims = [30, 50, 70]
    num_layers_list = [3, 5, 9]

    # 用于存储所有最佳模型结果的列表
    best_models_results = []

    # 定义模型保存目录
    model_save_dir = 'lstm_models'

    # 检查模型保存目录是否存在，如果不存在，则创建它
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for T, window_size, hidden_dim, num_layers in itertools.product(T_values, window_sizes, hidden_dims,
                                                                    num_layers_list):
        # 检查该种参数模型是否已经训练过，如果训练过，则跳过
        model_path = os.path.join(model_save_dir,
                                  f'best_lstm_model_T{T}_window{window_size}_hidden{hidden_dim}_layers{num_layers}.pth')
        if os.path.exists(model_path):
            # 将该模型的结果记录到best_models_results中
            best_models_results.append({
                'T': T,
                'window_size': window_size,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'best_val_loss': 0.0,
                'best_model_path': model_path
            })
            continue

        # 创建tensorboard writer
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = f'runs/lstm_experiment_{current_time}_T{T}_window{window_size}_hidden{hidden_dim}_layers{num_layers}'
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
        best_model_path = ""
        for epoch in range(epochs):
            train(model, device, train_loader, criterion, optimizer, epoch, writer)
            val_loss = validate(model, device, val_loader, criterion, writer, epoch)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存当前最佳模型
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
        # 打印best_val_loss
        print(
            f'Best Validation Loss for T {T}, window_size {window_size}, hidden_dim {hidden_dim}, num_layers {num_layers}: {best_val_loss}')
        test(model, device, test_loader, criterion)

        writer.close()

    # 所有训练完成后，将最佳模型的结果保存到Excel文件中
    results_df = pd.DataFrame(best_models_results)
    results_df.to_excel('../result/best_models_results.xlsx', index=False)


if __name__ == "__main__":
    main()
