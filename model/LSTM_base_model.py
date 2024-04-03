import subprocess

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
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,device):
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
def prepare_data(symbol, start_date, end_date,T, window_size):
    data = get_common_data(symbol, start_date, end_date,T)
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
    train_start_date = '2006-01-01'
    train_end_date = '2020-01-01'
    val_start_date = '2020-01-01'
    val_end_date = '2021-01-01'
    test_start_date = '2021-01-01'
    test_end_date = '2022-01-01'

    output_dim = 1
    epochs = 100
    batch_size = 16
    lr = 0.01 # 学习率


    # LSTM模型需要调试的参数
    T = 3
    window_size = 40
    hidden_dim = 50
    num_layers = 3


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 准备数据
    X_train, Y_train = prepare_data(symbol, train_start_date, train_end_date, T,window_size)
    X_val, Y_val = prepare_data(symbol, val_start_date, val_end_date,T, window_size)
    X_test, Y_test = prepare_data(symbol, test_start_date, test_end_date,T, window_size)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

    model = LSTMModel(X_train.shape[2], hidden_dim, num_layers, output_dim,device).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        train(model, device, train_loader, criterion, optimizer, epoch, writer)
        val_loss = validate(model, device, val_loader, criterion, writer, epoch)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    # 打印best_val_loss
    print(f'Best Validation Loss: {best_val_loss}')
    test(model, device, test_loader, criterion)

    # 等待输入确认，关闭SummaryWriter
    confirmation = input("Do you want to close the SummaryWriter? (yes/no): ")
    if confirmation.lower() == 'yes':
        writer.close()
        process.terminate()

if __name__ == "__main__":
    main()


