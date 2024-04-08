import numpy as np
from sklearn.preprocessing import StandardScaler


def pack_data_with_scaling(data, input_columns, output_column, window_size):
    """
    将DataFrame中的数据打包成LSTM的输入和输出，并对每个窗口的输入数据进行标准化处理。
    """
    X = []
    y = []
    scaler = StandardScaler()

    for i in range(len(data) - window_size):
        window_data = data[input_columns].iloc[i:i + window_size]  # python的区间是左闭右开
        scaled_window_data = scaler.fit_transform(window_data)
        X.append(scaled_window_data)  # 只对输入数据进行标准化处理
        y.append(data[output_column].iloc[i + window_size])

    return np.array(X), np.array(y)


def data_for_lstm(data, window_size=20):
    # 选择用于预测的特征列
    # input_columns = ['close', 'volume', 'log_return', 'MA_5', 'MA_10', 'MA_20', 'MA_60', 'MA_120',
    #                  'EMA_5', 'EMA_10', 'EMA_20', 'EMA_60', 'EMA_120',
    #                  'RSI', 'ATR', 'Bollinger_Width', 'VWAP', 'bdi', 'AR', 'BR']
    input_columns = data.columns.tolist()
    # print(input_columns)
    output_column = data.columns.tolist()[-1]  # 最后一列为预测目标
    # 打包数据
    X, y = pack_data_with_scaling(data, input_columns, output_column, window_size)
    # 说明：X的形状为(样本数量, 窗口大小, 特征数量)，y的形状为(样本数量,)。

    return X, y
