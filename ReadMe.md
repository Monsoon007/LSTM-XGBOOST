# 项目介绍

[论文](https://github.com/Monsoon007/LSTM-XGBOOST/blob/master/%E8%AE%BA%E6%96%87.pdf)

量化交易是一种利用数学模型和计算机算法进行交易的方法，已逐渐成为金融市场中的主流交易方式。随着机器学习和深度学习技术的发展，量化交易的策略开发和优化变得更加高效和精确。

本文提出了一种基于多频*FLIXNet*（LSTM-XGBoost集成网络）的量化交易择时策略，旨在捕捉不同频率的市场波动特征，提高预测准确性。本文用不同的未来$$T$$日平均日收益率训练了一系列*LSTM*模型，以捕捉市场的多频波动特征，然后将这些模型的输出作为*XGBoost*的输入，预测未来$$T_X$$日的平均日收益率。

基于*FLIXNet*模型，本文设计了一种量化交易策略：当*FLIXNet*预测的未来$$T_X$$日平均日收益率高于阈值$$R_{upper}$$时，执行全仓买入操作；当预测的平均日收益率低于阈值$$R_{lower}$$时，执行全仓卖出操作；在其他情况下，维持现有仓位。此外，当累计收益率超过$$200\%$$时，实行止盈清仓；当累计收益率低于$$-10\%$$时，实行止损清仓。

实验结果显示，*FLIXNet*在验证集上的表现优于传统多频*LSTM*模型，具有更高的预测准确性和稳定性。在测试集上的仿真交易以及额外的健壮性测试中，*FLIXNet*模型均取得了显著的经济效益，证明了该模型在量化交易中的实际应用价值。

# 项目结构

- data : 数据集存放与数据接口文件
    - get_data.py : 数据接口文件，主要功能为提供start_date到end_date的投资标的相关数据
    - macro.xlsx : 宏观经济数据, 从第三方平台下载
    - 数据工程.ipynb： 数据接口文件的前期探索性设计
    - 数据获取.ipynb： 部分通过第三方接口调用下载的数据
- model : 模型文件
    - LSTM_base_model : 基础LSTM模型
    - LSTM_data_handle : 将get_data.py提供的数据处理成LSTM模型的输入
    - my_xgboost.py：接受一系列不同T值的LSTM_base_model的输出，训练XGBoost模型
- LSTM Strategy：
    - main.py: 基于LSTM_base_model构建的量化交易策略
- FLIXNet Strategy：
  - main.py: 基于my_xgboost.py构建的量化交易策略

# 项目运行注意事项

- 获取数据和策略运行依赖[掘金量化平台](https://www.myquant.cn/)，需要注册账号、下载客户端，获取SDK和Token。




