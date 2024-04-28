import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载加州房价数据集
california_housing = fetch_california_housing()
X, y = california_housing.data, california_housing.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为DMatrix数据格式，XGBoost专用
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 指定模型参数，包括使用GPU的tree_method参数
param = {
    'max_depth': 5,
    'eta': 0.3,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist'  # 指定使用GPU
}

# 训练模型
num_round = 100
bst = xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')])

# 使用训练好的模型进行预测
y_pred = bst.predict(dtest)

# 计算并打印RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Validation RMSE: {rmse}")
