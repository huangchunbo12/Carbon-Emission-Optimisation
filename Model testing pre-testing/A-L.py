import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# 加载数据
file_path = r'Cluster_0_data.xlsx'  # 请确保数据文件路径正确
data = pd.read_excel(file_path)

# 提取需要的变量
X_columns = ['G', 'SI', 'ES', 'EI', 'DMSP']
y_column = 'CO2 Emissions'

# 划分数据集：2000-2015年作为训练集，2016-2020年作为测试集，2021-2022年作为验证集
train_data = data[(data['年份'] >= 2000) & (data['年份'] <= 2015)]
test_data = data[(data['年份'] >= 2016) & (data['年份'] <= 2020)]
val_data = data[(data['年份'] >= 2021) & (data['年份'] <= 2022)]

# 特征和目标
X_train = train_data[X_columns]
y_train = train_data[y_column]
X_test = test_data[X_columns]
y_test = test_data[y_column]
X_val = val_data[X_columns]
y_val = val_data[y_column]

# 数据标准化（MinMaxScaler）
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_val_scaled = scaler_X.transform(X_val)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))

# 转换为 PyTorch 张量
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_scaled = torch.tensor(y_test_scaled, dtype=torch.float32)
X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_scaled = torch.tensor(y_val_scaled, dtype=torch.float32)


# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])  # 仅取最后一个时间步的输出
        return predictions


# 初始化LSTM模型
input_size = X_train_scaled.shape[1]  # 特征数量
hidden_layer_size = 50
output_size = 1  # 预测CO2 Emissions
model = LSTMModel(input_size, hidden_layer_size, output_size)

# 定义损失函数和优化器
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# 训练模型
epochs = 600
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    y_pred = model(X_train_scaled.unsqueeze(1))  # LSTM 需要三维输入：batch_size, sequence_length, input_size

    # 计算损失
    loss = loss_function(y_pred, y_train_scaled)
    loss.backward()

    # 更新参数
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 模型评估
model.eval()
train_predict = model(X_train_scaled.unsqueeze(1))
test_predict = model(X_test_scaled.unsqueeze(1))
val_predict = model(X_val_scaled.unsqueeze(1))

# 将预测结果转换回原始的值（反归一化）
train_predict = scaler_y.inverse_transform(train_predict.detach().numpy())
test_predict = scaler_y.inverse_transform(test_predict.detach().numpy())
val_predict = scaler_y.inverse_transform(val_predict.detach().numpy())
y_train_rescaled = scaler_y.inverse_transform(y_train_scaled.detach().numpy())
y_test_rescaled = scaler_y.inverse_transform(y_test_scaled.detach().numpy())
y_val_rescaled = scaler_y.inverse_transform(y_val_scaled.detach().numpy())

# 计算并打印均方误差
train_rmse = np.sqrt(mean_squared_error(y_train_rescaled, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_rescaled, test_predict))
val_rmse = np.sqrt(mean_squared_error(y_val_rescaled, val_predict))

# R²
train_r2 = r2_score(y_train_rescaled, train_predict)
test_r2 = r2_score(y_test_rescaled, test_predict)
val_r2 = r2_score(y_val_rescaled, val_predict)

# MAPE
train_mape = np.mean(np.abs((y_train_rescaled - train_predict) / y_train_rescaled)) * 100
test_mape = np.mean(np.abs((y_test_rescaled - test_predict) / y_test_rescaled)) * 100
val_mape = np.mean(np.abs((y_val_rescaled - val_predict) / y_val_rescaled)) * 100

# VAF (Variance Accounted For)
train_vaf = 1 - np.var(y_train_rescaled - train_predict) / np.var(y_train_rescaled)
test_vaf = 1 - np.var(y_test_rescaled - test_predict) / np.var(y_test_rescaled)
val_vaf = 1 - np.var(y_val_rescaled - val_predict) / np.var(y_val_rescaled)

# 输出结果
print(f"Train R²: {train_r2}, RMSE: {train_rmse}, MAPE: {train_mape}, VAF: {train_vaf}")
print(f"Test R²: {test_r2}, RMSE: {test_rmse}, MAPE: {test_mape}, VAF: {test_vaf}")
print(f"Validation R²: {val_r2}, RMSE: {val_rmse}, MAPE: {val_mape}, VAF: {val_vaf}")

# 保存LSTM模型权重
model_weights = {
    'weights': model.state_dict(),
}
with open('lstm_model_weights.pth', 'wb') as f:
    pickle.dump(model_weights, f)
