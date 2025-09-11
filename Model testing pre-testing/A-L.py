import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# Load data
file_path = r'Cluster_0_data.xlsx'  # Ensure the data file path is correct
data = pd.read_excel(file_path)

# Extract the required variables
X_columns = ['G', 'SI', 'ES', 'EI', 'DMSP']
y_column = 'CO2 Emissions'

# Split the dataset: 2000-2015 for training, 2016-2020 for testing, 2021-2022 for validation
train_data = data[(data['Year'] >= 2000) & (data['Year'] <= 2015)]
test_data = data[(data['Year'] >= 2016) & (data['Year'] <= 2020)]
val_data = data[(data['Year'] >= 2021) & (data['Year'] <= 2022)]

# Features and target
X_train = train_data[X_columns]
y_train = train_data[y_column]
X_test = test_data[X_columns]
y_test = test_data[y_column]
X_val = val_data[X_columns]
y_val = val_data[y_column]

# Data normalization (MinMaxScaler)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_val_scaled = scaler_X.transform(X_val)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))

# Convert to PyTorch tensors
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_scaled = torch.tensor(y_test_scaled, dtype=torch.float32)
X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_scaled = torch.tensor(y_val_scaled, dtype=torch.float32)


# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])  # Only take the output of the last time step
        return predictions


# Initialize the LSTM model
input_size = X_train_scaled.shape[1]  # Number of features
hidden_layer_size = 50
output_size = 1  # Predicting CO2 Emissions
model = LSTMModel(input_size, hidden_layer_size, output_size)

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Train the model
epochs = 600
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train_scaled.unsqueeze(1))  # LSTM requires a 3D input: batch_size, sequence_length, input_size

    # Calculate loss
    loss = loss_function(y_pred, y_train_scaled)
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Model evaluation
model.eval()
train_predict = model(X_train_scaled.unsqueeze(1))
test_predict = model(X_test_scaled.unsqueeze(1))
val_predict = model(X_val_scaled.unsqueeze(1))

# Convert predictions back to original values (inverse transform)
train_predict = scaler_y.inverse_transform(train_predict.detach().numpy())
test_predict = scaler_y.inverse_transform(test_predict.detach().numpy())
val_predict = scaler_y.inverse_transform(val_predict.detach().numpy())
y_train_rescaled = scaler_y.inverse_transform(y_train_scaled.detach().numpy())
y_test_rescaled = scaler_y.inverse_transform(y_test_scaled.detach().numpy())
y_val_rescaled = scaler_y.inverse_transform(y_val_scaled.detach().numpy())

# Calculate and print evaluation metrics
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

# Output results
print(f"Train R²: {train_r2}, RMSE: {train_rmse}, MAPE: {train_mape}, VAF: {train_vaf}")
print(f"Test R²: {test_r2}, RMSE: {test_rmse}, MAPE: {test_mape}, VAF: {test_vaf}")
print(f"Validation R²: {val_r2}, RMSE: {val_rmse}, MAPE: {val_mape}, VAF: {val_vaf}")

# Save LSTM model weights
model_weights = {
    'weights': model.state_dict(),
}
with open('lstm_model_weights.pth', 'wb') as f:
    pickle.dump(model_weights, f)