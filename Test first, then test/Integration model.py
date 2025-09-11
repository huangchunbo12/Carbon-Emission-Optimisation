import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import random
import math
import matplotlib.pyplot as plt

# New imports
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
file_path = r'C:\Users\Cluster_0_data.xlsx'
data = pd.read_excel(file_path)
# Get the data file name (without extension) for subsequent file saving
cluster_name = os.path.splitext(os.path.basename(file_path))[0]
X_columns = ['G', 'SI', 'ES', 'EI', 'DMSP']
y_column = 'CO2 Emissions'

# Group by year
years = data['Year'].unique()

# Randomly select 7 years for the test set
random.seed(43)  # Set a random seed for reproducibility
test_years = random.sample(list(years), 7)

# Use the remaining years as the training set
train_years = [year for year in years if year not in test_years]

# Split the dataset
train_data = data[data['Year'].isin(train_years)]
test_data = data[data['Year'].isin(test_years)]

X_train = train_data[X_columns]
y_train = train_data[y_column]
X_test = test_data[X_columns]
y_test = test_data[y_column]

# Normalize features (for the LSTM part)
scaler_X = MinMaxScaler()
X_train_scaled_np = scaler_X.fit_transform(X_train)
X_test_scaled_np = scaler_X.transform(X_test)

# Normalize target variable as well (for the LSTM part)
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)
scaler_y = MinMaxScaler()
y_train_scaled_np = scaler_y.fit_transform(y_train_np)
y_test_scaled_np = scaler_y.transform(y_test_np)

# Convert X to tensors (LSTM requires 3D input: batch_size, seq_len, feature_dim)
X_train_scaled = torch.tensor(X_train_scaled_np, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled_np, dtype=torch.float32)

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------
# LSTM Regression Model Definition
# ---------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ---------------------------
# Evaluation Metrics Function: R2, VAF, RMSLE
# ---------------------------
def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    vaf = 1 - np.var(y_true - y_pred) / np.var(y_true)
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.maximum(0, y_pred))
    rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    return r2, vaf, rmsle


# ---------------------------
# Main Program: LSTM and AdaBoost Model Training, Prediction, and Stacking
# ---------------------------
if __name__ == '__main__':
    ##############################
    # Part 1: LSTM Model Training and Prediction
    ##############################
    X_train_seq = X_train_scaled.unsqueeze(1)
    X_test_seq = X_test_scaled.unsqueeze(1)

    y_train_tensor = torch.tensor(y_train_scaled_np, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_scaled_np, dtype=torch.float32).to(device)

    input_size = X_train_scaled.shape[1]
    hidden_size = 64
    num_layers = 1
    output_size = 1

    model = LSTMRegressor(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    num_epochs = 600
    convergence_curve = []

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_seq.to(device))
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        convergence_curve.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

    # LSTM model prediction
    model.eval()
    with torch.no_grad():
        train_pred_lstm = model(X_train_seq.to(device)).cpu().numpy()
        test_pred_lstm = model(X_test_seq.to(device)).cpu().numpy()

    train_pred_lstm_inv = scaler_y.inverse_transform(train_pred_lstm).ravel()
    test_pred_lstm_inv = scaler_y.inverse_transform(test_pred_lstm).ravel()

    ##############################
    # Part 2: AdaBoost Model Training and Prediction
    ##############################
    ada_model = AdaBoostRegressor()
    ada_model.fit(X_train_scaled_np, y_train_np.ravel())

    train_pred_ada = ada_model.predict(X_train_scaled_np)
    test_pred_ada = ada_model.predict(X_test_scaled_np)

    ##############################
    # Part 3: Stacking Ensemble Learning
    ##############################
    train_stack_X = np.column_stack((train_pred_lstm_inv, train_pred_ada))
    test_stack_X = np.column_stack((test_pred_lstm_inv, test_pred_ada))

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    meta_train_lr = np.zeros(train_stack_X.shape[0])
    meta_test_preds_lr = []

    for train_index, valid_index in kf.split(train_stack_X):
        X_meta_train = train_stack_X[train_index]
        y_meta_train = y_train_np.ravel()[train_index]

        lr_model = LinearRegression()
        lr_model.fit(X_meta_train, y_meta_train)
        meta_train_lr[valid_index] = lr_model.predict(train_stack_X[valid_index])
        meta_test_preds_lr.append(lr_model.predict(test_stack_X))

    test_pred_stack_lr = np.mean(meta_test_preds_lr, axis=0)

    ##############################
    # Evaluate All Models
    ##############################
    r2_train, vaf_train, rmsle_train = compute_metrics(y_train_np.ravel(), train_pred_lstm_inv)
    r2_test, vaf_test, rmsle_test = compute_metrics(y_test_np.ravel(), test_pred_lstm_inv)

    r2_train_ada, vaf_train_ada, rmsle_train_ada = compute_metrics(y_train_np.ravel(), train_pred_ada)
    r2_test_ada, vaf_test_ada, rmsle_test_ada = compute_metrics(y_test_np.ravel(), test_pred_ada)

    r2_test_stack_lr, vaf_test_stack_lr, rmsle_test_stack_lr = compute_metrics(y_test_np.ravel(), test_pred_stack_lr)

    print("\nLSTM Model Evaluation Metrics:")
    print(f"Training Set - R2: {r2_train:.4f}, VAF: {vaf_train:.4f}, RMSLE: {rmsle_train:.4f}")
    print(f"Test Set - R2: {r2_test:.4f}, VAF: {vaf_test:.4f}, RMSLE: {rmsle_test:.4f}")

    print("\nAdaBoost Model Evaluation Metrics:")
    print(f"Training Set - R2: {r2_train_ada:.4f}, VAF: {vaf_train_ada:.4f}, RMSLE: {rmsle_train_ada:.4f}")
    print(f"Test Set - R2: {r2_test_ada:.4f}, VAF: {vaf_test_ada:.4f}, RMSLE: {rmsle_test_ada:.4f}")

    print("\nStacking (Linear Regression Meta-Learner) Evaluation Metrics:")
    print(f"Test Set - R2: {r2_test_stack_lr:.4f}, VAF: {vaf_test_stack_lr:.4f}, RMSLE: {rmsle_test_stack_lr:.4f}")

    ##############################
    # Save Results
    ##############################
    test_results_df = pd.DataFrame({
        'True Values': y_test_np.ravel(),
        'LSTM Predicted': test_pred_lstm_inv,
        'AdaBoost Predicted': test_pred_ada,
        'Stacking LR Predicted': test_pred_stack_lr
    })

    eval_metrics_df = pd.DataFrame({
        'Model': ['LSTM', 'AdaBoost', 'Stacking LR'],
        'Test R2': [r2_test, r2_test_ada, r2_test_stack_lr],
        'Test VAF': [vaf_test, vaf_test_ada, vaf_test_stack_lr],
        'Test RMSLE': [rmsle_test, rmsle_test_ada, rmsle_test_stack_lr]
    })

    # Save to Excel file
    excel_file_name = f"{cluster_name}_Stacking_Results.xlsx"
    with pd.ExcelWriter(excel_file_name) as writer:
        test_results_df.to_excel(writer, sheet_name="Test Results", index=False)
        eval_metrics_df.to_excel(writer, sheet_name="Evaluation Metrics", index=False)

    print(f"\nAll tasks completed. Results saved to '{excel_file_name}'.")