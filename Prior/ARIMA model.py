import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

# Set font
plt.rcParams['font.family'] = 'Times New Roman'

# File paths
file_paths = [
    r"C:\Users\Cluster_0_data.xlsx",
    r"C:\Users\Cluster_1_data.xlsx",
    r"C:\Users\Cluster_2_data.xlsx",
    r"C:\Users\Cluster_3_data.xlsx"
]

# Name mapping
cluster_names = {
    0: 'Cluster I',
    1: 'Cluster II',
    2: 'Cluster III',
    3: 'Cluster IV'
}

# Model settings
model_settings = {
    0: {'type': 'ar', 'lags': 2, 'trend': 'c'},
    1: {'type': 'ar', 'lags': 2, 'trend': 'c'},
    2: {'type': 'arima', 'order': (1, 1, 1)},  # ARIMA(1,0,1)
    3: {'type': 'ar', 'lags': 2, 'trend': 'ct'}
}

# Main loop
for i, file_path in enumerate(file_paths):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    time_series = df.iloc[:, 1]

    settings = model_settings[i]
    name = cluster_names[i]

    print(f"\n{name} model fitting:")

    if settings['type'] == 'ar':
        model = AutoReg(time_series, lags=settings['lags'], trend=settings['trend']).fit()
        print(model.summary())

        # Calculate R²
        residuals = model.resid
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((time_series - np.mean(time_series)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R² for {name}: {r_squared:.4f}")

    elif settings['type'] == 'arima':
        model = ARIMA(time_series, order=settings['order']).fit()
        print(model.summary())

        # MA models do not have an explicit R², so calculate an approximation manually
        fitted = model.fittedvalues
        residuals = time_series[1:] - fitted[1:]  # The first value is excluded by differencing
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((time_series[1:] - np.mean(time_series[1:])) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"Approx. R² for {name} (ARIMA): {r_squared:.4f}")

    # Save residuals
    residuals_df = pd.DataFrame({'Residuals': model.resid})
    residuals_df.to_csv(f"Cluster_{i}_residuals.csv", index=False)
    print(f"Residuals for {name} saved as 'Cluster_{i}_residuals.csv'")