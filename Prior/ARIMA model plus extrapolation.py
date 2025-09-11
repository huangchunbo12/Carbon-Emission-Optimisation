import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'Times New Roman'

# List of file paths
file_paths = [
    r"C:\Users\Cluster_0_data.xlsx",
    r"C:\Users\Cluster_1_data.xlsx",
    r"C:\Users\Cluster_2_data.xlsx",
    r"C:\Users\Cluster_3_data.xlsx"
]

# Model parameters (extracted from the figures)
model_params = {
    "Cluster_0_data": {"const": 1.570e7, "trend": 0.0,    "phi1": 1.429, "phi2": -0.502, "diff": False},
    "Cluster_1_data": {"const": 1.631e7, "trend": 0.0,    "phi1": 0.929, "phi2": 0.0,    "diff": False},
    "Cluster_2_data": {"const": 0.0,     "trend": 0.0,    "phi1": 0.951, "phi2": 0.0,    "diff": True, "ma1": -0.852},
    "Cluster_3_data": {"const": 4.960e7, "trend": 6.560e6, "phi1": 1.096, "phi2": -0.367, "diff": False}
}

forecast_steps = 28

for file_path in file_paths:
    cluster_name = os.path.basename(file_path).replace(".xlsx", "")
    print(f"Processing: {cluster_name}")

    # Parameter extraction
    p = model_params[cluster_name]
    const, trend, phi1, phi2 = p["const"], p["trend"], p["phi1"], p["phi2"]
    use_diff = p.get("diff", False)
    ma1 = p.get("ma1", 0.0)

    # Data reading
    df = pd.read_excel(file_path, sheet_name='Sheet3').dropna()
    years = df.iloc[:, 0].tolist()
    series = df.iloc[:, 1].tolist()

    forecast = []
    t_start = int(years[-1]) + 1
    forecast_years = list(range(t_start, t_start + forecast_steps))

    if not use_diff:
        # Classic AR model extrapolation
        y_tm1, y_tm2 = series[-1], series[-2]
        for i in range(forecast_steps):
            t = len(series) + i
            y_t = const + trend * t + phi1 * y_tm1 + phi2 * y_tm2
            forecast.append(y_t)
            y_tm2, y_tm1 = y_tm1, y_t
    else:
        # Differencing model ARIMA(1,1,1)
        last_actual = series[-1]
        delta_tm1 = series[-1] - series[-2]
        err_tm1 = 0
        for _ in range(forecast_steps):
            err_t = np.random.normal(0, 1)  # Can be changed to 0 for fixed results
            delta_t = phi1 * delta_tm1 + ma1 * err_tm1 + err_t
            y_t = last_actual + delta_t
            forecast.append(y_t)
            # Update
            last_actual = y_t
            delta_tm1 = delta_t
            err_tm1 = err_t

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(years, series, label='Actual', marker='o')
    plt.plot(forecast_years, forecast, label='Forecast', linestyle='--', marker='x')
    plt.title(f'{cluster_name} - Forecast Based on Given Model', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('CO₂ Emissions')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()