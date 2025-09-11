import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'

# File paths
file_paths = [
    r"C:\Users\Cluster_0_data.xlsx",
    r"C:\Users\Cluster_1_data.xlsx",
    r"C:\Users\Cluster_2_data.xlsx",
    r"C:\Users\Cluster_3_data.xlsx"
]

# Cluster names for display
cluster_names = {
    0: 'Cluster I',
    1: 'Cluster II',
    2: 'Cluster III',
    3: 'Cluster IV'
}

# Differencing order for each cluster (based on ADF test conclusions)
diff_order = {
    0: 0,  # Cluster I: Original series is stationary
    1: 0,  # Cluster II: Original series is stationary
    2: 2,  # Cluster III: Stationary after 2nd-order differencing
    3: 1   # Cluster IV: Stationary after 1st-order differencing
}

# Main loop
for i, file_path in enumerate(file_paths):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    time_series = df.iloc[:, 1].dropna()

    # Perform differencing as needed
    d = diff_order[i]
    series_for_adf = time_series.copy()
    for _ in range(d):
        series_for_adf = series_for_adf.diff().dropna()

    # ADF test: No trend (regression='n')
    adf_result = adfuller(series_for_adf, autolag='AIC', regression='n')

    # Print results
    print(f"\n{cluster_names[i]} ADF Test Results (difference order = {d}):")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print(f"Critical Values: {adf_result[4]}")

    # Plotting
    max_lags = min(40, len(series_for_adf) // 2)
    title_suffix = f" (difference order = {d})"

    plt.figure(figsize=(7, 3))
    plt.subplot(121)
    plot_acf(series_for_adf, lags=max_lags, ax=plt.gca(), title=f'{cluster_names[i]} ACF{title_suffix}')
    plt.subplot(122)
    plot_pacf(series_for_adf, lags=max_lags, ax=plt.gca(), title=f'{cluster_names[i]} PACF{title_suffix}')
    plt.tight_layout()
    plt.show()