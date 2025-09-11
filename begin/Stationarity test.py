import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# File paths
file_paths = [
    r"C:\Users\Cluster_0_data.xlsx",
    r"C:\Users\Cluster_1_data.xlsx",
    r"C:\Users\Cluster_2_data.xlsx",
    r"C:\Users\Cluster_3_data.xlsx"
]

# List to store results
results = []

# Define ADF test function
def adf_test(series, trend='c'):
    result = adfuller(series, autolag='AIC', regression=trend)
    return result[0], result[1], result[2]  # t-statistic, p-value, lag

# Main loop
for file_path in file_paths:
    # Read data
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    time_series = df.iloc[:, 1].dropna()

    # Test the original series
    t_n, p_n, _ = adf_test(time_series, trend='n')  # No trend
    t_c, p_c, _ = adf_test(time_series, trend='c')  # Constant
    t_ct, p_ct, _ = adf_test(time_series, trend='ct')  # Constant + trend

    # First-order differencing
    diff_series = time_series.diff().dropna()
    dt_n, dp_n, _ = adf_test(diff_series, trend='n')
    dt_c, dp_c, _ = adf_test(diff_series, trend='c')
    dt_ct, dp_ct, _ = adf_test(diff_series, trend='ct')

    results.append({
        'File': file_path,
        't (Original: No Trend)': t_n, 'p (Original: No Trend)': p_n,
        't (Original: Constant)': t_c, 'p (Original: Constant)': p_c,
        't (Original: Constant+Trend)': t_ct, 'p (Original: Constant+Trend)': p_ct,
        't (1st Diff: No Trend)': dt_n, 'p (1st Diff: No Trend)': dp_n,
        't (1st Diff: Constant)': dt_c, 'p (1st Diff: Constant)': dp_c,
        't (1st Diff: Constant+Trend)': dt_ct, 'p (1st Diff: Constant+Trend)': dp_ct,
    })

# Write to Excel
results_df = pd.DataFrame(results)
output_path = r"C:\Users\ADF_Test_Results.xlsx"
results_df.to_excel(output_path, index=False)

print(f"ADF test results have been saved to {output_path}")