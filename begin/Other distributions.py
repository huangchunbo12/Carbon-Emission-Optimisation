import pandas as pd
import numpy as np
from scipy import stats

# === 1. Load data ===
input_file = 'non_normal_data.xlsx'  # Replace with your Excel file name
df = pd.read_excel(input_file)

# Select the first two columns and drop missing values
col1 = df.iloc[:, 0].dropna()
col2 = df.iloc[:, 1].dropna()

# === 2. Define distributions to fit ===
distributions = ['lognorm', 'expon', 'gamma', 'beta', 'weibull_min']

# === 3. Define the testing function ===
def ks_test_all_distributions(data, var_name):
    results = []
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            stat, p = stats.kstest(data, dist_name, args=params)
            results.append({
                'Variable Name': var_name,
                'Distribution Tested': dist_name,
                'KS Statistic': round(stat, 4),
                'P-value': round(p, 4)
            })
        except Exception as e:
            results.append({
                'Variable Name': var_name,
                'Distribution Tested': dist_name,
                'KS Statistic': None,
                'P-value': None
            })
    return results

# === 4. Test the first and second columns separately ===
results_col1 = ks_test_all_distributions(col1, df.columns[0])
results_col2 = ks_test_all_distributions(col2, df.columns[1])

# === 5. Combine results and save ===
final_results = pd.DataFrame(results_col1 + results_col2)
output_file = 'distribution_test_results.xlsx'
final_results.to_excel(output_file, index=False)
print(f"Testing complete. Results saved to {output_file}")