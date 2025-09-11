import pandas as pd
import numpy as np
from scipy import stats

# Significance marker function
def get_significance(p, alpha=0.05):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < alpha:
        return '*'
    else:
        return ''

# List of distributions (can be adjusted as needed)
distributions = ['lognorm', 'expon', 'gamma', 'beta', 'weibull_min']

# KS test function to test a single column against multiple distributions
def ks_test_all_distributions(data, var_name, sheet_name):
    results = []
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            stat, p = stats.kstest(data, dist_name, args=params)
            results.append({
                'Sheet Name': sheet_name,
                'Variable Name': var_name,
                'Distribution Tested': dist_name,
                'KS Statistic': round(stat, 4),
                'P-value': round(p, 4),
                'Significance': get_significance(p)
            })
        except Exception:
            results.append({
                'Sheet Name': sheet_name,
                'Variable Name': var_name,
                'Distribution Tested': dist_name,
                'KS Statistic': None,
                'P-value': None,
                'Significance': ''
            })
    return results

# === Main program ===
input_file = r"C:\new XLSX worksheet (2).xlsx"
output_file = r"C:\Multi-Distribution_Test_Results.xlsx"

xlsx = pd.ExcelFile(input_file)
all_results = []

for sheet_name in xlsx.sheet_names:
    df = xlsx.parse(sheet_name)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) >= 3 and data.std(ddof=0) > 0:
            res = ks_test_all_distributions(data, col, sheet_name)
            all_results.extend(res)

# Save all results
final_df = pd.DataFrame(all_results)
final_df.to_excel(output_file, index=False)
print(f"Multi-distribution K-S test completed, results saved to: {output_file}")