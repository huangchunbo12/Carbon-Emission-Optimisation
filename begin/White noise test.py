import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

# Read Excel data
file_path = r"C:\Users\ARIMA_residuals.xlsx"
data = pd.read_excel(file_path)

# Ljung-Box Q-test
results = {}
for column in data.columns:
    # Perform Ljung-Box Q-test, with lags set to a maximum of 10 or (n-1)
    lb_test = acorr_ljungbox(data[column], lags=min(10, len(data)-1), return_df=True)
    results[column] = lb_test

# Organize results into a single DataFrame
output_df = pd.DataFrame()

for column, result in results.items():
    output_df[f'{column}_Q_statistic'] = result['lb_stat']
    output_df[f'{column}_p_value'] = result['lb_pvalue']

# Save results to an Excel file
output_file = r"C:\Users\15549\Desktop\Ljung_Box_Test_Results.xlsx"
output_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")