import pandas as pd
import numpy as np
from scipy.stats import shapiro


def get_significance_marker(p, alpha=0.05):
    """Returns significance markers based on the p-value: '***' for p<0.001, '**' for p<0.01, '*' for p<0.05, otherwise an empty string."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < alpha:
        return '*'
    else:
        return ''


# Define input and output Excel file paths
input_file = r"C:\Users\new XLSX worksheet (2).xlsx"
output_file = r'C:\Users\SW_test_results.xlsx'

# Read the Excel file (which may contain multiple sheets)
xlsx = pd.ExcelFile(input_file)
writer = pd.ExcelWriter(output_file, engine='openpyxl')

# Significance level
alpha = 0.05

# Iterate through each sheet
for sheet_name in xlsx.sheet_names:
    df = xlsx.parse(sheet_name)

    # Select numerical columns (excluding non-numerical ones like 'Year' or 'Province')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []  # Store the results for the current sheet

    # Perform Shapiro-Wilk test for each column
    for col in numeric_cols:
        data = df[col].dropna()  # Drop missing values

        # The test requires at least 3 samples and non-zero standard deviation
        if len(data) < 3 or data.std(ddof=0) == 0:
            sw_stat, sw_p = np.nan, np.nan
        else:
            sw_stat, sw_p = shapiro(data)

        # Format the statistic and p-value to 3 decimal places
        sw_stat_str = f"{sw_stat:.3f}" if not np.isnan(sw_stat) else ""
        sw_p_str = f"{sw_p:.3f}" if not np.isnan(sw_p) else ""

        # Add significance marker
        signif = get_significance_marker(sw_p, alpha) if not np.isnan(sw_p) else ""
        sw_p_marked = f"{sw_p_str}{signif}" if sw_p_str != "" else ""

        results.append([col, sw_stat_str, sw_p_marked])

    # Create a result DataFrame with columns: 'Column Name', 'SW Statistic', 'SW P-value & Significance'
    result_df = pd.DataFrame(results, columns=['Column Name', 'SW Statistic', 'SW P-value & Significance'])
    # Write the results to the corresponding sheet in the output Excel file
    result_df.to_excel(writer, sheet_name=sheet_name, index=False)

# Save the output Excel file
writer.close()
print("Shapiro-Wilk test results have been saved to", output_file)