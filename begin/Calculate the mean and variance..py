import pandas as pd
import numpy as np

# Define input and output Excel file paths
input_file = r"C:\Users\new XLSX worksheet (2).xlsx"
output_file = r'C:\Users\variable_mean_std_results.xlsx'

# Read the Excel file (which may contain multiple sheets)
xlsx = pd.ExcelFile(input_file)
writer = pd.ExcelWriter(output_file, engine='openpyxl')

# Iterate through each sheet
for sheet_name in xlsx.sheet_names:
    df = xlsx.parse(sheet_name)

    # Select numerical columns (excluding non-numerical ones)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []  # Store the mean and standard deviation for each column

    # Calculate the mean and standard deviation for each column
    for col in numeric_cols:
        data = df[col].dropna()  # Drop missing values
        mean_val = round(data.mean(), 3)
        std_val = round(data.std(ddof=0), 3)  # Standard deviation (population)
        results.append([col, mean_val, std_val])

    # Create a result DataFrame with columns: 'Column Name', 'Mean', 'Standard Deviation'
    result_df = pd.DataFrame(results, columns=['Column Name', 'Mean', 'Standard Deviation'])

    # Write the results to the corresponding sheet
    result_df.to_excel(writer, sheet_name=sheet_name, index=False)

# Save the results
writer.close()
print("Variable means and standard deviations have been saved to", output_file)