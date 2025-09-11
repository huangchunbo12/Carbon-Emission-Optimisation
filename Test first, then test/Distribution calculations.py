import pandas as pd
import numpy as np

# Read the original Excel file
file_path = r'C:\Users\Cluster_3_data.xlsx'  # Please replace with your local file path
df = pd.read_excel(file_path)

# Calculate the growth rate for each province
df['G_growth'] = df.groupby('Province')['G'].pct_change() * 100
df['P_growth'] = df.groupby('Province')['P'].pct_change() * 100
df['ES_growth'] = df.groupby('Province')['ES'].pct_change() * 100
df['EI_growth'] = df.groupby('Province')['EI'].pct_change() * 100
df['TDN_growth'] = df.groupby('Province')['TDN'].pct_change() * 100
df['DMSP_growth'] = df.groupby('Province')['DMSP'].pct_change() * 100
df['SI_growth'] = df.groupby('Province')['SI'].pct_change() * 100

# Drop the original data columns (keep only the growth rates)
growth_rate_df = df[['Province', 'G_growth', 'P_growth',
                     'ES_growth', 'EI_growth',
                     'TDN_growth', 'DMSP_growth', 'SI_growth']]

# Drop rows with NaN values
growth_rate_df = growth_rate_df.dropna()

# Save the growth rate data to a new Excel file
output_file = 'Cluster_3_growth_rates.xlsx'  # Path to save the file
growth_rate_df.to_excel(output_file, index=False)

print(f"Data has been saved to {output_file}")