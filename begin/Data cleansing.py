import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

# 1. Import data from an Excel file
file_path = r'C:\Users\panel_data.xlsx'  # Please replace with your actual file path
data = pd.read_excel(file_path)

# 2. Check and preprocess the data
print(data.head())  # View the first few rows to ensure data is imported correctly

# Check for missing values
print(data.isnull().sum())  # Check the number of missing values in each column

# Select columns to impute (excluding those like 'Year' and 'Province')
columns_to_impute = [
    'G', 'P', 'SI', 'Energy Consumption', 'EI', 'TI', 'PCI', 'TDN', 'DMSP', 'TM', 'ES', 'UR', 'UPR', 'CO2 Emissions'
]


# 3. Machine learning imputation: Use KNN to fill missing values
# KNNImputer will use a K-nearest neighbors approach to fill in missing values
imputer = KNNImputer(n_neighbors=5)  # You can adjust n_neighbors to change the K value
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

# 4. View the imputed data
print(data.head())  # Check the data after imputation

# 5. Export the imputed data to a new Excel file
output_file_path = 'path_to_output_data.xlsx'  # Please replace with your actual output file path
data.to_excel(output_file_path, index=False)

# 6. Check if there are any remaining missing values
print(data.isnull().sum())  # Ensure there are no missing values left