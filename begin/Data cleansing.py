import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

# 1. 从Excel文件导入数据
file_path = r'C:\Users\面板数据.xlsx'  # 请替换为实际的文件路径
data = pd.read_excel(file_path)

# 2. 检查数据并进行预处理
print(data.head())  # 查看前几行数据，确保数据被正确导入

# 检查数据缺失情况
print(data.isnull().sum())  # 查看每列的缺失值数量

# 选择需要填补的列（除了年份和省份这些不会被插值的列）
columns_to_impute = [
    'G', 'P', 'SI', '能源消费总量', 'EI', 'TI', 'PCI', 'TDN', 'DMSP', 'TM', 'ES', 'UR', 'UPR', 'CO2 Emissions'
]


# 3. 机器学习插值：使用KNN填补缺失值
# KNNImputer将使用K最近邻的方式填补缺失值
imputer = KNNImputer(n_neighbors=5)  # 可以调整n_neighbors来改变K值
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

# 4. 查看填补后的数据
print(data.head())  # 检查填补后的数据

# 5. 将填补后的数据导出到新的Excel文件
output_file_path = 'path_to_output_data.xlsx'  # 请替换为实际的输出文件路径
data.to_excel(output_file_path, index=False)

# 6. 检查是否还有缺失值
print(data.isnull().sum())  # 确保没有缺失值
