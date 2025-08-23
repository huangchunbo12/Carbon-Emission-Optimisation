import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


# Haversine函数计算两点之间的地理距离
def haversine(lon1, lat1, lon2, lat2):
    # 转换为弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # 地球半径（单位：km）
    R = 6371  # 地球半径（千米）
    distance = R * c
    return distance


# 加载数据
data_path = r"C:\Users\面板数据.xlsx"
df = pd.read_excel(data_path)

# 计算每个省份的GDP增长率和碳排放增长率
df['G Growth Rate'] = df.groupby('省份')['G'].pct_change() * 100  # GDP增长率
df['CO2 Growth Rate'] = df.groupby('省份')['CO2 Emissions'].pct_change() * 100  # 碳排放增长率

# 删除第一年无法计算增长率的记录
df = df.dropna(subset=['G Growth Rate', 'CO2 Growth Rate'])

# 计算过去5年的滚动平均增长率，作为时间因素
df['5-Year G Growth Rate'] = df.groupby('省份')['G Growth Rate'].rolling(window=5).mean().reset_index(level=0,
                                                                                                      drop=True)
df['5-Year CO2 Growth Rate'] = df.groupby('省份')['CO2 Growth Rate'].rolling(window=5).mean().reset_index(level=0,
                                                                                                          drop=True)

# 假设您有经纬度数据，去掉 "°" 符号后进行转换
df['Longitude Center'] = (df['最西经度'].apply(lambda x: float(x.replace('°', '').replace('E', ''))) + df[
    '最东经度'].apply(lambda x: float(x.replace('°', '').replace('E', '')))) / 2
df['Latitude Center'] = (df['最南纬度'].apply(lambda x: float(x.replace('°', '').replace('N', ''))) + df[
    '最北纬度'].apply(lambda x: float(x.replace('°', '').replace('N', '')))) / 2

# 打印查看结果
print(df[['省份', 'Longitude Center', 'Latitude Center']])

# 生成经纬度矩阵
coordinates = df[['Longitude Center', 'Latitude Center']].values

# 计算省份之间的地理距离（使用Haversine距离）
num_provinces = len(coordinates)
distance_matrix = np.zeros((num_provinces, num_provinces))

for i in range(num_provinces):
    for j in range(i + 1, num_provinces):
        distance = haversine(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1])
        distance_matrix[i, j] = distance_matrix[j, i] = distance

# 设定一个阈值来创建空间权重矩阵，邻近的省份赋予较大的权重
threshold = 500  # 设置一个合适的阈值，单位：公里
W = distance_matrix < threshold  # 根据距离生成邻接矩阵，距离较近的省份被认为是邻接省份

# 对空间权重矩阵W进行正则化，确保每行权重总和为1
row_sums = W.sum(axis=1)
W_normalized = W / row_sums[:, np.newaxis]  # 保证每行的总和为1

# 使用空间权重矩阵 W 对增长率进行加权调整
df['Adjusted G Growth Rate'] = df['G Growth Rate'] + np.dot(W_normalized, df['G Growth Rate'])
df['Adjusted CO2 Growth Rate'] = df['CO2 Growth Rate'] + np.dot(W_normalized, df['CO2 Growth Rate'])

# 合并时间和空间因素，计算最终的综合增长率
df['综合GDP增长因子'] = df['5-Year G Growth Rate'] + df['Adjusted G Growth Rate']
df['综合C02排放量因子'] = df['5-Year CO2 Growth Rate'] + df['Adjusted CO2 Growth Rate']

# 将结果保存到新Excel文件
output_path = r"C:\Users\面板数据_with_temporal_spatial_factors.xlsx"
df.to_excel(output_path, index=False)

# 查看最终的结果
print(df[['省份', '年份', '综合GDP增长因子', '综合C02排放量因子']])
