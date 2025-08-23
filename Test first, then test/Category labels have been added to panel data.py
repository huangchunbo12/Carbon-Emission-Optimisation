import pandas as pd

# 读取聚类结果文件（聚类_with_cluster.xlsx）
cluster_data_path = r"C:\Users\聚类_with_cluster.xlsx"
cluster_df = pd.read_excel(cluster_data_path)

# 读取面板数据文件（面板数据.xlsx）
panel_data_path = r"C:\Users\面板数据.xlsx"
panel_df = pd.read_excel(panel_data_path)

# 合并聚类结果和面板数据
merged_df = pd.merge(panel_df, cluster_df[['省份', 'Cluster']], on='省份', how='left')

# 将带有Cluster列的数据保存回新的Excel文件
output_path = r"C:\Users\面板数据_with_cluster.xlsx"
merged_df.to_excel(output_path, index=False)

# 打印保存路径
print(f"Data with Cluster added has been saved to: {output_path}")
