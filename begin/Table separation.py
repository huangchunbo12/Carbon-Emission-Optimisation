import pandas as pd

# 读取聚类结果文件（聚类_with_cluster.xlsx）
cluster_data_path = r"C:\Users\聚类_with_cluster.xlsx"
cluster_df = pd.read_excel(cluster_data_path)

# 读取面板数据文件（面板数据.xlsx）
panel_data_path = r"C:\Users\面板数据.xlsx"
panel_df = pd.read_excel(panel_data_path)

# 合并聚类结果和面板数据
merged_df = pd.merge(panel_df, cluster_df[['省份', 'Cluster']], on='省份', how='left')

# 根据Cluster列进行分组，并保存为不同的Excel文件
for cluster in merged_df['Cluster'].unique():
    cluster_data = merged_df[merged_df['Cluster'] == cluster]

    # 定义保存文件的路径
    output_path = f"C:/Users/Cluster_{cluster}_data.xlsx"

    # 保存到Excel文件
    cluster_data.to_excel(output_path, index=False)

    print(f"Cluster {cluster} data saved to {output_path}")
