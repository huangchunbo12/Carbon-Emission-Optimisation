import pandas as pd

# 读取原始Excel文件
file_path = r'C:\Users\Cluster_3_data.xlsx' # 请替换为您本地文件的路径
df = pd.read_excel(file_path)

# 计算每个省份的增长率
df['G_growth'] = df.groupby('省份')['G'].pct_change() * 100
df['P_growth'] = df.groupby('省份')['P'].pct_change() * 100
df['ES_growth'] = df.groupby('省份')['ES'].pct_change() * 100
df['EI_growth'] = df.groupby('省份')['EI'].pct_change() * 100
df['TDN_growth'] = df.groupby('省份')['TDN'].pct_change() * 100
df['DMSP_growth'] = df.groupby('省份')['DMSP'].pct_change() * 100
df['SI_growth'] = df.groupby('省份')['SI'].pct_change() * 100

# 删除原始数据列（只保留增长率）
growth_rate_df = df[['省份', 'G_growth', 'P_growth',
                     'ES_growth', 'EI_growth',
                     'TDN_growth', 'DMSP_growth', 'SI_growth']]

# 删除包含NaN值的行
growth_rate_df = growth_rate_df.dropna()

# 将增长率数据保存为新的Excel文件
output_file = 'Cluster_3_growth_rates.xlsx'  # 保存文件的路径
growth_rate_df.to_excel(output_file, index=False)

print(f"数据已保存至 {output_file}")
