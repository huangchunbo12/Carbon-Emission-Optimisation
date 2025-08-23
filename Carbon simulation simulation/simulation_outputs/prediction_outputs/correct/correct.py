import pandas as pd
import numpy as np

# 定义聚类场景和对应的 Excel 文件前缀
scenarios = ['BAU', 'HG', 'MS']
clusters = ['ClusterI', 'ClusterII', 'ClusterIII', 'ClusterIV']
province_mapping = {'云南': 'ClusterI', '上海': 'ClusterII', '山西': 'ClusterIII', '宁夏': 'ClusterIV'}

# 读取聚类中心 - ARIMA.xlsx 文件（年份作为索引）
try:
    arima_df = pd.read_excel('聚类中心-ARIMA.xlsx', index_col='年份')
except FileNotFoundError:
    print("未找到 '聚类中心-ARIMA.xlsx' 文件，请检查文件路径。")
    exit(1)

# 遍历每个场景和聚类
for scenario in scenarios:
    for cluster in clusters:
        # 生成对应 Excel 文件名称
        file_name = f'{cluster}_{scenario}_CO2_预测.xlsx'
        try:
            df = pd.read_excel(file_name)
        except FileNotFoundError:
            print(f"未找到 '{file_name}' 文件，请检查文件路径。")
            continue

        # 根据聚类获取对应的省份名称
        try:
            province = [k for k, v in province_mapping.items() if v == cluster][0]
        except IndexError:
            print(f"聚类 {cluster} 未在映射中找到对应的省份！")
            continue

        # 将 '省份' 和 '年份' 列设置为多重索引
        df.set_index(['省份', '年份'], inplace=True)

        # 筛选出需要修正的数值型列
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 获取该省份的 ARIMA 预测值（arima_df 的索引为年份）
        arima_values = arima_df[province]
        arima_avg = arima_values.mean()  # 计算 ARIMA 预测值的均值
        mean_values = df.groupby(level='省份').mean()  # 单省份预测的均值

        # 用字典存储修正后的数值列
        corrected_dict = {}
        for col in numeric_cols:
            year_factor = df.index.get_level_values('年份').map(arima_values)
            if cluster == 'ClusterIV':
                # 类别四使用 arima 预测值 / 单省份预测均值 * 原值
                mean_val = mean_values.loc[province, col]
                if mean_val == 0:
                    print(f"场景: {scenario}, 聚类: {cluster}, 列: {col} 的单省份预测均值为零，跳过修正，直接使用原值。")
                    corrected_series = df[col].copy()
                else:
                    corrected_series = df[col] * (year_factor / mean_val)
            else:
                # 其他类别使用 arima 预测值 / arima 均值 * 原值
                if arima_avg == 0:
                    print(f"场景: {scenario}, 聚类: {cluster}, ARIMA 预测值均值为零，跳过修正，直接使用原值。")
                    corrected_series = df[col].copy()
                else:
                    corrected_series = df[col] * (year_factor / arima_avg)
            
            # 填充计算过程中可能产生的缺失值
            if corrected_series.isna().any():
                print(f"场景: {scenario}, 聚类: {cluster}, 列: {col} 存在缺失值，将用原始值填充。")
                corrected_series = corrected_series.fillna(df[col])
            
            corrected_dict[f'{col}_修正'] = corrected_series

        # 构建只包含修正数据的 DataFrame（同时保留多重索引信息）
        corrected_df = pd.DataFrame(corrected_dict, index=df.index)

        # 重置索引，将 '省份' 和 '年份' 变为普通列
        corrected_df = corrected_df.reset_index()

        # 输出到新的 Excel 文件，只包含修正后的列和对应的省份、年份信息（非汇总）
        output_file_name = f'{cluster}_{scenario}_CO2_预测_修正.xlsx'
        try:
            corrected_df.to_excel(output_file_name, index=False)
            print(f"场景: {scenario}, 聚类: {cluster} —— 已生成仅含修正值的文件：{output_file_name}")
        except Exception as e:
            print(f"写入文件 {output_file_name} 出错：{e}")

        # 根据相同的年份汇总同一聚类内来自不同省份的修正值相加（汇总）
        agg_df = corrected_df.groupby('年份', as_index=False).sum(numeric_only=True)

        # 输出汇总文件（即：同一聚类、同一情景下，不同省份相同年份的修正值相加）
        output_agg_file_name = f'{cluster}_{scenario}_CO2_预测_修正_汇总.xlsx'
        try:
            agg_df.to_excel(output_agg_file_name, index=False)
            print(f"场景: {scenario}, 聚类: {cluster} —— 已生成相同年份数据相加的汇总文件：{output_agg_file_name}")
        except Exception as e:
            print(f"写入文件 {output_agg_file_name} 出错：{e}")