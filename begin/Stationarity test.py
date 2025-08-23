import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# 文件路径
file_paths = [
    r"C:\Users\Cluster_0_data.xlsx",
    r"C:\Users\Cluster_1_data.xlsx",
    r"C:\Users\Cluster_2_data.xlsx",
    r"C:\Users\Cluster_3_data.xlsx"
]

# 用于存储结果
results = []

# 定义ADF检验函数
def adf_test(series, trend='c'):
    result = adfuller(series, autolag='AIC', regression=trend)
    return result[0], result[1], result[2]  # t统计量, p值, lag

# 主循环
for file_path in file_paths:
    # 读取数据
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    time_series = df.iloc[:, 1].dropna()

    # 原始序列检验
    t_n, p_n, _ = adf_test(time_series, trend='n')   # 无趋势
    t_c, p_c, _ = adf_test(time_series, trend='c')   # 常数
    t_ct, p_ct, _ = adf_test(time_series, trend='ct')  # 常数 + 趋势

    # 一阶差分
    diff_series = time_series.diff().dropna()
    dt_n, dp_n, _ = adf_test(diff_series, trend='n')
    dt_c, dp_c, _ = adf_test(diff_series, trend='c')
    dt_ct, dp_ct, _ = adf_test(diff_series, trend='ct')

    results.append({
        '文件': file_path,
        't（原：无趋势）': t_n, 'p（原：无趋势）': p_n,
        't（原：常数）': t_c, 'p（原：常数）': p_c,
        't（原：常数+趋势）': t_ct, 'p（原：常数+趋势）': p_ct,
        't（一阶差分：无趋势）': dt_n, 'p（一阶差分：无趋势）': dp_n,
        't（一阶差分：常数）': dt_c, 'p（一阶差分：常数）': dp_c,
        't（一阶差分：常数+趋势）': dt_ct, 'p（一阶差分：常数+趋势）': dp_ct,
    })

# 写入 Excel
results_df = pd.DataFrame(results)
output_path = r"C:\Users\ADF检验结果.xlsx"
results_df.to_excel(output_path, index=False)

print(f"ADF检验结果已保存到 {output_path}")
