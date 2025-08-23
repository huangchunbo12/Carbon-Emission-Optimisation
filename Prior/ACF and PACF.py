import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 设置新罗马字体
plt.rcParams['font.family'] = 'Times New Roman'

# 文件路径
file_paths = [
    r"C:\Users\Cluster_0_data.xlsx",
    r"C:\Users\Cluster_1_data.xlsx",
    r"C:\Users\Cluster_2_data.xlsx",
    r"C:\Users\Cluster_3_data.xlsx"
]

# 显示用的 Cluster 名称
cluster_names = {
    0: 'Cluster I',
    1: 'Cluster II',
    2: 'Cluster III',
    3: 'Cluster IV'
}

# 每个 Cluster 的差分次数（根据ADF检验结论）
diff_order = {
    0: 0,  # Cluster I：原始序列平稳
    1: 0,  # Cluster II：原始序列平稳
    2: 2,  # Cluster III：二阶差分平稳
    3: 1   # Cluster IV：一阶差分平稳
}

# 主循环
for i, file_path in enumerate(file_paths):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    time_series = df.iloc[:, 1].dropna()

    # 按需差分
    d = diff_order[i]
    series_for_adf = time_series.copy()
    for _ in range(d):
        series_for_adf = series_for_adf.diff().dropna()

    # ADF检验：不带趋势（regression='n'）
    adf_result = adfuller(series_for_adf, autolag='AIC', regression='n')

    # 打印结果
    print(f"\n{cluster_names[i]} ADF Test Results (difference order = {d}):")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print(f"Critical Values: {adf_result[4]}")

    # 绘图
    max_lags = min(40, len(series_for_adf) // 2)
    title_suffix = f" (difference order = {d})"

    plt.figure(figsize=(7, 3))
    plt.subplot(121)
    plot_acf(series_for_adf, lags=max_lags, ax=plt.gca(), title=f'{cluster_names[i]} ACF{title_suffix}')
    plt.subplot(122)
    plot_pacf(series_for_adf, lags=max_lags, ax=plt.gca(), title=f'{cluster_names[i]} PACF{title_suffix}')
    plt.tight_layout()
    plt.show()
