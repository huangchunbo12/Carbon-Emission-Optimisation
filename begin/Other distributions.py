import pandas as pd
import numpy as np
from scipy import stats

# === 1. 加载数据 ===
input_file = '非正态数据.xlsx'  # 替换为你的 Excel 文件名
df = pd.read_excel(input_file)

# 只选择前两列，并删除缺失值
col1 = df.iloc[:, 0].dropna()
col2 = df.iloc[:, 1].dropna()

# === 2. 定义要拟合的分布类型 ===
distributions = ['lognorm', 'expon', 'gamma', 'beta', 'weibull_min']

# === 3. 定义检验函数 ===
def ks_test_all_distributions(data, var_name):
    results = []
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            stat, p = stats.kstest(data, dist_name, args=params)
            results.append({
                '变量名': var_name,
                '检验分布': dist_name,
                'KS统计量': round(stat, 4),
                'P值': round(p, 4)
            })
        except Exception as e:
            results.append({
                '变量名': var_name,
                '检验分布': dist_name,
                'KS统计量': None,
                'P值': None
            })
    return results

# === 4. 分别检验第一列和第二列 ===
results_col1 = ks_test_all_distributions(col1, df.columns[0])
results_col2 = ks_test_all_distributions(col2, df.columns[1])

# === 5. 合并结果并保存 ===
final_results = pd.DataFrame(results_col1 + results_col2)
output_file = '分布检验结果.xlsx'
final_results.to_excel(output_file, index=False)
print(f"检验完成，结果已保存为 {output_file}")
