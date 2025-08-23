import pandas as pd
import numpy as np
from scipy import stats

# 显著性标记函数
def get_significance(p, alpha=0.05):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < alpha:
        return '*'
    else:
        return ''

# 分布列表（可根据需要调整）
distributions = ['lognorm', 'expon', 'gamma', 'beta', 'weibull_min']

# KS检验函数，对一列数据测试多种分布
def ks_test_all_distributions(data, var_name, sheet_name):
    results = []
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            stat, p = stats.kstest(data, dist_name, args=params)
            results.append({
                'sheet名': sheet_name,
                '变量名': var_name,
                '检验分布': dist_name,
                'KS统计量': round(stat, 4),
                'P值': round(p, 4),
                '显著性': get_significance(p)
            })
        except Exception:
            results.append({
                'sheet名': sheet_name,
                '变量名': var_name,
                '检验分布': dist_name,
                'KS统计量': None,
                'P值': None,
                '显著性': ''
            })
    return results

# === 主程序部分 ===
input_file = r"C:\新建 XLSX 工作表 (2).xlsx"
output_file = r"C:\多分布检验结果.xlsx"

xlsx = pd.ExcelFile(input_file)
all_results = []

for sheet_name in xlsx.sheet_names:
    df = xlsx.parse(sheet_name)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) >= 3 and data.std(ddof=0) > 0:
            res = ks_test_all_distributions(data, col, sheet_name)
            all_results.extend(res)

# 保存所有结果
final_df = pd.DataFrame(all_results)
final_df.to_excel(output_file, index=False)
print(f"多分布K-S检验完成，结果已保存为：{output_file}")
