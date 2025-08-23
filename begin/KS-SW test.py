import pandas as pd
import numpy as np
from scipy.stats import shapiro


def get_significance(p, alpha=0.05):
    """根据P值返回显著性标记：p<0.001返回'***'，p<0.01返回'**'，p<0.05返回'*'，否则返回空字符串。"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < alpha:
        return '*'
    else:
        return ''


# 输入输出Excel文件路径
input_file = r"C:\Users\新建 XLSX 工作表 (2).xlsx"
output_file = r'C:\Users\SW_test_results.xlsx'

# 读取Excel文件（包含多个sheet）
xlsx = pd.ExcelFile(input_file)
writer = pd.ExcelWriter(output_file, engine='openpyxl')

# 显著性水平
alpha = 0.05

# 遍历每个sheet
for sheet_name in xlsx.sheet_names:
    df = xlsx.parse(sheet_name)

    # 选择数值型列（排除年份、省份等非数值型列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []  # 存储本sheet的结果

    # 对每一列分别计算SW检验
    for col in numeric_cols:
        data = df[col].dropna()  # 删除缺失值

        # 如果数据样本量太少或标准差为0，则无法进行检验
        if len(data) < 3 or data.std(ddof=0) == 0:
            sw_stat, sw_p = np.nan, np.nan
        else:
            sw_stat, sw_p = shapiro(data)

        # 格式化统计量和P值均保留3位小数
        sw_stat_str = f"{sw_stat:.3f}" if not np.isnan(sw_stat) else ""
        sw_p_str = f"{sw_p:.3f}" if not np.isnan(sw_p) else ""

        # 添加显著性标记
        signif = get_significance(sw_p, alpha) if not np.isnan(sw_p) else ""
        sw_p_marked = f"{sw_p_str}{signif}" if sw_p_str != "" else ""

        results.append([col, sw_stat_str, sw_p_marked])

    # 生成结果DataFrame，列为：列名, SW统计量, SW P值及显著性
    result_df = pd.DataFrame(results, columns=['列名', 'SW统计量', 'SW P值及显著性'])
    # 将结果写入输出Excel的对应sheet
    result_df.to_excel(writer, sheet_name=sheet_name, index=False)

# 保存输出Excel
writer.close()
print("SW检验结果已保存到", output_file)

