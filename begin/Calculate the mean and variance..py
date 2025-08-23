import pandas as pd
import numpy as np

# 输入输出Excel文件路径
input_file = r"C:\Users\新建 XLSX 工作表 (2).xlsx"
output_file = r'C:\Users\变量均值方差结果.xlsx'

# 读取Excel文件（包含多个sheet）
xlsx = pd.ExcelFile(input_file)
writer = pd.ExcelWriter(output_file, engine='openpyxl')

# 遍历每个sheet
for sheet_name in xlsx.sheet_names:
    df = xlsx.parse(sheet_name)

    # 选择数值型列（排除非数值列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []  # 存储每一列的均值与方差

    # 计算每一列的均值与方差
    for col in numeric_cols:
        data = df[col].dropna()  # 删除缺失值
        mean_val = round(data.mean(), 3)
        std_val = round(data.std(ddof=0), 3)  # 标准差（总体）
        results.append([col, mean_val, std_val])

    # 生成结果DataFrame，列名为：列名, 均值, 方差
    result_df = pd.DataFrame(results, columns=['列名', '均值', '方差'])

    # 写入对应sheet
    result_df.to_excel(writer, sheet_name=sheet_name, index=False)

# 保存结果
writer.close()
print("变量均值与方差结果已保存到", output_file)
