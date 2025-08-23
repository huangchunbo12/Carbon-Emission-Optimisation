import pandas as pd
import numpy as np

# ====== 明确导入 12 个文件 ======
df1 = pd.read_excel(r'C:\Users\ClusterI_BAU_CO2_预测.xlsx')
df2 = pd.read_excel(r'C:\Users\ClusterI_HG_CO2_预测.xlsx')
df3 = pd.read_excel(r'C:\Users\ClusterI_MS_CO2_预测.xlsx')
df4 = pd.read_excel(r'C:\Users\ClusterII_BAU_CO2_预测.xlsx')
df5 = pd.read_excel(r'C:\Users\ClusterII_HG_CO2_预测.xlsx')
df6 = pd.read_excel(r'C:\Users\ClusterII_MS_CO2_预测.xlsx')
df7 = pd.read_excel(r'C:\Users\ClusterIII_BAU_CO2_预测.xlsx')
df8 = pd.read_excel(r'C:\Users\ClusterIII_HG_CO2_预测.xlsx')
df9 = pd.read_excel(r'C:\Users\ClusterIII_MS_CO2_预测.xlsx')
df10 = pd.read_excel(r'C:\Users\ClusterIV_BAU_CO2_预测.xlsx')
df11 = pd.read_excel(r'C:\Users\ClusterIV_HG_CO2_预测.xlsx')
df12 = pd.read_excel(r'C:\Users\ClusterIV_MS_CO2_预测.xlsx')

# ====== 放入列表统一处理 ======
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]
results = []

z = 2.576  # 正态分布下95%置信区间的Z值

for df in dfs:
    provinces = df.iloc[:, 0]
    years = df.iloc[:, 1]
    data = df.iloc[:, 2:]

    mean = data.mean(axis=1, skipna=True)
    std = data.std(axis=1, skipna=True, ddof=1)
    count = data.count(axis=1)  # 每行有效样本数 n

    ci_halfwidth = z * std / np.sqrt(count)

    result = pd.DataFrame({
        '省份': provinces,
        'Year': years,
        'Mean': mean,
        'CI_HalfWidth': ci_halfwidth
    })
    results.append(result)

# ====== 保存到一个 Excel 文件（多个 sheet） ======
save_path = r'C:\Users\15549\Desktop\均值置信区间_合集_2.xlsx'
with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
    for i, result in enumerate(results, start=1):
        sheet_name = f'Sheet_{i}'
        result.to_excel(writer, sheet_name=sheet_name, index=False)

print(f'保存完成，文件路径：{save_path}')
