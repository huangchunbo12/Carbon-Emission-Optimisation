import pandas as pd
import numpy as np

# ====== Explicitly import 12 files ======
df1 = pd.read_excel(r'C:\Users\ClusterI_BAU_CO2_Predictions.xlsx')
df2 = pd.read_excel(r'C:\Users\ClusterI_HG_CO2_Predictions.xlsx')
df3 = pd.read_excel(r'C:\Users\ClusterI_MS_CO2_Predictions.xlsx')
df4 = pd.read_excel(r'C:\Users\ClusterII_BAU_CO2_Predictions.xlsx')
df5 = pd.read_excel(r'C:\Users\ClusterII_HG_CO2_Predictions.xlsx')
df6 = pd.read_excel(r'C:\Users\ClusterII_MS_CO2_Predictions.xlsx')
df7 = pd.read_excel(r'C:\Users\ClusterIII_BAU_CO2_Predictions.xlsx')
df8 = pd.read_excel(r'C:\Users\ClusterIII_HG_CO2_Predictions.xlsx')
df9 = pd.read_excel(r'C:\Users\ClusterIII_MS_CO2_Predictions.xlsx')
df10 = pd.read_excel(r'C:\Users\ClusterIV_BAU_CO2_Predictions.xlsx')
df11 = pd.read_excel(r'C:\Users\ClusterIV_HG_CO2_Predictions.xlsx')
df12 = pd.read_excel(r'C:\Users\ClusterIV_MS_CO2_Predictions.xlsx')

# ====== Place in a list for unified processing ======
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]
results = []

z = 2.576  # Z-value for 99% confidence interval (95% CI is 1.96)

for df in dfs:
    provinces = df.iloc[:, 0]
    years = df.iloc[:, 1]
    data = df.iloc[:, 2:]

    mean = data.mean(axis=1, skipna=True)
    std = data.std(axis=1, skipna=True, ddof=1)
    count = data.count(axis=1)  # Number of valid samples per row (n)

    ci_halfwidth = z * std / np.sqrt(count)

    result = pd.DataFrame({
        'Province': provinces,
        'Year': years,
        'Mean': mean,
        'CI_HalfWidth': ci_halfwidth
    })
    results.append(result)

# ====== Save to one Excel file with multiple sheets ======
save_path = r'C:\Users\15549\Desktop\Means_and_Confidence_Intervals.xlsx'
with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
    for i, result in enumerate(results, start=1):
        sheet_name = f'Sheet_{i}'
        result.to_excel(writer, sheet_name=sheet_name, index=False)

print(f'Saving complete. File path: {save_path}')