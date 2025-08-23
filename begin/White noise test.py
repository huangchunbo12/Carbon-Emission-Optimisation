import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

# 读取Excel数据
file_path = r"C:\Users\ARIMA残差表.xlsx"
data = pd.read_excel(file_path)

# Ljung-Box Q检验
results = {}
for column in data.columns:
    # 进行Ljung-Box Q检验，滞后项设置为最大可以的滞后（样本点数-1）
    lb_test = acorr_ljungbox(data[column], lags=min(10, len(data)-1), return_df=True)
    results[column] = lb_test

# 将结果整理成一个DataFrame
output_df = pd.DataFrame()

for column, result in results.items():
    output_df[f'{column}_Q_statistic'] = result['lb_stat']
    output_df[f'{column}_p_value'] = result['lb_pvalue']

# 将结果保存为Excel文件
output_file = r"C:\Users\15549\Desktop\Ljung_Box_Test_Results.xlsx"
output_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
