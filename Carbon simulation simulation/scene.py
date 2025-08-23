import pandas as pd
import numpy as np
import os

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 需要模拟的指标（只处理这5个指标）
indicators = ['G', 'SI', 'ES', 'EI', 'DMSP']

# 每个簇对应的数据文件及参数文件中“城市类型”的映射
cluster_files = {
    0: "Cluster_0_data.xlsx",
    1: "Cluster_1_data.xlsx",
    2: "Cluster_2_data.xlsx",
    3: "Cluster_3_data.xlsx"
}
cluster_sim_names = {
    0: "Cluster I",
    1: "Cluster II",
    2: "Cluster III",
    3: "Cluster IV"
}

# 读取仿真参数文件，文件中包含以下列：
# 城市类型, 情景, 时间阶段, G_均值, G_方差, SI_均值, SI_方差, EI_均值, EI_方差, ES_均值, ES_方差, DMSP_均值, DMSP_方差
param_file = "仿真参数设定表_最终版.xlsx"
sim_params = pd.read_excel(param_file)

# 输出结果保存目录
output_dir = "simulation_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 模拟目标年份：从2023年到2050年
target_years = list(range(2023, 2051))

# 定义辅助函数，根据年份判断所属时间阶段字符串
def get_time_stage(year):
    if 2023 <= year <= 2030:
        return "2023–2030"
    elif 2031 <= year <= 2040:
        return "2031–2040"
    elif 2041 <= year <= 2050:
        return "2041–2050"
    else:
        return None

# 遍历每个簇
for cluster in range(4):
    cluster_file = cluster_files[cluster]
    cluster_name = cluster_sim_names[cluster]
    
    try:
        # 读取当前簇的全部数据
        data = pd.read_excel(cluster_file)
    except Exception as e:
        print(f"读取文件 {cluster_file} 失败：{e}")
        continue

    # 检查数据是否包含“年份”和“省份”
    if not set(['年份', '省份']).issubset(data.columns):
        print(f"文件 {cluster_file} 缺少‘年份’和‘省份’列")
        continue

    # 提取2022年的数据作为初始基准
    baseline = data[data['年份'] == 2022].copy()
    if baseline.empty:
        print(f"文件 {cluster_file} 中未找到2022年的数据")
        continue
    baseline.reset_index(drop=True, inplace=True)

    # 筛选当前簇对应的参数（按“城市类型”匹配）
    cluster_params = sim_params[sim_params["城市类型"] == cluster_name]
    if cluster_params.empty:
        print(f"未找到 {cluster_name} 的仿真参数")
        continue

    # 当前簇可能存在多个情景（例如 MS、BAU、HG）
    scenarios = cluster_params["情景"].unique()
    for scenario in scenarios:
        # 筛选该情景下的所有参数（不同时间阶段）
        scenario_params = cluster_params[cluster_params["情景"] == scenario]
        
        # 构造输出文件名，例如 "ClusterI_MS_2023-2050.xlsx"
        cluster_name_str = cluster_name.replace(" ", "")
        output_filename = os.path.join(output_dir, f"{cluster_name_str}_{scenario}_2023-2050.xlsx")
        
        # 利用 ExcelWriter 写入多个工作表（每个指标一个工作表）
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # 对每个指标进行链式模拟
            for indicator in indicators:
                records = []
                # 对于baseline中每个省份，针对该指标进行链式模拟
                for idx, row in baseline.iterrows():
                    province = row["省份"]
                    # 初始值为2022年的实际值
                    prev_value = row[indicator]
                    
                    # 对每个目标年份，逐年计算模拟值
                    for year in target_years:
                        time_stage = get_time_stage(year)
                        if time_stage is None:
                            continue
                        # 从情景参数中找到对应时间阶段的参数行（假设每阶段只有一行）
                        param_row = scenario_params[scenario_params["时间阶段"] == time_stage]
                        if param_row.empty:
                            print(f"{cluster_name} {scenario} 无法找到时间阶段 {time_stage} 的参数")
                            continue
                        param_row = param_row.iloc[0]
                        # 构造对应指标的均值和方差列名
                        mean_col = f"{indicator}_均值"
                        var_col = f"{indicator}_方差"
                        if mean_col not in param_row or var_col not in param_row:
                            print(f"缺少 {indicator} 的参数列")
                            continue
                        mean_val = param_row[mean_col]
                        var_val = param_row[var_col]
                        std_val = np.sqrt(var_val)
                        
                        # 为当前年份生成500个随机增长率（单位%）
                        sim_rates = np.random.normal(loc=mean_val, scale=std_val, size=500)
                        # 模拟公式：本年值 = 前一年值 * (1 + 增长率/100)
                        sim_values = prev_value * (1 + sim_rates/100)
                        
                        # 记录当前年份各模拟路径的值
                        record = {"省份": province, "年份": year}
                        for i, val in enumerate(sim_values):
                            record[f"模拟_{i+1}"] = val
                        records.append(record)
                        
                        # 更新 prev_value：这里为链式模拟，每个模拟路径独立
                        # 为了保证链式，每个模拟路径的前值不同，我们保存500条链式结果
                        prev_value = sim_values  # 此时prev_value为一个数组，后续每次与新增长率相乘
                # 转换为 DataFrame
                result_df = pd.DataFrame(records)
                result_df.sort_values(by=["省份", "年份"], inplace=True)
                # 写入Excel工作表，工作表名称为指标名称
                result_df.to_excel(writer, sheet_name=indicator, index=False)
        
        print(f"生成模拟结果文件：{output_filename}")
