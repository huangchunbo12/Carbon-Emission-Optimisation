import pandas as pd
import numpy as np
import pickle
import os
import sys
import hashlib
from sklearn.preprocessing import MinMaxScaler

# 修复 numpy 兼容性问题
import numpy.core as np_core
sys.modules['numpy._core'] = np_core

# 设置随机种子
np.random.seed(42)

# 定义5个指标
indicators = ['G', 'SI', 'ES', 'EI', 'DMSP']

# 定义每个簇对应的模型配置
cluster_info = {
    0: {
        "cluster_name": "Cluster I",
        "base_model_pop": "pop50",
        "stacking_model_file": "Cluster_0_data_pop50_stacking_tree_model.pkl",
        "scaler_file": "Cluster_0_data_pop50_scaler_X.pkl",
        "sim_files": {
            "MS": "ClusterI_MS_2023-2050.xlsx",
            "HG": "ClusterI_HG_2023-2050.xlsx",
            "BAU": "ClusterI_BAU_2023-2050.xlsx"
        }
    },
    1: {
        "cluster_name": "Cluster II",
        "base_model_pop": "pop200",
        "stacking_model_file": "Cluster_1_data_pop200_stacking_tree_model.pkl",
        "scaler_file": "Cluster_1_data_pop200_scaler_X.pkl",
        "sim_files": {
            "MS": "ClusterII_MS_2023-2050.xlsx",
            "HG": "ClusterII_HG_2023-2050.xlsx",
            "BAU": "ClusterII_BAU_2023-2050.xlsx"
        }
    },
    2: {
        "cluster_name": "Cluster III",
        "base_model_pop": "pop250",
        "stacking_model_file": "Cluster_2_data_pop250_stacking_tree_model.pkl",
        "scaler_file": "Cluster_2_data_pop250_scaler_X.pkl",
        "sim_files": {
            "MS": "ClusterIII_MS_2023-2050.xlsx",
            "HG": "ClusterIII_HG_2023-2050.xlsx",
            "BAU": "ClusterIII_BAU_2023-2050.xlsx"
        }
    },
    3: {
        "cluster_name": "Cluster IV",
        "base_model_pop": "pop250",
        "stacking_model_file": "Cluster_3_data_pop250_stacking_linear_model.pkl",
        "scaler_file": "Cluster_3_data_pop250_scaler_X.pkl",
        "sim_files": {
            "MS": "ClusterIV_MS_2023-2050.xlsx",
            "HG": "ClusterIV_HG_2023-2050.xlsx",
            "BAU": "ClusterIV_BAU_2023-2050.xlsx"
        }
    }
}

# 输出目录
output_dir = "prediction_outputs"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个簇
for cluster_num in cluster_info:
    info = cluster_info[cluster_num]
    cluster_name = info["cluster_name"]
    base_pop = info["base_model_pop"]
    scaler_file = info["scaler_file"]

    print(f"\n{'='*40}")
    print(f"处理 {cluster_name} ...")

    # ================ 加载归一化 scaler =================
    if os.path.exists(scaler_file):
        try:
            with open(scaler_file, "rb") as f:
                scaler_X = pickle.load(f)
            print(f"成功加载归一化 scaler：{scaler_file}")
        except Exception as e:
            print(f"加载归一化 scaler 失败: {str(e)}")
            scaler_X = MinMaxScaler()
    else:
        print(f"归一化 scaler 文件不存在: {scaler_file}")
        scaler_X = MinMaxScaler()

    # ================ 加载基模型 =================
    base_models = {}
    base_model_types = ['XGB', 'GBDT', 'AdaBoost', 'RF']
    for model_type in base_model_types:
        model_file = f"Cluster_{cluster_num}_data_{base_pop}_{model_type}_model.pkl"
        try:
            with open(model_file, "rb") as f:
                base_models[model_type] = pickle.load(f)
            print(f"成功加载基模型：{model_type}")
        except Exception as e:
            print(f"加载基模型 {model_type} 失败: {str(e)}")
            continue

    if len(base_models) != 4:
        print(f"基模型加载不完整，跳过 {cluster_name}")
        continue

    stacking_model_file = info["stacking_model_file"]
    if os.path.exists(stacking_model_file):
        try:
            with open(stacking_model_file, "rb") as f:
                stacking_model = pickle.load(f)
            print(f"成功加载融合模型：{stacking_model_file}")
        except Exception as e:
            print(f"加载融合模型失败: {str(e)}")
            continue
    else:
        print(f"融合模型文件不存在: {stacking_model_file}")
        continue

    # ================ 处理模拟情景 =================
    for scenario, sim_file in info["sim_files"].items():
        print(f"\n处理情景 {scenario} ...")
        if not os.path.exists(sim_file):
            print(f"模拟文件不存在：{sim_file}")
            continue

        try:
            # 预先读取 Excel 文件中各指标对应的 sheet
            xls = pd.ExcelFile(sim_file)
            indicator_data = {}
            for ind in indicators:
                try:
                    df = pd.read_excel(xls, sheet_name=ind)
                    indicator_data[ind] = df
                    print(f"成功读取指标 {ind} 的数据")
                except Exception as e:
                    print(f"读取指标 {ind} 失败: {str(e)}")
            if len(indicator_data) != len(indicators):
                print("部分指标数据读取失败，跳过该情景")
                continue
        except Exception as e:
            print(f"读取模拟文件失败: {str(e)}")
            continue

        num_sim = 500  # 模拟次数
        predictions_list = []
        base_df = None
        history_hashes = set()

        for sim_idx in range(1, num_sim + 1):
            sim_data = {}
            valid_sim = True
            for ind in indicators:
                df = indicator_data[ind]
                col_name = f"模拟_{sim_idx}"
                if col_name not in df.columns:
                    print(f"指标 {ind} 缺少列 {col_name}，跳过模拟 {sim_idx}")
                    valid_sim = False
                    break
                try:
                    df_sim = df.loc[:, ["省份", "年份", col_name]].copy()
                    df_sim = df_sim.rename(columns={col_name: ind})
                    sim_data[ind] = df_sim
                except Exception as e:
                    print(f"处理指标 {ind} 在模拟 {sim_idx} 时失败: {str(e)}")
                    valid_sim = False
                    break
            if not valid_sim:
                continue

            merged_sim = sim_data[indicators[0]]
            for ind in indicators[1:]:
                merged_sim = pd.merge(merged_sim, sim_data[ind], on=["省份", "年份"], how="inner")

            if sim_idx == 1:
                base_df = merged_sim[["省份", "年份"]].copy()

            feature_cols = indicators
            missing_cols = [col for col in feature_cols if col not in merged_sim.columns]
            if missing_cols:
                print(f"模拟 {sim_idx} 缺少特征列：{missing_cols}")
                continue

            # 生成哈希值，避免重复数据
            current_hash = hashlib.sha256(merged_sim.to_csv().encode()).hexdigest()
            if current_hash in history_hashes:
                print(f"警告：模拟 {sim_idx} 数据与历史轮次重复！")
            else:
                history_hashes.add(current_hash)

            try:
                # 保持DataFrame结构以保留列名
                X_base = merged_sim[feature_cols].copy()

                # 确保 scaler 已经 fit（如果没有，则用当前数据进行 fit）
                if not hasattr(scaler_X, 'min_'):
                    print("scaler_X 尚未 fit，现在使用当前数据进行 fit")
                    scaler_X.fit(X_base)

                X_base_scaled = scaler_X.transform(X_base)

                preds = []
                for model_type in base_model_types:
                    pred = base_models[model_type].predict(X_base_scaled)
                    preds.append(pred.reshape(-1, 1))
                X_stacked = np.hstack(preds)
                y_pred = stacking_model.predict(X_stacked)

                sim_pred_df = pd.DataFrame({f"预测_{sim_idx}": y_pred.flatten()}, index=merged_sim.index)
                predictions_list.append(sim_pred_df)
            except Exception as e:
                print(f"模拟 {sim_idx} 预测失败: {str(e)}")
                continue

        if not predictions_list:
            print("当前情景未生成任何预测结果")
            continue

        # 合并所有模拟的预测结果，与省份和年份数据拼接
        predictions_df = pd.concat(predictions_list, axis=1)
        pred_df = pd.concat([base_df, predictions_df], axis=1)

        # 保存原始预测结果
        output_file = os.path.join(output_dir, f"{cluster_name.replace(' ', '')}_{scenario}_CO2_预测.xlsx")
        try:
            pred_df.to_excel(output_file, index=False)
            print(f"成功保存预测结果：{output_file}")
        except Exception as e:
            print(f"保存结果失败: {str(e)}")

        # ================ 根据“年份”分组，对每个模拟轮次内省份不同但日期相同的数据累加 =================
        # 复制一份 DataFrame，减少内存碎片问题
        pred_df = pred_df.copy()
        agg_pred_df = pred_df.groupby("年份", as_index=False).sum()
        output_file_agg = os.path.join(output_dir, f"{cluster_name.replace(' ', '')}_{scenario}_CO2_预测_aggregated.xlsx")
        try:
            agg_pred_df.to_excel(output_file_agg, index=False)
            print(f"成功保存汇总预测结果：{output_file_agg}")
        except Exception as e:
            print(f"保存汇总结果失败: {str(e)}")

print("\n全部处理完成！")
