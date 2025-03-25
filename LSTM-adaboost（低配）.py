import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import random
import math
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import pickle
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# ---------------------------
# 默认参数（可根据需要进行修改）
# ---------------------------
# XGB参数
default_n_estimators_xgb = 10
default_learning_rate_xgb = 0.1
default_max_depth_xgb = 3

# GBDT参数
default_n_estimators_gbdt = 10
default_learning_rate_gbdt = 0.1
default_max_depth_gbdt = 3

# AdaBoost参数
default_n_estimators_ada = 10
default_learning_rate_ada = 1.0

# Random Forest参数
default_n_estimators_rf = 10
default_max_depth_rf = None  # 使用 None 表示树的深度不限

# ---------------------------
# 数据加载及预处理
# ---------------------------
file_path = r'Cluster_0_data.xlsx'
data = pd.read_excel(file_path)
cluster_name = os.path.splitext(os.path.basename(file_path))[0]

X_columns = ['G', 'SI', 'ES', 'EI', 'DMSP']
y_column = 'CO2 Emissions'

# 修改数据集划分：
# 训练集：2000-2015 和 2020-2222
train_data = data[((data['年份'] >= 2000) & (data['年份'] <= 2015)) | ((data['年份'] >= 2020) & (data['年份'] <= 2222))]
# 测试集保持不变：2016-2019
test_data  = data[(data['年份'] >= 2016) & (data['年份'] <= 2019)]
# 验证集修改为：2000-2002
val_data   = data[(data['年份'] >= 2000) & (data['年份'] <= 2002)]

X_train = train_data[X_columns]
y_train = train_data[y_column]
X_test  = test_data[X_columns]
y_test  = test_data[y_column]
X_val   = val_data[X_columns]
y_val   = val_data[y_column]

# 对特征归一化（所有模型均采用归一化后的特征）
scaler_X = MinMaxScaler()
X_train_scaled_np = scaler_X.fit_transform(X_train)
X_test_scaled_np  = scaler_X.transform(X_test)
X_val_scaled_np   = scaler_X.transform(X_val)

# 目标值采用原始数值（各模型均使用原始目标）
y_train_np = y_train.values.reshape(-1, 1)
y_test_np  = y_test.values.reshape(-1, 1)
y_val_np   = y_val.values.reshape(-1, 1)

# ---------------------------
# 评估指标计算函数：R2, VAF, RMSLE, KGE
# ---------------------------
def compute_kge(y_true, y_pred):
    # Kling-Gupta Efficiency 指标
    cc = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    vaf = 1 - np.var(y_true - y_pred) / np.var(y_true)
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.maximum(0, y_pred))
    rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    kge = compute_kge(y_true, y_pred)
    return r2, vaf, rmsle, kge

# ---------------------------
# 主程序：使用各模型默认参数（去除ALA优化算法）
# ---------------------------
if __name__ == '__main__':
    # 使用默认超参数训练各模型
    
    # 1. XGB模型
    print("训练XGB模型（默认参数）……")
    model_xgb = xgb.XGBRegressor(
        n_estimators=default_n_estimators_xgb,
        learning_rate=default_learning_rate_xgb,
        max_depth=default_max_depth_xgb,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )
    model_xgb.fit(X_train_scaled_np, y_train.values.ravel())
    train_pred_xgb = model_xgb.predict(X_train_scaled_np)
    test_pred_xgb = model_xgb.predict(X_test_scaled_np)
    val_pred_xgb = model_xgb.predict(X_val_scaled_np)
    
    # 2. GBDT模型
    print("训练GBDT模型（默认参数）……")
    model_gbdt = GradientBoostingRegressor(
        n_estimators=default_n_estimators_gbdt,
        learning_rate=default_learning_rate_gbdt,
        max_depth=default_max_depth_gbdt,
        random_state=42
    )
    model_gbdt.fit(X_train_scaled_np, y_train.values.ravel())
    train_pred_gbdt = model_gbdt.predict(X_train_scaled_np)
    test_pred_gbdt = model_gbdt.predict(X_test_scaled_np)
    val_pred_gbdt = model_gbdt.predict(X_val_scaled_np)
    
    # 3. AdaBoost模型
    print("训练AdaBoost模型（默认参数）……")
    model_ada = AdaBoostRegressor(
        n_estimators=default_n_estimators_ada,
        learning_rate=default_learning_rate_ada,
        random_state=42
    )
    model_ada.fit(X_train_scaled_np, y_train.values.ravel())
    train_pred_ada = model_ada.predict(X_train_scaled_np)
    test_pred_ada = model_ada.predict(X_test_scaled_np)
    val_pred_ada = model_ada.predict(X_val_scaled_np)
    
    # 4. Random Forest模型
    print("训练Random Forest模型（默认参数）……")
    model_rf = RandomForestRegressor(
        n_estimators=default_n_estimators_rf,
        max_depth=default_max_depth_rf,
        random_state=42
    )
    model_rf.fit(X_train_scaled_np, y_train.values.ravel())
    train_pred_rf = model_rf.predict(X_train_scaled_np)
    test_pred_rf = model_rf.predict(X_test_scaled_np)
    val_pred_rf = model_rf.predict(X_val_scaled_np)
    
    # ---------------------------
    # 5. 交叉验证：计算各基模型的CV指标
    # ---------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {'XGB': [], 'GBDT': [], 'AdaBoost': [], 'RF': []}
    
    for train_idx, valid_idx in kf.split(X_train_scaled_np):
        X_cv_train, X_cv_valid = X_train_scaled_np[train_idx], X_train_scaled_np[valid_idx]
        y_cv_train, y_cv_valid = y_train.values.ravel()[train_idx], y_train.values.ravel()[valid_idx]
        
        # XGB
        m = xgb.XGBRegressor(
            n_estimators=default_n_estimators_xgb,
            learning_rate=default_learning_rate_xgb,
            max_depth=default_max_depth_xgb,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0
        )
        m.fit(X_cv_train, y_cv_train)
        preds = m.predict(X_cv_valid)
        cv_results['XGB'].append(compute_metrics(y_cv_valid, preds))
        
        # GBDT
        m = GradientBoostingRegressor(
            n_estimators=default_n_estimators_gbdt,
            learning_rate=default_learning_rate_gbdt,
            max_depth=default_max_depth_gbdt,
            random_state=42
        )
        m.fit(X_cv_train, y_cv_train)
        preds = m.predict(X_cv_valid)
        cv_results['GBDT'].append(compute_metrics(y_cv_valid, preds))
        
        # AdaBoost
        m = AdaBoostRegressor(
            n_estimators=default_n_estimators_ada,
            learning_rate=default_learning_rate_ada,
            random_state=42
        )
        m.fit(X_cv_train, y_cv_train)
        preds = m.predict(X_cv_valid)
        cv_results['AdaBoost'].append(compute_metrics(y_cv_valid, preds))
        
        # Random Forest
        m = RandomForestRegressor(
            n_estimators=default_n_estimators_rf,
            max_depth=default_max_depth_rf,
            random_state=42
        )
        m.fit(X_cv_train, y_cv_train)
        preds = m.predict(X_cv_valid)
        cv_results['RF'].append(compute_metrics(y_cv_valid, preds))
    
    # 计算各基模型CV指标均值（顺序：R2, VAF, RMSLE, KGE）
    cv_metrics_mean = {}
    for model_name in cv_results:
        cv_array = np.array(cv_results[model_name])
        cv_metrics_mean[model_name] = cv_array.mean(axis=0)
    
    # ---------------------------
    # 6. Stacking集成
    # 以四个基模型的预测作为元特征
    # ---------------------------
    train_stack = np.column_stack((train_pred_xgb, train_pred_gbdt, train_pred_ada, train_pred_rf))
    test_stack  = np.column_stack((test_pred_xgb, test_pred_gbdt, test_pred_ada, test_pred_rf))
    val_stack   = np.column_stack((val_pred_xgb, val_pred_gbdt, val_pred_ada, val_pred_rf))
    
    meta_train_linear = np.zeros(train_stack.shape[0])
    meta_train_tree = np.zeros(train_stack.shape[0])
    meta_test_preds_linear = []
    meta_val_preds_linear = []
    meta_test_preds_tree = []
    meta_val_preds_tree = []
    
    for train_idx, valid_idx in kf.split(train_stack):
        X_meta_train, X_meta_valid = train_stack[train_idx], train_stack[valid_idx]
        y_meta_train = y_train.values.ravel()[train_idx]
        
        # 线性模型
        model_linear = LinearRegression()
        model_linear.fit(X_meta_train, y_meta_train)
        meta_train_linear[valid_idx] = model_linear.predict(X_meta_valid)
        meta_test_preds_linear.append(model_linear.predict(test_stack))
        meta_val_preds_linear.append(model_linear.predict(val_stack))
        
        # 决策树模型
        model_tree = DecisionTreeRegressor(random_state=42)
        model_tree.fit(X_meta_train, y_meta_train)
        meta_train_tree[valid_idx] = model_tree.predict(X_meta_valid)
        meta_test_preds_tree.append(model_tree.predict(test_stack))
        meta_val_preds_tree.append(model_tree.predict(val_stack))
    
    test_pred_stack_linear = np.mean(meta_test_preds_linear, axis=0)
    val_pred_stack_linear  = np.mean(meta_val_preds_linear, axis=0)
    test_pred_stack_tree = np.mean(meta_test_preds_tree, axis=0)
    val_pred_stack_tree  = np.mean(meta_val_preds_tree, axis=0)
    
    # ---------------------------
    # 7. 计算评估指标
    # ---------------------------
    metrics = {}
    for name, preds_train, preds_test, preds_val in [
        ("XGB", train_pred_xgb, test_pred_xgb, val_pred_xgb),
        ("GBDT", train_pred_gbdt, test_pred_gbdt, val_pred_gbdt),
        ("AdaBoost", train_pred_ada, test_pred_ada, val_pred_ada),
        ("RF", train_pred_rf, test_pred_rf, val_pred_rf)
    ]:
        metrics[name] = {
            'Train': compute_metrics(y_train.values.ravel(), preds_train),
            'Test': compute_metrics(y_test.values.ravel(), preds_test),
            'Validation': compute_metrics(y_val.values.ravel(), preds_val)
        }
    
    stacking_metrics = {}
    stacking_metrics['Linear'] = {
        'Test': compute_metrics(y_test.values.ravel(), test_pred_stack_linear),
        'Validation': compute_metrics(y_val.values.ravel(), val_pred_stack_linear)
    }
    stacking_metrics['Tree'] = {
        'Test': compute_metrics(y_test.values.ravel(), test_pred_stack_tree),
        'Validation': compute_metrics(y_val.values.ravel(), val_pred_stack_tree)
    }
    
    # ---------------------------
    # 8. 保存结果到Excel文件
    # ---------------------------
    excel_file_name = f"{cluster_name}_Default_Results.xlsx"
    
    base_models_results = {}
    for name in metrics:
        base_models_results[name] = pd.DataFrame({
            'Dataset': ['Train', 'Test', 'Validation'],
            'R2': [metrics[name]['Train'][0], metrics[name]['Test'][0], metrics[name]['Validation'][0]],
            'VAF': [metrics[name]['Train'][1], metrics[name]['Test'][1], metrics[name]['Validation'][1]],
            'RMSLE': [metrics[name]['Train'][2], metrics[name]['Test'][2], metrics[name]['Validation'][2]],
            'KGE': [metrics[name]['Train'][3], metrics[name]['Test'][3], metrics[name]['Validation'][3]]
        })
    
    cv_metrics_df = pd.DataFrame({
        'Model': list(cv_metrics_mean.keys()),
        'R2': [cv_metrics_mean[m][0] for m in cv_metrics_mean],
        'VAF': [cv_metrics_mean[m][1] for m in cv_metrics_mean],
        'RMSLE': [cv_metrics_mean[m][2] for m in cv_metrics_mean],
        'KGE': [cv_metrics_mean[m][3] for m in cv_metrics_mean]
    })
    
    stacking_results_linear = pd.DataFrame({
        'Dataset': ['Test', 'Validation'],
        'R2': [stacking_metrics['Linear']['Test'][0], stacking_metrics['Linear']['Validation'][0]],
        'VAF': [stacking_metrics['Linear']['Test'][1], stacking_metrics['Linear']['Validation'][1]],
        'RMSLE': [stacking_metrics['Linear']['Test'][2], stacking_metrics['Linear']['Validation'][2]],
        'KGE': [stacking_metrics['Linear']['Test'][3], stacking_metrics['Linear']['Validation'][3]]
    })
    stacking_results_tree = pd.DataFrame({
        'Dataset': ['Test', 'Validation'],
        'R2': [stacking_metrics['Tree']['Test'][0], stacking_metrics['Tree']['Validation'][0]],
        'VAF': [stacking_metrics['Tree']['Test'][1], stacking_metrics['Tree']['Validation'][1]],
        'RMSLE': [stacking_metrics['Tree']['Test'][2], stacking_metrics['Tree']['Validation'][2]],
        'KGE': [stacking_metrics['Tree']['Test'][3], stacking_metrics['Tree']['Validation'][3]]
    })
    
    with pd.ExcelWriter(excel_file_name) as writer:
        for name in base_models_results:
            base_models_results[name].to_excel(writer, sheet_name=f"{name} Metrics", index=False)
        cv_metrics_df.to_excel(writer, sheet_name="CV Metrics", index=False)
        stacking_results_linear.to_excel(writer, sheet_name="Stacking Linear", index=False)
        stacking_results_tree.to_excel(writer, sheet_name="Stacking Tree", index=False)
        default_params_df = pd.DataFrame({
            'Model': ['XGB', 'GBDT', 'AdaBoost', 'RF'],
            'Parameters': [
                f"n_estimators={default_n_estimators_xgb}, learning_rate={default_learning_rate_xgb}, max_depth={default_max_depth_xgb}",
                f"n_estimators={default_n_estimators_gbdt}, learning_rate={default_learning_rate_gbdt}, max_depth={default_max_depth_gbdt}",
                f"n_estimators={default_n_estimators_ada}, learning_rate={default_learning_rate_ada}",
                f"n_estimators={default_n_estimators_rf}, max_depth={default_max_depth_rf}"
            ]
        })
        default_params_df.to_excel(writer, sheet_name="Default Parameters", index=False)
    
    print(f"结果保存为 '{excel_file_name}'")
    
    # ---------------------------
    # 9. 保存Stacking模型权重
    # ---------------------------
    meta_linear_final = LinearRegression().fit(train_stack, y_train.values.ravel())
    meta_tree_final = DecisionTreeRegressor(random_state=42).fit(train_stack, y_train.values.ravel())
    
    stacking_linear_file = f"{cluster_name}_stacking_linear_model_default.pkl"
    with open(stacking_linear_file, "wb") as f:
        pickle.dump(meta_linear_final, f)
    stacking_tree_file = f"{cluster_name}_stacking_tree_model_default.pkl"
    with open(stacking_tree_file, "wb") as f:
        pickle.dump(meta_tree_final, f)
    
    print(f"保存Stacking模型权重：'{stacking_linear_file}' 和 '{stacking_tree_file}'")
    print("\n任务已完成。")
