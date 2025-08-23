import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# ---------------------------
# 数据加载及预处理（训练、测试、验证按指定年份划分）
# ---------------------------
file_path = r'Cluster_0_data.xlsx'  # 修改文件名
data = pd.read_excel(file_path)
cluster_name = os.path.splitext(os.path.basename(file_path))[0]

X_columns = ['G', 'SI', 'ES', 'EI', 'DMSP']
y_column = 'CO2 Emissions'

# 数据集划分保持不变
train_data = data[((data['年份'] >= 2000) & (data['年份'] <= 2003)) |
                  ((data['年份'] >= 2020) & (data['年份'] <= 2022)) |
                  ((data['年份'] >= 2007) & (data['年份'] <= 2015))]
test_data = data[(data['年份'] >= 2016) & (data['年份'] <= 2019)]
val_data = data[(data['年份'] >= 2004) & (data['年份'] <= 2006)]

X_train = train_data[X_columns]
y_train = train_data[y_column]
X_test = test_data[X_columns]
y_test = test_data[y_column]
X_val = val_data[X_columns]
y_val = val_data[y_column]

# 对特征归一化
scaler_X = MinMaxScaler()
X_train_scaled_np = scaler_X.fit_transform(X_train)
X_test_scaled_np = scaler_X.transform(X_test)
X_val_scaled_np = scaler_X.transform(X_val)

# 目标值使用原始数值
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)
y_val_np = y_val.values.reshape(-1, 1)

# ---------------------------
# 评估指标计算函数：R2, VAF, RMSLE, KGE
# ---------------------------
def compute_kge(y_true, y_pred):
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
# 主程序：直接使用给定超参数训练模型
# ---------------------------
if __name__ == '__main__':
    # 初始化各模型（使用更新后的超参数）
    models = {
        'XGB': xgb.XGBRegressor(
            n_estimators=282,  # 修改参数
            learning_rate=0.22624,
            max_depth=3,
            reg_alpha=0.51622,
            reg_lambda=0.99947,
            random_state=42,
            verbosity=0
        ),
        'GBDT': GradientBoostingRegressor(
            n_estimators=983,  # 修改参数
            learning_rate=0.12682,
            max_depth=6,
            subsample=0.50004,
            min_samples_leaf=5,
            random_state=42
        ),
        'AdaBoost': AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=3,  # 修改参数
                min_samples_leaf=1
            ),
            n_estimators=946,  # 修改参数
            learning_rate=0.78341,
            random_state=42
        ),
        'RF': RandomForestRegressor(
            n_estimators=21,  # 修改参数
            max_depth=10,
            max_features=0.43295,
            random_state=42
        )
    }

    model_objects = {}
    for model_name in models:
        print(f"训练{model_name}模型...")
        model = models[model_name]
        model.fit(X_train_scaled_np, y_train.values.ravel())
        model_objects[model_name] = model

    # ---------------------------
    # 生成Stacking特征
    # ---------------------------
    train_stack = np.zeros((X_train.shape[0], len(models)))
    test_stack = np.zeros((X_test.shape[0], len(models)))
    val_stack = np.zeros((X_val.shape[0], len(models)))

    for i, model_name in enumerate(models):
        model = model_objects[model_name]
        train_stack[:, i] = model.predict(X_train_scaled_np)
        test_stack[:, i] = model.predict(X_test_scaled_np)
        val_stack[:, i] = model.predict(X_val_scaled_np)

    # ---------------------------
    # 训练Stacking元模型
    # ---------------------------
    meta_linear = LinearRegression()
    meta_linear.fit(train_stack, y_train.values.ravel())

    meta_tree = DecisionTreeRegressor(random_state=42)
    meta_tree.fit(train_stack, y_train.values.ravel())

    # ---------------------------
    # 计算评估指标
    # ---------------------------
    metrics = {}
    for model_name in models:
        model = model_objects[model_name]
        train_pred = model.predict(X_train_scaled_np)
        test_pred = model.predict(X_test_scaled_np)
        val_pred = model.predict(X_val_scaled_np)
        metrics[model_name] = {
            'Train': compute_metrics(y_train, train_pred),
            'Test': compute_metrics(y_test, test_pred),
            'Validation': compute_metrics(y_val, val_pred)
        }

    stacking_metrics = {
        'Linear': {
            'Test': compute_metrics(y_test, meta_linear.predict(test_stack)),
            'Validation': compute_metrics(y_val, meta_linear.predict(val_stack))
        },
        'Tree': {
            'Test': compute_metrics(y_test, meta_tree.predict(test_stack)),
            'Validation': compute_metrics(y_val, meta_tree.predict(val_stack))
        }
    }

    # ---------------------------
    # 保存所有模型权重
    # ---------------------------
    for model_name in model_objects:
        filename = f"Cluster_0_data_pop50_{model_name}_model.pkl"  # 修改文件名
        with open(filename, "wb") as f:
            pickle.dump(model_objects[model_name], f)
        print(f"已保存基模型：{filename}")

    stacking_linear_file = f"Cluster_0_data_pop50_stacking_linear_model.pkl"  # 修改文件名
    with open(stacking_linear_file, "wb") as f:
        pickle.dump(meta_linear, f)
    print(f"已保存集成模型：{stacking_linear_file}")

    stacking_tree_file = f"Cluster_0_data_pop50_stacking_tree_model.pkl"  # 修改文件名
    with open(stacking_tree_file, "wb") as f:
        pickle.dump(meta_tree, f)
    print(f"已保存集成模型：{stacking_tree_file}")

    # 保存归一化对象
    scaler_filename = f"Cluster_0_data_pop50_scaler_X.pkl"  # 修改文件名
    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler_X, f)
    print(f"已保存归一化对象：{scaler_filename}")

    # ---------------------------
    # 保存结果到Excel
    # ---------------------------
    excel_file_name = f"{cluster_name}_pop50_Optimized_Results.xlsx"  # 自动使用Cluster_0_data

    # 基模型结果
    base_results = []
    for model_name in metrics:
        for dataset in ['Train', 'Test', 'Validation']:
            r2, vaf, rmsle, kge = metrics[model_name][dataset]
            base_results.append({
                'Model': model_name,
                'Dataset': dataset,
                'R2': r2,
                'VAF': vaf,
                'RMSLE': rmsle,
                'KGE': kge
            })

    # 集成模型结果
    stacking_results = []
    for method in ['Linear', 'Tree']:
        for dataset in ['Test', 'Validation']:
            r2, vaf, rmsle, kge = stacking_metrics[method][dataset]
            stacking_results.append({
                'Model': f'Stacking-{method}',
                'Dataset': dataset,
                'R2': r2,
                'VAF': vaf,
                'RMSLE': rmsle,
                'KGE': kge
            })

    results_df = pd.DataFrame(base_results + stacking_results)
    params_df = pd.DataFrame({
        'Model': ['XGB', 'GBDT', 'AdaBoost', 'RF'],
        'Hyperparameters': [
            "n_estimators=282, learning_rate=0.22624, max_depth=3, reg_alpha=0.51622, reg_lambda=0.99947",
            "n_estimators=983, learning_rate=0.12682, max_depth=6, subsample=0.50004, min_samples_leaf=5",
            "n_estimators=946, learning_rate=0.78341, base_max_depth=3, base_min_samples_leaf=1",
            "n_estimators=21, max_depth=10, max_features=0.43295"
        ]
    })

    with pd.ExcelWriter(excel_file_name) as writer:
        results_df.to_excel(writer, sheet_name="Model Metrics", index=False)
        params_df.to_excel(writer, sheet_name="Hyperparameters", index=False)

    print(f"\n结果已保存至：{excel_file_name}")
    print("任务已完成。")