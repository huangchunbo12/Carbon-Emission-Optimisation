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
# 数据加载及预处理（训练、测试、验证按指定年份划分）
# ---------------------------
file_path = r'Cluster_0_data.xlsx'
data = pd.read_excel(file_path)
cluster_name = os.path.splitext(os.path.basename(file_path))[0]

X_columns = ['G', 'SI', 'ES', 'EI', 'DMSP']
y_column = 'CO2 Emissions'

# 训练集：2000-2003, 2007-2015 和 2020-2022（示例中多个区间合并）
train_data = data[((data['年份'] >= 2000) & (data['年份'] <= 2003)) | 
                  ((data['年份'] >= 2020) & (data['年份'] <= 2022)) | 
                  ((data['年份'] >= 2007) & (data['年份'] <= 2015))]
# 测试集：2016-2019
test_data  = data[(data['年份'] >= 2016) & (data['年份'] <= 2019)]
# 验证集：2004-2006
val_data   = data[(data['年份'] >= 2004) & (data['年份'] <= 2006)]

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

# 目标值使用原始数值
y_train_np = y_train.values.reshape(-1, 1)
y_test_np  = y_test.values.reshape(-1, 1)
y_val_np   = y_val.values.reshape(-1, 1)

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
# ALA超参数优化算法相关函数
# ---------------------------
def initialization(N, dim, ub, lb):
    return np.random.rand(N, dim) * (ub - lb) + lb

def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(np.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step

# ---------------------------
# 目标函数：XGB超参数优化
# 超参数顺序：[n_estimators, learning_rate, max_depth, reg_alpha, reg_lambda]
# ---------------------------
def objective_function_xgb(params, gpu_id=0):
    n_estimators = int(params[0])
    learning_rate = params[1]
    max_depth = int(params[2])
    reg_alpha = params[3]
    reg_lambda = params[4]
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate,
        max_depth=max_depth,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective='reg:squarederror', 
        random_state=42, 
        verbosity=0
    )
    model.fit(X_train_scaled_np, y_train.values.ravel())
    y_val_pred = model.predict(X_val_scaled_np)
    rmse_val = np.sqrt(mean_squared_error(y_val.values.ravel(), y_val_pred))
    return rmse_val

# ---------------------------
# 目标函数：GBDT超参数优化
# 超参数顺序：[n_estimators, learning_rate, max_depth, subsample, min_samples_leaf]
# ---------------------------
def objective_function_gbdt(params, gpu_id=0):
    n_estimators = int(params[0])
    learning_rate = params[1]
    max_depth = int(params[2])
    subsample = params[3]
    min_samples_leaf = int(params[4])
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate,
        max_depth=max_depth, 
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train_scaled_np, y_train.values.ravel())
    y_val_pred = model.predict(X_val_scaled_np)
    rmse_val = np.sqrt(mean_squared_error(y_val.values.ravel(), y_val_pred))
    return rmse_val

# ---------------------------
# 目标函数：AdaBoost超参数优化
# 超参数顺序：[n_estimators, learning_rate, base_max_depth, base_min_samples_leaf]
# ---------------------------
def objective_function_adaboost(params, gpu_id=0):
    n_estimators = int(params[0])
    learning_rate = params[1]
    base_max_depth = int(params[2])
    base_min_samples_leaf = int(params[3])
    base_est = DecisionTreeRegressor(max_depth=base_max_depth, min_samples_leaf=base_min_samples_leaf)
    model = AdaBoostRegressor(
        estimator=base_est,
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        random_state=42
    )
    model.fit(X_train_scaled_np, y_train.values.ravel())
    y_val_pred = model.predict(X_val_scaled_np)
    rmse_val = np.sqrt(mean_squared_error(y_val.values.ravel(), y_val_pred))
    return rmse_val

# ---------------------------
# 目标函数：RF超参数优化
# 超参数顺序：[n_estimators, max_depth, max_features]
# ---------------------------
def objective_function_rf(params, gpu_id=0):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    max_features = params[2]  # float 类型
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        max_features=max_features, 
        random_state=42
    )
    model.fit(X_train_scaled_np, y_train.values.ravel())
    y_val_pred = model.predict(X_val_scaled_np)
    rmse_val = np.sqrt(mean_squared_error(y_val.values.ravel(), y_val_pred))
    return rmse_val

# ---------------------------
# ALA算法：并行计算候选解适应度
# ---------------------------
def ALA(N, Max_iter, lb, ub, dim, fobj):
    X = initialization(N, dim, ub, lb)
    Position = np.zeros(dim)
    Score = np.inf
    fitness = np.zeros(N)
    Convergence_curve = []  # 保存每次迭代的最佳RMSE
    vec_flag = [1, -1]
    
    pbar = tqdm(total=Max_iter, desc="ALA Optimization Progress")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        # 初始种群适应度评估
        futures = []
        for i in range(N):
            gpu_id = i % 1
            futures.append(executor.submit(fobj, X[i, :], gpu_id))
        results = [future.result() for future in futures]
        for i in range(N):
            fitness[i] = results[i]
            if fitness[i] < Score:
                Position = X[i, :].copy()
                Score = fitness[i]
                
        Iter = 1
        while Iter <= Max_iter:
            new_futures = []
            new_candidates = []
            for i in range(N):
                RB = np.random.randn(dim)
                F = random.choice(vec_flag)
                theta = 2 * np.arctan(1 - Iter / Max_iter)
                E = 2 * np.log(1 / random.random()) * theta
                if E > 1:
                    if random.random() < 0.3:
                        r1 = 2 * np.random.rand(dim) - 1
                        Xnew = Position + F * RB * (r1 * (Position - X[i, :]) + (1 - r1) * (X[i, :] - X[random.randint(0, N - 1), :]))
                    else:
                        r2 = random.random() * (1 + np.sin(0.5 * Iter))
                        Xnew = X[i, :] + F * r2 * (Position - X[random.randint(0, N - 1), :])
                else:
                    if random.random() < 0.5:
                        radius = np.sqrt(np.sum((Position - X[i, :]) ** 2))
                        r3 = random.random()
                        spiral = radius * (np.sin(2 * np.pi * r3) + np.cos(2 * np.pi * r3))
                        Xnew = Position + F * X[i, :] * spiral * random.random()
                    else:
                        G = 2 * (np.sign(random.random() - 0.5)) * (1 - Iter / Max_iter)
                        Xnew = Position + F * G * Levy(dim) * (Position - X[i, :])
                Xnew = np.clip(Xnew, lb, ub)
                new_candidates.append(Xnew)
                gpu_id = i % 1
                new_futures.append(executor.submit(fobj, Xnew, gpu_id))
            new_results = [future.result() for future in new_futures]
            for i in range(N):
                new_fit = new_results[i]
                if new_fit < fitness[i]:
                    X[i, :] = new_candidates[i]
                    fitness[i] = new_fit
                if fitness[i] < Score:
                    Position = X[i, :].copy()
                    Score = fitness[i]
            Convergence_curve.append(Score)
            print(f"Iteration {Iter}/{Max_iter} - Best Score (RMSE): {Score}")
            pbar.update(1)
            Iter += 1
        pbar.close()
    return Score, Position, Convergence_curve

# ---------------------------
# 主程序：对不同种群规模分别运行
# ---------------------------
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    # 定义种群规模列表
    population_list = [50, 100, 150, 200, 250, 300]  # 可根据需要调整
    Max_iter = 200  # 每个模型的优化迭代次数
    
    # 固定搜索边界及维度设置
    # XGB：参数维度5：[n_estimators, learning_rate, max_depth, reg_alpha, reg_lambda]
    lb_xgb = np.array([1, 0.001, 1, 0, 0])
    ub_xgb = np.array([1000, 0.3, 10, 1, 1])
    dim_xgb = 5

    # GBDT：参数维度5：[n_estimators, learning_rate, max_depth, subsample, min_samples_leaf]
    lb_gbdt = np.array([1, 0.001, 1, 0.5, 1])
    ub_gbdt = np.array([1000, 0.3, 10, 1, 20])
    dim_gbdt = 5

    # AdaBoost：参数维度4：[n_estimators, learning_rate, base_max_depth, base_min_samples_leaf]
    lb_ada = np.array([1, 0.001, 1, 1])
    ub_ada = np.array([1000, 1, 10, 20])
    dim_ada = 4

    # RF：参数维度3：[n_estimators, max_depth, max_features]
    lb_rf = np.array([1, 1, 0.1])
    ub_rf = np.array([1000, 50, 1])
    dim_rf = 3

    # 用于存放交叉验证的指标
    cv_metrics = {}

    for pop_size in population_list:
        print(f"\n=========== 当前种群规模：{pop_size} ===========")
        # ---------------------------
        # 1. XGB超参数优化
        # ---------------------------
        print("开始使用ALA算法优化XGB超参数……")
        best_score_xgb, best_position_xgb, conv_curve_xgb = ALA(pop_size, Max_iter, lb_xgb, ub_xgb, dim_xgb, objective_function_xgb)
        best_n_estimators_xgb = int(best_position_xgb[0])
        best_learning_rate_xgb = best_position_xgb[1]
        best_max_depth_xgb = int(best_position_xgb[2])
        best_reg_alpha_xgb = best_position_xgb[3]
        best_reg_lambda_xgb = best_position_xgb[4]
        print(f"最优XGB超参数：n_estimators={best_n_estimators_xgb}, learning_rate={best_learning_rate_xgb:.5f}, max_depth={best_max_depth_xgb}, reg_alpha={best_reg_alpha_xgb:.5f}, reg_lambda={best_reg_lambda_xgb:.5f}")
        
        model_xgb = xgb.XGBRegressor(
            n_estimators=best_n_estimators_xgb, 
            learning_rate=best_learning_rate_xgb,
            max_depth=best_max_depth_xgb,
            reg_alpha=best_reg_alpha_xgb,
            reg_lambda=best_reg_lambda_xgb,
            objective='reg:squarederror', 
            random_state=42, 
            verbosity=0
        )
        model_xgb.fit(X_train_scaled_np, y_train.values.ravel())
        train_pred_xgb = model_xgb.predict(X_train_scaled_np)
        test_pred_xgb = model_xgb.predict(X_test_scaled_np)
        val_pred_xgb = model_xgb.predict(X_val_scaled_np)
        
        # ---------------------------
        # 2. GBDT超参数优化
        # ---------------------------
        print("\n开始使用ALA算法优化GBDT超参数……")
        best_score_gbdt, best_position_gbdt, conv_curve_gbdt = ALA(pop_size, Max_iter, lb_gbdt, ub_gbdt, dim_gbdt, objective_function_gbdt)
        best_n_estimators_gbdt = int(best_position_gbdt[0])
        best_learning_rate_gbdt = best_position_gbdt[1]
        best_max_depth_gbdt = int(best_position_gbdt[2])
        best_subsample = best_position_gbdt[3]
        best_min_samples_leaf = int(best_position_gbdt[4])
        print(f"最优GBDT超参数：n_estimators={best_n_estimators_gbdt}, learning_rate={best_learning_rate_gbdt:.5f}, max_depth={best_max_depth_gbdt}, subsample={best_subsample:.5f}, min_samples_leaf={best_min_samples_leaf}")
        
        model_gbdt = GradientBoostingRegressor(
            n_estimators=best_n_estimators_gbdt, 
            learning_rate=best_learning_rate_gbdt,
            max_depth=best_max_depth_gbdt, 
            subsample=best_subsample,
            min_samples_leaf=best_min_samples_leaf,
            random_state=42
        )
        model_gbdt.fit(X_train_scaled_np, y_train.values.ravel())
        train_pred_gbdt = model_gbdt.predict(X_train_scaled_np)
        test_pred_gbdt = model_gbdt.predict(X_test_scaled_np)
        val_pred_gbdt = model_gbdt.predict(X_val_scaled_np)
        
        # ---------------------------
        # 3. AdaBoost超参数优化
        # ---------------------------
        print("\n开始使用ALA算法优化AdaBoost超参数……")
        best_score_ada, best_position_ada, conv_curve_ada = ALA(pop_size, Max_iter, lb_ada, ub_ada, dim_ada, objective_function_adaboost)
        best_n_estimators_ada = int(best_position_ada[0])
        best_learning_rate_ada = best_position_ada[1]
        best_base_max_depth = int(best_position_ada[2])
        best_base_min_samples_leaf = int(best_position_ada[3])
        print(f"最优AdaBoost超参数：n_estimators={best_n_estimators_ada}, learning_rate={best_learning_rate_ada:.5f}, base_max_depth={best_base_max_depth}, base_min_samples_leaf={best_base_min_samples_leaf}")
        
        base_est = DecisionTreeRegressor(max_depth=best_base_max_depth, min_samples_leaf=best_base_min_samples_leaf)
        model_ada = AdaBoostRegressor(
            estimator=base_est,
            n_estimators=best_n_estimators_ada, 
            learning_rate=best_learning_rate_ada, 
            random_state=42
        )
        model_ada.fit(X_train_scaled_np, y_train.values.ravel())
        train_pred_ada = model_ada.predict(X_train_scaled_np)
        test_pred_ada = model_ada.predict(X_test_scaled_np)
        val_pred_ada = model_ada.predict(X_val_scaled_np)
        
        # ---------------------------
        # 4. RF超参数优化
        # ---------------------------
        print("\n开始使用ALA算法优化RF超参数……")
        best_score_rf, best_position_rf, conv_curve_rf = ALA(pop_size, Max_iter, lb_rf, ub_rf, dim_rf, objective_function_rf)
        best_n_estimators_rf = int(best_position_rf[0])
        best_max_depth_rf = int(best_position_rf[1])
        best_max_features = best_position_rf[2]
        print(f"最优RF超参数：n_estimators={best_n_estimators_rf}, max_depth={best_max_depth_rf}, max_features={best_max_features:.5f}")
        
        model_rf = RandomForestRegressor(
            n_estimators=best_n_estimators_rf, 
            max_depth=best_max_depth_rf, 
            max_features=best_max_features, 
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
                n_estimators=best_n_estimators_xgb,
                learning_rate=best_learning_rate_xgb,
                max_depth=best_max_depth_xgb,
                reg_alpha=best_reg_alpha_xgb,
                reg_lambda=best_reg_lambda_xgb,
                objective='reg:squarederror',
                random_state=42,
                verbosity=0
            )
            m.fit(X_cv_train, y_cv_train)
            preds = m.predict(X_cv_valid)
            cv_results['XGB'].append(compute_metrics(y_cv_valid, preds))
            
            # GBDT
            m = GradientBoostingRegressor(
                n_estimators=best_n_estimators_gbdt,
                learning_rate=best_learning_rate_gbdt,
                max_depth=best_max_depth_gbdt,
                subsample=best_subsample,
                min_samples_leaf=best_min_samples_leaf,
                random_state=42
            )
            m.fit(X_cv_train, y_cv_train)
            preds = m.predict(X_cv_valid)
            cv_results['GBDT'].append(compute_metrics(y_cv_valid, preds))
            
            # AdaBoost
            base_est_cv = DecisionTreeRegressor(max_depth=best_base_max_depth, min_samples_leaf=best_base_min_samples_leaf)
            m = AdaBoostRegressor(
                estimator=base_est_cv,
                n_estimators=best_n_estimators_ada,
                learning_rate=best_learning_rate_ada,
                random_state=42
            )
            m.fit(X_cv_train, y_cv_train)
            preds = m.predict(X_cv_valid)
            cv_results['AdaBoost'].append(compute_metrics(y_cv_valid, preds))
            
            # RF
            m = RandomForestRegressor(
                n_estimators=best_n_estimators_rf,
                max_depth=best_max_depth_rf,
                max_features=best_max_features,
                random_state=42
            )
            m.fit(X_cv_train, y_cv_train)
            preds = m.predict(X_cv_valid)
            cv_results['RF'].append(compute_metrics(y_cv_valid, preds))
        
        cv_metrics_mean = {}
        for model_name in cv_results:
            cv_array = np.array(cv_results[model_name])
            cv_metrics_mean[model_name] = cv_array.mean(axis=0)
        
        # ---------------------------
        # 6. Stacking集成
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
            
            model_linear = LinearRegression()
            model_linear.fit(X_meta_train, y_meta_train)
            meta_train_linear[valid_idx] = model_linear.predict(X_meta_valid)
            meta_test_preds_linear.append(model_linear.predict(test_stack))
            meta_val_preds_linear.append(model_linear.predict(val_stack))
            
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
        excel_file_name = f"{cluster_name}_pop{pop_size}_Final_Results.xlsx"
        
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
            hyperparams_df = pd.DataFrame({
                'Model': ['XGB', 'GBDT', 'AdaBoost', 'RF'],
                'Hyperparameters': [
                    f"n_estimators={best_n_estimators_xgb}, learning_rate={best_learning_rate_xgb:.5f}, max_depth={best_max_depth_xgb}, reg_alpha={best_reg_alpha_xgb:.5f}, reg_lambda={best_reg_lambda_xgb:.5f}",
                    f"n_estimators={best_n_estimators_gbdt}, learning_rate={best_learning_rate_gbdt:.5f}, max_depth={best_max_depth_gbdt}, subsample={best_subsample:.5f}, min_samples_leaf={best_min_samples_leaf}",
                    f"n_estimators={best_n_estimators_ada}, learning_rate={best_learning_rate_ada:.5f}, base_max_depth={best_base_max_depth}, base_min_samples_leaf={best_base_min_samples_leaf}",
                    f"n_estimators={best_n_estimators_rf}, max_depth={best_max_depth_rf}, max_features={best_max_features:.5f}"
                ]
            })
            hyperparams_df.to_excel(writer, sheet_name="Hyperparameters", index=False)
            
            conv_df = pd.DataFrame({
                'Iteration': np.arange(1, len(conv_curve_xgb)+1),
                'XGB_RMSE': conv_curve_xgb
            })
            conv_df.to_excel(writer, sheet_name="XGB ALA Curve", index=False)
            conv_df = pd.DataFrame({
                'Iteration': np.arange(1, len(conv_curve_gbdt)+1),
                'GBDT_RMSE': conv_curve_gbdt
            })
            conv_df.to_excel(writer, sheet_name="GBDT ALA Curve", index=False)
            conv_df = pd.DataFrame({
                'Iteration': np.arange(1, len(conv_curve_ada)+1),
                'AdaBoost_RMSE': conv_curve_ada
            })
            conv_df.to_excel(writer, sheet_name="AdaBoost ALA Curve", index=False)
            conv_df = pd.DataFrame({
                'Iteration': np.arange(1, len(conv_curve_rf)+1),
                'RF_RMSE': conv_curve_rf
            })
            conv_df.to_excel(writer, sheet_name="RF ALA Curve", index=False)
        
        print(f"结果保存为 '{excel_file_name}'")
        
        # ---------------------------
        # 9. 保存Stacking模型权重（保存最终训练的元模型：线性和决策树）
        # ---------------------------
        meta_linear_final = LinearRegression().fit(train_stack, y_train.values.ravel())
        meta_tree_final = DecisionTreeRegressor(random_state=42).fit(train_stack, y_train.values.ravel())
        
        stacking_linear_file = f"{cluster_name}_pop{pop_size}_stacking_linear_model.pkl"
        with open(stacking_linear_file, "wb") as f:
            pickle.dump(meta_linear_final, f)
        stacking_tree_file = f"{cluster_name}_pop{pop_size}_stacking_tree_model.pkl"
        with open(stacking_tree_file, "wb") as f:
            pickle.dump(meta_tree_final, f)
        
        print(f"保存Stacking模型权重：'{stacking_linear_file}' 和 '{stacking_tree_file}'")
    
    print("\n所有种群规模的任务均已完成。")
