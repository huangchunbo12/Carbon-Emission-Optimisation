import numpy as np
import pandas as pd
import math, random
import concurrent.futures
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------
# 公共函数
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
# 随机森林目标函数：传入超参数，训练随机森林并在验证集上计算RMSE
# 超参数顺序：[n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, min_impurity_decrease]
# ---------------------------
def objective_function_rf(params, X_train, y_train, X_val, y_val):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_leaf = int(params[2])
    min_samples_split = int(params[3])
    max_features = params[4]
    min_impurity_decrease = params[5]

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  max_features=max_features,
                                  min_impurity_decrease=min_impurity_decrease,
                                  random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

# ---------------------------
# 优化算法（保持不变）
# ---------------------------
def WOA(fobj, pop_size, dim, lb, ub, max_iter, X_train, y_train, X_val, y_val):
    X = initialization(pop_size, dim, ub, lb)
    fitness = np.array([fobj(ind) for ind in X])
    best_idx = np.argmin(fitness)
    best_sol = X[best_idx].copy()
    best_fit = fitness[best_idx]
    convergence = []
    a_linear = 2
    for t in tqdm(range(max_iter), desc="WOA progress", leave=True):
        a = a_linear - t * (a_linear / max_iter)
        for i in range(pop_size):
            r1 = random.random()
            r2 = random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = random.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * best_sol - X[i])
                    X[i] = best_sol - A * D
                else:
                    rand_idx = random.randint(0, pop_size - 1)
                    rand_sol = X[rand_idx]
                    D = abs(C * rand_sol - X[i])
                    X[i] = rand_sol - A * D
            else:
                distance = abs(best_sol - X[i])
                b = 1
                l = random.uniform(-1, 1)
                X[i] = distance * math.exp(b * l) * math.cos(2 * math.pi * l) + best_sol
            X[i] = np.clip(X[i], lb, ub)
            fnew = fobj(X[i])
            if fnew < fitness[i]:
                fitness[i] = fnew
            if fnew < best_fit:
                best_fit = fnew
                best_sol = X[i].copy()
        convergence.append(best_fit)
    return best_sol, best_fit, convergence

def SSA(fobj, pop_size, dim, lb, ub, max_iter, X_train, y_train, X_val, y_val):
    X = initialization(pop_size, dim, ub, lb)
    fitness = np.array([fobj(ind) for ind in X])
    best_idx = np.argmin(fitness)
    best_sol = X[best_idx].copy()
    best_fit = fitness[best_idx]
    convergence = []
    for t in tqdm(range(max_iter), desc="SSA progress", leave=True):
        c1 = 2 * math.exp(-((4 * t / max_iter) ** 2))
        for i in range(pop_size):
            if i == 0:
                for j in range(dim):
                    c2 = random.random()
                    c3 = random.random()
                    if c3 < 0.5:
                        X[i, j] = best_sol[j] + c1 * ((ub[j] - lb[j]) * c2 + lb[j])
                    else:
                        X[i, j] = best_sol[j] - c1 * ((ub[j] - lb[j]) * c2 + lb[j])
            else:
                X[i] = (X[i] + X[i - 1]) / 2
            X[i] = np.clip(X[i], lb, ub)
            fnew = fobj(X[i])
            if fnew < fitness[i]:
                fitness[i] = fnew
            if fnew < best_fit:
                best_fit = fnew
                best_sol = X[i].copy()
        convergence.append(best_fit)
    return best_sol, best_fit, convergence

def GWO(fobj, pop_size, dim, lb, ub, max_iter, X_train, y_train, X_val, y_val):
    X = initialization(pop_size, dim, ub, lb)
    fitness = np.array([fobj(ind) for ind in X])
    sorted_idx = np.argsort(fitness)
    alpha = X[sorted_idx[0]].copy()
    beta = X[sorted_idx[1]].copy()
    delta = X[sorted_idx[2]].copy()
    best_fit = fitness[sorted_idx[0]]
    convergence = []
    for t in tqdm(range(max_iter), desc="GWO progress", leave=True):
        a = 2 - t * (2 / max_iter)
        for i in range(pop_size):
            for j in range(dim):
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - X[i, j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - X[i, j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - X[i, j])
                X3 = delta[j] - A3 * D_delta

                X[i, j] = (X1 + X2 + X3) / 3
            X[i] = np.clip(X[i], lb, ub)
            fitness[i] = fobj(X[i])
        sorted_idx = np.argsort(fitness)
        alpha = X[sorted_idx[0]].copy()
        beta = X[sorted_idx[1]].copy()
        delta = X[sorted_idx[2]].copy()
        best_fit = fitness[sorted_idx[0]]
        convergence.append(best_fit)
    return alpha, best_fit, convergence

def GA(fobj, pop_size, dim, lb, ub, max_iter, X_train, y_train, X_val, y_val):
    X = initialization(pop_size, dim, ub, lb)
    fitness = np.array([fobj(ind) for ind in X])
    convergence = []
    for t in tqdm(range(max_iter), desc="GA progress", leave=True):
        new_population = []
        for i in range(pop_size):
            i1, i2 = random.randint(0, pop_size - 1), random.randint(0, pop_size - 1)
            if fitness[i1] < fitness[i2]:
                new_population.append(X[i1].copy())
            else:
                new_population.append(X[i2].copy())
        new_population = np.array(new_population)
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size and random.random() < 0.8:
                point = random.randint(1, dim - 1)
                new_population[i, :point], new_population[i + 1, :point] = new_population[i + 1, :point].copy(), new_population[i, :point].copy()
        for i in range(pop_size):
            if random.random() < 0.1:
                new_population[i] = np.random.uniform(lb, ub, dim)
        X = new_population.copy()
        fitness = np.array([fobj(ind) for ind in X])
        best_fit = np.min(fitness)
        convergence.append(best_fit)
    best_idx = np.argmin(fitness)
    return X[best_idx].copy(), fitness[best_idx], convergence

def PSO(fobj, pop_size, dim, lb, ub, max_iter, X_train, y_train, X_val, y_val):
    X = initialization(pop_size, dim, ub, lb)
    V = np.zeros((pop_size, dim))
    fitness = np.array([fobj(ind) for ind in X])
    pbest = X.copy()
    pbest_fit = fitness.copy()
    gbest = X[np.argmin(fitness)].copy()
    gbest_fit = np.min(fitness)
    convergence = []
    w = 0.5; c1 = 1; c2 = 2
    for t in tqdm(range(max_iter), desc="PSO progress", leave=True):
        for i in range(pop_size):
            V[i] = w * V[i] + c1 * random.random() * (pbest[i] - X[i]) + c2 * random.random() * (gbest - X[i])
            X[i] = X[i] + V[i]
            X[i] = np.clip(X[i], lb, ub)
            fit = fobj(X[i])
            if fit < pbest_fit[i]:
                pbest[i] = X[i].copy()
                pbest_fit[i] = fit
            if fit < gbest_fit:
                gbest = X[i].copy()
                gbest_fit = fit
        convergence.append(gbest_fit)
    return gbest, gbest_fit, convergence

def ALA(fobj, pop_size, dim, lb, ub, max_iter, X_train, y_train, X_val, y_val):
    X = initialization(pop_size, dim, ub, lb)
    Position = np.zeros(dim)
    Score = np.inf
    fitness = np.zeros(pop_size)
    convergence = []
    vec_flag = [1, -1]
    for i in range(pop_size):
        fitness[i] = fobj(X[i])
        if fitness[i] < Score:
            Score = fitness[i]
            Position = X[i].copy()
    for Iter in tqdm(range(1, max_iter + 1), desc="ALA progress", leave=True):
        for i in range(pop_size):
            RB = np.random.randn(dim)
            F = random.choice(vec_flag)
            theta = 2 * np.arctan(1 - Iter / max_iter)
            E = 2 * np.log(1 / random.random()) * theta
            if E > 1:
                if random.random() < 0.3:
                    r1 = 2 * np.random.rand(dim) - 1
                    Xnew = Position + F * RB * (r1 * (Position - X[i]) + (1 - r1) * (X[i] - X[random.randint(0, pop_size - 1)]))
                else:
                    r2 = random.random() * (1 + np.sin(0.5 * Iter))
                    Xnew = X[i] + F * r2 * (Position - X[random.randint(0, pop_size - 1)])
            else:
                if random.random() < 0.5:
                    radius = np.sqrt(np.sum((Position - X[i]) ** 2))
                    r3 = random.random()
                    spiral = radius * (np.sin(2 * np.pi * r3) + np.cos(2 * np.pi * r3))
                    Xnew = Position + F * X[i] * spiral * random.random()
                else:
                    G = 2 * (np.sign(random.random() - 0.5)) * (1 - Iter / max_iter)
                    Xnew = Position + F * G * Levy(dim) * (Position - X[i])
            Xnew = np.clip(Xnew, lb, ub)
            new_fit = fobj(Xnew)
            if new_fit < fitness[i]:
                X[i] = Xnew
                fitness[i] = new_fit
            if fitness[i] < Score:
                Score = fitness[i]
                Position = X[i].copy()
        convergence.append(Score)
    return Position, Score, convergence

# ---------------------------
# 数据读取、预处理及划分（8:1:1比例：训练80%，验证10%，测试10%）
# ---------------------------
def load_and_split_data(path, feature_cols, target_col):
    df = pd.read_csv(path)
    y = df.iloc[:, target_col].values
    X = df.drop(df.columns[target_col], axis=1)
    X = pd.get_dummies(X)
    X = X.values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

# ---------------------------
# 对单个数据集和优化算法进行处理
# ---------------------------
def process_dataset_optimizer(dname, info, opt_name, optimizer, pop_size, dim, lb, ub, max_iter):
    print(f"\nProcessing dataset: {dname}, Using optimizer: {opt_name}", flush=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data(info['path'], info['feature_idx'], info['target_idx'])
    fobj = lambda params: objective_function_rf(params, X_train, y_train, X_val, y_val)
    best_params, best_rmse, conv = optimizer(fobj, pop_size, dim, lb, ub, max_iter, X_train, y_train, X_val, y_val)
    print(f"{dname} - {opt_name} best RMSE: {best_rmse}, best parameters: {best_params}", flush=True)
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)
    model = RandomForestRegressor(n_estimators=int(best_params[0]),
                                  max_depth=int(best_params[1]),
                                  min_samples_leaf=int(best_params[2]),
                                  min_samples_split=int(best_params[3]),
                                  max_features=best_params[4],
                                  min_impurity_decrease=best_params[5],
                                  random_state=24)
    model.fit(X_train_val, y_train_val)
    y_pred = model.predict(X_test)
    R2 = r2_score(y_test, y_pred)
    filename = f'优化结果_{dname}_{opt_name}.xlsx'
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            pd.DataFrame([{'Dataset': dname, 'Optimizer': opt_name, 'Test_R2': R2}]).to_excel(writer, sheet_name='Test_R2', index=False)
            params_df = pd.DataFrame({
                'Parameter': ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'max_features', 'min_impurity_decrease', 'RMSE'],
                'Value': [best_params[0], best_params[1], best_params[2], best_params[3], best_params[4], best_params[5], best_rmse]
            })
            params_df.to_excel(writer, sheet_name='Best_Parameters', index=False)
            conv_df = pd.DataFrame({
                'Iteration': list(range(1, len(conv) + 1)),
                'RMSE': conv
            })
            conv_df.to_excel(writer, sheet_name='Convergence_Curve', index=False)
        print(f"{dname} - {opt_name}: Excel file '{filename}' saved.", flush=True)
    except Exception as e:
        print(f"Error saving {filename}: {e}", flush=True)
    return {'Dataset': dname, 'Optimizer': opt_name, 'Test_R2': R2}, {dname: {opt_name: conv}}

# ---------------------------
# 主流程：参数范围调整
# ---------------------------
def main():
    dim = 6
    lb = np.array([1, 1, 1, 2, 0.1, 0.0])
    ub = np.array([200, 20, 20, 10, 1.0, 0.1])
    pop_size = 50
    max_iter = 500

    optimizers = {
        'WOA': WOA,
        'SSA': SSA,
        'GWO': GWO,
        'GA': GA,
        'PSO': PSO,
        'ALA': ALA
    }

    datasets = {
        '医疗保险': {'path': '医疗保险/Train_Data.csv', 'feature_idx': list(range(6)), 'target_idx': 6},
        '房价': {'path': '房价/house_price_regression_dataset.csv', 'feature_idx': list(range(7)), 'target_idx': 7},
        '成绩': {'path': '成绩/student_performance_dataset.csv', 'feature_idx': list(range(8)), 'target_idx': 8}
    }

    results_R2 = []
    convergence_data = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        futures = []
        for dname, info in datasets.items():
            for opt_name, optimizer in optimizers.items():
                future = executor.submit(process_dataset_optimizer, dname, info, opt_name, optimizer,
                                         pop_size, dim, lb, ub, max_iter)
                futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            result_R2, conv_data = future.result()
            results_R2.append(result_R2)
            for d, opt_conv in conv_data.items():
                if d not in convergence_data:
                    convergence_data[d] = {}
                convergence_data[d].update(opt_conv)

    overall_filename = '总体优化结果.xlsx'
    try:
        with pd.ExcelWriter(overall_filename, engine='openpyxl') as writer:
            df_R2 = pd.DataFrame(results_R2)
            df_R2.to_excel(writer, sheet_name='R2_Results', index=False)
            for dname, opt_data in convergence_data.items():
                df_conv = pd.DataFrame(opt_data)
                df_conv.to_excel(writer, sheet_name=f'{dname}_RMSE_Convergence', index=False)
        print(f"\nOverall results saved to '{overall_filename}'", flush=True)
    except Exception as e:
        print(f"Error saving overall results: {e}", flush=True)

if __name__ == '__main__':
    main()
    