import pandas as pd
import numpy as np
import pickle
import os
import sys
import hashlib
from sklearn.preprocessing import MinMaxScaler

# Fix numpy compatibility issue
import numpy.core as np_core
sys.modules['numpy._core'] = np_core

# Set a random seed for reproducibility
np.random.seed(42)

# Define the 5 indicators
indicators = ['G', 'SI', 'ES', 'EI', 'DMSP']

# Define model configurations for each cluster
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

# Output directory
output_dir = "prediction_outputs"
os.makedirs(output_dir, exist_ok=True)

# Iterate through each cluster
for cluster_num in cluster_info:
    info = cluster_info[cluster_num]
    cluster_name = info["cluster_name"]
    base_pop = info["base_model_pop"]
    scaler_file = info["scaler_file"]

    print(f"\n{'='*40}")
    print(f"Processing {cluster_name} ...")

    # ================ Load MinMaxScaler =================
    if os.path.exists(scaler_file):
        try:
            with open(scaler_file, "rb") as f:
                scaler_X = pickle.load(f)
            print(f"Successfully loaded scaler: {scaler_file}")
        except Exception as e:
            print(f"Failed to load scaler: {str(e)}")
            scaler_X = MinMaxScaler()
    else:
        print(f"Scaler file does not exist: {scaler_file}")
        scaler_X = MinMaxScaler()

    # ================ Load base models =================
    base_models = {}
    base_model_types = ['XGB', 'GBDT', 'AdaBoost', 'RF']
    for model_type in base_model_types:
        model_file = f"Cluster_{cluster_num}_data_{base_pop}_{model_type}_model.pkl"
        try:
            with open(model_file, "rb") as f:
                base_models[model_type] = pickle.load(f)
            print(f"Successfully loaded base model: {model_type}")
        except Exception as e:
            print(f"Failed to load base model {model_type}: {str(e)}")
            continue

    if len(base_models) != 4:
        print(f"Incomplete base models loaded, skipping {cluster_name}")
        continue

    stacking_model_file = info["stacking_model_file"]
    if os.path.exists(stacking_model_file):
        try:
            with open(stacking_model_file, "rb") as f:
                stacking_model = pickle.load(f)
            print(f"Successfully loaded stacking model: {stacking_model_file}")
        except Exception as e:
            print(f"Failed to load stacking model: {str(e)}")
            continue
    else:
        print(f"Stacking model file does not exist: {stacking_model_file}")
        continue

    # ================ Process simulation scenarios =================
    for scenario, sim_file in info["sim_files"].items():
        print(f"\nProcessing scenario {scenario} ...")
        if not os.path.exists(sim_file):
            print(f"Simulation file does not exist: {sim_file}")
            continue

        try:
            # Pre-read each indicator's sheet from the Excel file
            xls = pd.ExcelFile(sim_file)
            indicator_data = {}
            for ind in indicators:
                try:
                    df = pd.read_excel(xls, sheet_name=ind)
                    indicator_data[ind] = df
                    print(f"Successfully read data for indicator {ind}")
                except Exception as e:
                    print(f"Failed to read indicator {ind}: {str(e)}")
            if len(indicator_data) != len(indicators):
                print("Failed to read data for some indicators, skipping this scenario.")
                continue
        except Exception as e:
            print(f"Failed to read simulation file: {str(e)}")
            continue

        num_sim = 500  # Number of simulations
        predictions_list = []
        base_df = None
        history_hashes = set()

        for sim_idx in range(1, num_sim + 1):
            sim_data = {}
            valid_sim = True
            for ind in indicators:
                df = indicator_data[ind]
                col_name = f"Simulation_{sim_idx}"
                if col_name not in df.columns:
                    print(f"Indicator {ind} is missing column {col_name}, skipping simulation {sim_idx}")
                    valid_sim = False
                    break
                try:
                    df_sim = df.loc[:, ["Province", "Year", col_name]].copy()
                    df_sim = df_sim.rename(columns={col_name: ind})
                    sim_data[ind] = df_sim
                except Exception as e:
                    print(f"Failed to process indicator {ind} in simulation {sim_idx}: {str(e)}")
                    valid_sim = False
                    break
            if not valid_sim:
                continue

            merged_sim = sim_data[indicators[0]]
            for ind in indicators[1:]:
                merged_sim = pd.merge(merged_sim, sim_data[ind], on=["Province", "Year"], how="inner")

            if sim_idx == 1:
                base_df = merged_sim[["Province", "Year"]].copy()

            feature_cols = indicators
            missing_cols = [col for col in feature_cols if col not in merged_sim.columns]
            if missing_cols:
                print(f"Simulation {sim_idx} is missing feature columns: {missing_cols}")
                continue

            # Generate hash to avoid duplicate data
            current_hash = hashlib.sha256(merged_sim.to_csv().encode()).hexdigest()
            if current_hash in history_hashes:
                print(f"Warning: Simulation {sim_idx} data is a duplicate of a previous run!")
            else:
                history_hashes.add(current_hash)

            try:
                # Keep DataFrame structure to retain column names
                X_base = merged_sim[feature_cols].copy()

                # Ensure scaler is fitted (if not, fit it with the current data)
                if not hasattr(scaler_X, 'min_'):
                    print("scaler_X has not been fitted, fitting with current data now.")
                    scaler_X.fit(X_base)

                X_base_scaled = scaler_X.transform(X_base)

                preds = []
                for model_type in base_model_types:
                    pred = base_models[model_type].predict(X_base_scaled)
                    preds.append(pred.reshape(-1, 1))
                X_stacked = np.hstack(preds)
                y_pred = stacking_model.predict(X_stacked)

                sim_pred_df = pd.DataFrame({f"Prediction_{sim_idx}": y_pred.flatten()}, index=merged_sim.index)
                predictions_list.append(sim_pred_df)
            except Exception as e:
                print(f"Simulation {sim_idx} prediction failed: {str(e)}")
                continue

        if not predictions_list:
            print("No prediction results generated for this scenario.")
            continue

        # Concatenate all simulation predictions and merge with Province and Year data
        predictions_df = pd.concat(predictions_list, axis=1)
        pred_df = pd.concat([base_df, predictions_df], axis=1)

        # Save the raw prediction results
        output_file = os.path.join(output_dir, f"{cluster_name.replace(' ', '')}_{scenario}_CO2_Predictions.xlsx")
        try:
            pred_df.to_excel(output_file, index=False)
            print(f"Successfully saved prediction results: {output_file}")
        except Exception as e:
            print(f"Failed to save results: {str(e)}")

        # ================ Group by 'Year' and sum data across provinces for each simulation run =================
        # Make a copy to prevent memory fragmentation issues
        pred_df = pred_df.copy()
        agg_pred_df = pred_df.groupby("Year", as_index=False).sum()
        output_file_agg = os.path.join(output_dir, f"{cluster_name.replace(' ', '')}_{scenario}_CO2_Predictions_Aggregated.xlsx")
        try:
            agg_pred_df.to_excel(output_file_agg, index=False)
            print(f"Successfully saved aggregated prediction results: {output_file_agg}")
        except Exception as e:
            print(f"Failed to save aggregated results: {str(e)}")

print("\nAll processing complete!")