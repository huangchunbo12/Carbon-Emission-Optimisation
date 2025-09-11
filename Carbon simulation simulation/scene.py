import pandas as pd
import numpy as np
import os

# Set a random seed for reproducibility
np.random.seed(42)

# Indicators to simulate (only these 5 indicators)
indicators = ['G', 'SI', 'ES', 'EI', 'DMSP']

# Mapping of cluster number to data file and 'City Type' in the parameters file
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

# Read the simulation parameters file, which contains the following columns:
# City Type, Scenario, Time Period, G_Mean, G_Variance, SI_Mean, SI_Variance, etc.
param_file = "simulation_parameters_final.xlsx"
sim_params = pd.read_excel(param_file)

# Output directory for results
output_dir = "simulation_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define target years for simulation: from 2023 to 2050
target_years = list(range(2023, 2051))

# Helper function to determine the time period string based on the year
def get_time_stage(year):
    if 2023 <= year <= 2030:
        return "2023-2030"
    elif 2031 <= year <= 2040:
        return "2031-2040"
    elif 2041 <= year <= 2050:
        return "2041-2050"
    else:
        return None

# Iterate through each cluster
for cluster in range(4):
    cluster_file = cluster_files[cluster]
    cluster_name = cluster_sim_names[cluster]
    
    try:
        # Read all data for the current cluster
        data = pd.read_excel(cluster_file)
    except Exception as e:
        print(f"Failed to read file {cluster_file}: {e}")
        continue

    # Check if data contains 'Year' and 'Province' columns
    if not set(['Year', 'Province']).issubset(data.columns):
        print(f"File {cluster_file} is missing 'Year' and 'Province' columns")
        continue

    # Extract 2022 data as the initial baseline
    baseline = data[data['Year'] == 2022].copy()
    if baseline.empty:
        print(f"No data found for the year 2022 in file {cluster_file}")
        continue
    baseline.reset_index(drop=True, inplace=True)

    # Filter parameters for the current cluster (match by 'City Type')
    cluster_params = sim_params[sim_params["City Type"] == cluster_name]
    if cluster_params.empty:
        print(f"No simulation parameters found for {cluster_name}")
        continue

    # A cluster might have multiple scenarios (e.g., MS, BAU, HG)
    scenarios = cluster_params["Scenario"].unique()
    for scenario in scenarios:
        # Filter all parameters for this scenario (for different time periods)
        scenario_params = cluster_params[cluster_params["Scenario"] == scenario]
        
        # Construct output filename, e.g., "ClusterI_MS_2023-2050.xlsx"
        cluster_name_str = cluster_name.replace(" ", "")
        output_filename = os.path.join(output_dir, f"{cluster_name_str}_{scenario}_2023-2050.xlsx")
        
        # Use ExcelWriter to write multiple sheets (one for each indicator)
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Perform chained simulation for each indicator
            for indicator in indicators:
                records = []
                # For each province in the baseline, perform chained simulation for this indicator
                for idx, row in baseline.iterrows():
                    province = row["Province"]
                    # The initial value is the actual value from 2022
                    prev_value = row[indicator]
                    
                    # Calculate the simulated value for each target year
                    for year in target_years:
                        time_stage = get_time_stage(year)
                        if time_stage is None:
                            continue
                        # Find the parameter row for the corresponding time period (assuming one row per period)
                        param_row = scenario_params[scenario_params["Time Period"] == time_stage]
                        if param_row.empty:
                            print(f"Could not find parameters for {cluster_name} {scenario} in time period {time_stage}")
                            continue
                        param_row = param_row.iloc[0]
                        # Construct the column names for the indicator's mean and variance
                        mean_col = f"{indicator}_Mean"
                        var_col = f"{indicator}_Variance"
                        if mean_col not in param_row or var_col not in param_row:
                            print(f"Missing parameter columns for {indicator}")
                            continue
                        mean_val = param_row[mean_col]
                        var_val = param_row[var_col]
                        std_val = np.sqrt(var_val)
                        
                        # Generate 500 random growth rates (in %) for the current year
                        sim_rates = np.random.normal(loc=mean_val, scale=std_val, size=500)
                        # Simulation formula: current year value = previous year value * (1 + growth rate/100)
                        sim_values = prev_value * (1 + sim_rates/100)
                        
                        # Record the values for each simulation path for the current year
                        record = {"Province": province, "Year": year}
                        for i, val in enumerate(sim_values):
                            record[f"Simulation_{i+1}"] = val
                        records.append(record)
                        
                        # Update prev_value: for chained simulation, each path is independent
                        # To ensure the chain, the previous value for each simulation path is different, so we save 500 chained results
                        prev_value = sim_values  # prev_value is now an array, which will be multiplied by new growth rates in each subsequent step
                
                # Convert to DataFrame
                result_df = pd.DataFrame(records)
                result_df.sort_values(by=["Province", "Year"], inplace=True)
                # Write to an Excel sheet, with the sheet name being the indicator's name
                result_df.to_excel(writer, sheet_name=indicator, index=False)
        
        print(f"Generated simulation output file: {output_filename}")