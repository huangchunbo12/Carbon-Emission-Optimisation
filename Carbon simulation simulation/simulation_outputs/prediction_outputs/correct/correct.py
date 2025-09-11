import pandas as pd
import numpy as np
import os

# Define clustering scenarios and corresponding Excel file prefixes
scenarios = ['BAU', 'HG', 'MS']
clusters = ['ClusterI', 'ClusterII', 'ClusterIII', 'ClusterIV']
province_mapping = {'Yunnan': 'ClusterI', 'Shanghai': 'ClusterII', 'Shanxi': 'ClusterIII', 'Ningxia': 'ClusterIV'}

# Read the ARIMA-Cluster Centers.xlsx file (with 'Year' as index)
try:
    arima_df = pd.read_excel('Cluster_Centers-ARIMA.xlsx', index_col='Year')
except FileNotFoundError:
    print("Could not find 'Cluster_Centers-ARIMA.xlsx'. Please check the file path.")
    exit(1)

# Iterate through each scenario and cluster
for scenario in scenarios:
    for cluster in clusters:
        # Generate the corresponding Excel file name
        file_name = f'{cluster}_{scenario}_CO2_Predictions.xlsx'
        try:
            df = pd.read_excel(file_name)
        except FileNotFoundError:
            print(f"Could not find '{file_name}'. Please check the file path.")
            continue

        # Get the corresponding province name for the cluster
        try:
            province = [k for k, v in province_mapping.items() if v == cluster][0]
        except IndexError:
            print(f"No corresponding province found for cluster {cluster} in the mapping!")
            continue

        # Set 'Province' and 'Year' columns as a multi-index
        df.set_index(['Province', 'Year'], inplace=True)

        # Filter out the numerical columns that need correction
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Get the ARIMA prediction values for that province ('Year' is the index of arima_df)
        arima_values = arima_df[province]
        arima_avg = arima_values.mean()  # Calculate the mean of ARIMA predictions
        mean_values = df.groupby(level='Province').mean()  # Mean of single-province predictions

        # Use a dictionary to store the corrected numerical columns
        corrected_dict = {}
        for col in numeric_cols:
            year_factor = df.index.get_level_values('Year').map(arima_values)
            if cluster == 'ClusterIV':
                # Category IV uses ARIMA prediction / single-province prediction mean * original value
                mean_val = mean_values.loc[province, col]
                if mean_val == 0:
                    print(f"Scenario: {scenario}, Cluster: {cluster}, Column: {col}. The mean of single-province predictions is zero. Skipping correction and using original values.")
                    corrected_series = df[col].copy()
                else:
                    corrected_series = df[col] * (year_factor / mean_val)
            else:
                # Other categories use ARIMA prediction / ARIMA mean * original value
                if arima_avg == 0:
                    print(f"Scenario: {scenario}, Cluster: {cluster}. The mean of ARIMA predictions is zero. Skipping correction and using original values.")
                    corrected_series = df[col].copy()
                else:
                    corrected_series = df[col] * (year_factor / arima_avg)
            
            # Fill any missing values that might occur during calculation
            if corrected_series.isna().any():
                print(f"Scenario: {scenario}, Cluster: {cluster}, Column: {col} has missing values. They will be filled with the original values.")
                corrected_series = corrected_series.fillna(df[col])
            
            corrected_dict[f'{col}_Corrected'] = corrected_series

        # Create a DataFrame containing only the corrected data (while preserving the multi-index)
        corrected_df = pd.DataFrame(corrected_dict, index=df.index)

        # Reset the index, turning 'Province' and 'Year' into regular columns
        corrected_df = corrected_df.reset_index()

        # Output to a new Excel file, containing only the corrected columns and corresponding province/year info (non-aggregated)
        output_file_name = f'{cluster}_{scenario}_CO2_Predictions_Corrected.xlsx'
        try:
            corrected_df.to_excel(output_file_name, index=False)
            print(f"Scenario: {scenario}, Cluster: {cluster} -- Generated file with only corrected values: {output_file_name}")
        except Exception as e:
            print(f"Error writing to file {output_file_name}: {e}")

        # Aggregate corrected values from different provinces within the same cluster by summing for the same year
        agg_df = corrected_df.groupby('Year', as_index=False).sum(numeric_only=True)

        # Output the aggregated file (i.e., corrected values for different provinces in the same cluster and scenario summed by year)
        output_agg_file_name = f'{cluster}_{scenario}_CO2_Predictions_Corrected_Aggregated.xlsx'
        try:
            agg_df.to_excel(output_agg_file_name, index=False)
            print(f"Scenario: {scenario}, Cluster: {cluster} -- Generated aggregated file with values summed by year: {output_agg_file_name}")
        except Exception as e:
            print(f"Error writing to file {output_agg_file_name}: {e}")