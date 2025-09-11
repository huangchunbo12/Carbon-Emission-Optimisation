import pandas as pd

# Read the clustering results file (clustering_with_cluster.xlsx)
cluster_data_path = r"C:\Users\clustering_with_cluster.xlsx"
cluster_df = pd.read_excel(cluster_data_path)

# Read the panel data file (panel_data.xlsx)
panel_data_path = r"C:\Users\panel_data.xlsx"
panel_df = pd.read_excel(panel_data_path)

# Merge the clustering results with the panel data
merged_df = pd.merge(panel_df, cluster_df[['Province', 'Cluster']], on='Province', how='left')

# Save the data with the 'Cluster' column added to a new Excel file
output_path = r"C:\Users\panel_data_with_cluster.xlsx"
merged_df.to_excel(output_path, index=False)

# Print the save path
print(f"Data with Cluster added has been saved to: {output_path}")