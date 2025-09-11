import pandas as pd

# Read the clustering results file (clustering_with_cluster.xlsx)
cluster_data_path = r"C:\Users\clustering_with_cluster.xlsx"
cluster_df = pd.read_excel(cluster_data_path)

# Read the panel data file (panel_data.xlsx)
panel_data_path = r"C:\Users\panel_data.xlsx"
panel_df = pd.read_excel(panel_data_path)

# Merge clustering results and panel data based on the 'Province' column
merged_df = pd.merge(panel_df, cluster_df[['Province', 'Cluster']], on='Province', how='left')

# Group data by the 'Cluster' column and save as separate Excel files
for cluster in merged_df['Cluster'].unique():
    cluster_data = merged_df[merged_df['Cluster'] == cluster]

    # Define the output file path
    output_path = f"C:/Users/Cluster_{int(cluster)}_data.xlsx"

    # Save to an Excel file
    cluster_data.to_excel(output_path, index=False)

    print(f"Cluster {int(cluster)} data saved to {output_path}")