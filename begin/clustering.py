import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull

# Set font to Times New Roman for English output
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 14})

# ✅ Please modify to your Excel path
data_path = r"C:\Users\factor_extraction.xlsx"
df = pd.read_excel(data_path)

# Rename original Chinese columns to English
df.rename(columns={
    'Composite GDP Growth Factor': 'Composite GDP Growth Factor',
    'Composite CO2 Emission Growth Factor': 'Composite CO2 Emission Growth Factor',
    'Province': 'Province'
}, inplace=True)

# Extract fields for clustering
X = df[['Composite GDP Growth Factor', 'Composite CO2 Emission Growth Factor']]

# K-Means clustering
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Custom colors
custom_colors = ['#f6c6af', '#afd4e3', '#b5d4be', '#cfafd4']

# Plotting
plt.figure(figsize=(14, 8))

# Draw convex hull
for cluster in np.unique(df['Cluster']):
    points = df[df['Cluster'] == cluster][['Composite GDP Growth Factor', 'Composite CO2 Emission Growth Factor']].values
    if len(points) > 2:
        hull = ConvexHull(points)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha=0.5, color=custom_colors[cluster])
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k--', alpha=0.5)

# Scatter plot
sns.scatterplot(
    x='Composite GDP Growth Factor', y='Composite CO2 Emission Growth Factor',
    hue='Cluster', style='Cluster',
    palette=custom_colors, data=df,
    s=100, markers=['o', '^', 'D', 's'], edgecolor='black'
)

# Plot cluster centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker='*', label='Centroid')

# Draw mean lines
mean_x = df['Composite GDP Growth Factor'].mean()
mean_y = df['Composite CO2 Emission Growth Factor'].mean()
plt.axvline(mean_x, color='black', linewidth=1, linestyle='--')
plt.axhline(mean_y, color='black', linewidth=1, linestyle='--')

# Bold axis labels
plt.xlabel('Composite GDP Growth Factor', fontsize=16, fontweight='bold')
plt.ylabel('Composite CO₂ Emission Growth Factor', fontsize=16, fontweight='bold')
plt.title('K-Means Clustering with Convex Hull and Mean Lines', fontsize=18)

# Modify legend to 'Cluster I–IV'
handles, labels = plt.gca().get_legend_handles_labels()
roman_labels = ['Cluster I', 'Cluster II', 'Cluster III', 'Cluster IV']
new_labels = [roman_labels[int(label)] if label.isdigit() else label for label in labels]
plt.legend(handles, new_labels, title='Cluster', loc='upper left', fontsize=12)

# Save high-resolution image
output_image_path = r"C:\Users\cluster_plot_high_res.png"
plt.savefig(output_image_path, dpi=2000)

plt.tight_layout()
plt.show()

# Save clustering results to a new Excel file
output_path = r"C:\Users\clustering_results.xlsx"
df.to_excel(output_path, index=False)

# Output the most representative provinces (if 'Province' column exists)
if 'Province' in df.columns:
    print("\nMost representative provinces for each cluster:")
    for cluster in range(4):
        subset = df[df['Cluster'] == cluster]
        points = subset[['Composite GDP Growth Factor', 'Composite CO2 Emission Growth Factor']].values
        names = subset['Province'].values
        center = centroids[cluster]
        distances = np.linalg.norm(points - center, axis=1)
        print(f"Cluster {cluster+1} -> {names[np.argmin(distances)]}")
else:
    print("⚠️ No 'Province' column found in the data.")