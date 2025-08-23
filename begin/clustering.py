import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull

# 设置字体为 Times New Roman，图像中支持英文输出
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 14})

# ✅ 请修改为你的 Excel 路径
data_path = r"C:\Users\因子提取.xlsx"
df = pd.read_excel(data_path)

# 提取用于聚类的字段
X = df[['综合GDP增速因子', '综合CO2排放增速因子']]

# K-Means 聚类
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# 自定义颜色
custom_colors = ['#f6c6af', '#afd4e3', '#b5d4be', '#cfafd4']

# 英文坐标名称替换
df.rename(columns={
    '综合GDP增速因子': 'GDP Factor',
    '综合CO2排放增速因子': 'CO2 Factor'
}, inplace=True)

# 绘图
plt.figure(figsize=(14, 8))

# 凸包绘制
for cluster in np.unique(df['Cluster']):
    points = df[df['Cluster'] == cluster][['GDP Factor', 'CO2 Factor']].values
    if len(points) > 2:
        hull = ConvexHull(points)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha=0.5, color=custom_colors[cluster])
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k--', alpha=0.5)

# 散点图
sns.scatterplot(
    x='GDP Factor', y='CO2 Factor',
    hue='Cluster', style='Cluster',
    palette=custom_colors, data=df,
    s=100, markers=['o', '^', 'D', 's'], edgecolor='black'
)

# 聚类中心
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker='*', label='Centroid')

# 均值线
mean_x = df['GDP Factor'].mean()
mean_y = df['CO2 Factor'].mean()
plt.axvline(mean_x, color='black', linewidth=1, linestyle='--')
plt.axhline(mean_y, color='black', linewidth=1, linestyle='--')

# 加粗坐标轴标签
plt.xlabel('Composite GDP Growth Factor', fontsize=16, fontweight='bold')
plt.ylabel('Composite CO₂ Emission Growth Factor', fontsize=16, fontweight='bold')
plt.title('K-Means Clustering with Convex Hull and Mean Lines', fontsize=18)

# 修改图例为 Cluster I–IV
handles, labels = plt.gca().get_legend_handles_labels()
roman_labels = ['Cluster I', 'Cluster II', 'Cluster III', 'Cluster IV']
new_labels = [roman_labels[int(label)] if label.isdigit() else label for label in labels]
plt.legend(handles, new_labels, title='Cluster', loc='upper left', fontsize=12)

# 保存高分辨率图像
output_image_path = r"C:\Users\聚类图_英文版高分辨率.png"
plt.savefig(output_image_path, dpi=2000)

plt.tight_layout()
plt.show()

# 保存结果
output_path = r"C:\Users\聚类结果英文版.xlsx"
df.to_excel(output_path, index=False)

# 输出最近省份（若存在列名“省份”）
if '省份' in df.columns:
    print("\nMost representative provinces for each cluster:")
    for cluster in range(4):
        subset = df[df['Cluster'] == cluster]
        points = subset[['GDP Factor', 'CO2 Factor']].values
        names = subset['省份'].values
        center = centroids[cluster]
        distances = np.linalg.norm(points - center, axis=1)
        print(f"Cluster {cluster+1} → {names[np.argmin(distances)]}")
else:
    print("⚠️ No '省份' column found in the data.")
