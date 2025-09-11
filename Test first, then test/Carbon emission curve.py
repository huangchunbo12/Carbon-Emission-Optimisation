import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Read Excel data
#    Change the path to your file location, e.g., r"C:\Users\xxx\Documents\emissions_data.xlsx"
file_path = r"C:\Users\Workbook1.xlsx"

# 2. Identify year and province columns (assuming the first column is 'Year', the rest are provinces)
year_col = df.columns[0]
province_cols = df.columns[1:]

# 3. Find the top 5 provinces with the highest emissions in 2022
last_year = df[year_col].max()  # Max year (e.g., 2022)
top5_series = (
    df.loc[df[year_col] == last_year, province_cols]  # Get emissions for each province in that year
      .T.squeeze()  # Transpose to a Series
      .sort_values(ascending=False)  # Sort from high to low
      .head(5)  # Get the top 5
)
top5 = top5_series.index.tolist()

# 4. Start plotting
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(top5)))  # Prepare color scheme for the top 5 provinces

for col in province_cols:
    if col in top5:
        # Top 5 provinces: colored, thick lines
        plt.plot(
            df[year_col], df[col],
            label=f"{col} ({int(top5_series[col]):,} tons)",
            linewidth=2.5,
            color=colors[top5.index(col)]
        )
    else:
        # Other provinces: gray, thin lines, low opacity
        plt.plot(
            df[year_col], df[col],
            color="grey",
            linewidth=0.8,
            alpha=0.35
        )

# 5. Enhance the chart
plt.title("Provincial Carbon Emission Trends, 2000–2022 (Highlighting Top 5 in 2022)", fontsize=15)
plt.xlabel("Year", fontsize=13)
plt.ylabel("Carbon Emissions (tons)", fontsize=13)

# Enlarge axis tick labels
plt.tick_params(axis="both", labelsize=13)  # Apply to both x and y-axis
# Can also be done separately: plt.xticks(fontsize=13); plt.yticks(fontsize=13)

plt.grid(linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend(
    title="Top 5 Provinces (2022 Emissions)",
    fontsize="small",
    title_fontsize="small",
    loc="upper left"
)
plt.tight_layout()
plt.show()