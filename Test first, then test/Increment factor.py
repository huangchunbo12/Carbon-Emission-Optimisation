import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


# Haversine function to calculate geographical distance between two points
def haversine(lon1, lat1, lon2, lat2):
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius (in km)
    R = 6371  # Earth's radius (km)
    distance = R * c
    return distance


# Load data
data_path = r"C:\Users\panel_data.xlsx"
df = pd.read_excel(data_path)

# Calculate GDP growth rate and CO2 emissions growth rate for each province
df['G Growth Rate'] = df.groupby('Province')['G'].pct_change() * 100  # GDP growth rate
df['CO2 Growth Rate'] = df.groupby('Province')['CO2 Emissions'].pct_change() * 100  # CO2 emissions growth rate

# Drop the first year's records where growth rates cannot be calculated
df = df.dropna(subset=['G Growth Rate', 'CO2 Growth Rate'])

# Calculate the 5-year rolling average growth rate as a temporal factor
df['5-Year G Growth Rate'] = df.groupby('Province')['G Growth Rate'].rolling(window=5).mean().reset_index(level=0,
                                                                                                               drop=True)
df['5-Year CO2 Growth Rate'] = df.groupby('Province')['CO2 Growth Rate'].rolling(window=5).mean().reset_index(level=0,
                                                                                                                   drop=True)

# Assuming you have longitude and latitude data, convert it after removing the "°" symbol
df['Longitude Center'] = (df['Westernmost Longitude'].apply(lambda x: float(str(x).replace('°', '').replace('E', ''))) + df[
    'Easternmost Longitude'].apply(lambda x: float(str(x).replace('°', '').replace('E', '')))) / 2
df['Latitude Center'] = (df['Southernmost Latitude'].apply(lambda x: float(str(x).replace('°', '').replace('N', ''))) + df[
    'Northernmost Latitude'].apply(lambda x: float(str(x).replace('°', '').replace('N', '')))) / 2

# Print the results for review
print(df[['Province', 'Longitude Center', 'Latitude Center']])

# Generate the longitude and latitude matrix
coordinates = df[['Longitude Center', 'Latitude Center']].values

# Calculate geographical distance between provinces (using Haversine distance)
num_provinces = len(coordinates)
distance_matrix = np.zeros((num_provinces, num_provinces))

for i in range(num_provinces):
    for j in range(i + 1, num_provinces):
        distance = haversine(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1])
        distance_matrix[i, j] = distance_matrix[j, i] = distance

# Set a threshold to create a spatial weight matrix; closer provinces get a larger weight
threshold = 500  # Set an appropriate threshold in km
W = distance_matrix < threshold  # Create an adjacency matrix based on distance; nearby provinces are considered adjacent

# Normalize the spatial weight matrix W to ensure each row sums to 1
row_sums = W.sum(axis=1)
W_normalized = W / row_sums[:, np.newaxis]  # Ensure the sum of each row is 1

# Use the spatial weight matrix W to apply weighted adjustments to the growth rates
df['Adjusted G Growth Rate'] = df['G Growth Rate'] + np.dot(W_normalized, df['G Growth Rate'])
df['Adjusted CO2 Growth Rate'] = df['CO2 Growth Rate'] + np.dot(W_normalized, df['CO2 Growth Rate'])

# Combine temporal and spatial factors to calculate the final composite growth rate
df['Composite GDP Growth Factor'] = df['5-Year G Growth Rate'] + df['Adjusted G Growth Rate']
df['Composite CO2 Emissions Factor'] = df['5-Year CO2 Growth Rate'] + df['Adjusted CO2 Growth Rate']

# Save the results to a new Excel file
output_path = r"C:\Users\panel_data_with_temporal_spatial_factors.xlsx"
df.to_excel(output_path, index=False)

# View the final results
print(df[['Province', 'Year', 'Composite GDP Growth Factor', 'Composite CO2 Emissions Factor']])