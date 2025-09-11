import pandas as pd
import numpy as np

# Define city types, scenarios, time periods, and variables
clusters = ['Cluster I', 'Cluster II', 'Cluster III', 'Cluster IV']
scenarios = ['MS', 'BAU', 'HG']
periods = ['2023-2030', '2031-2040', '2041-2050']
variables = ['G', 'SI', 'EI', 'ES', 'DMSP']

# Original means and variances
raw_data = {
    'Cluster I': {'G': (13.343/2, 5.595/2), 'ES': (-5.981, 1.282), 'EI': (-6.14, 6.329), 'DMSP': (13.052, 23.794), 'SI': (0.132, 2.966)},
    'Cluster II': {'G': (10.614/2, 5.819/2), 'ES': (-4.582, 1.086), 'EI': (-5.414, 5.51), 'DMSP': (8.129, 25.357), 'SI': (-1.443, 3.679)},
    'Cluster III': {'G': (12.521/2, 6.042/2), 'ES': (-3.493, 1.647), 'EI': (-5.207, 4.739), 'DMSP': (11.015, 20.139), 'SI': (0.124, 4.165)},
    'Cluster IV': {'G': (13.548/2, 6.871/2), 'ES': (-2.069, 1.389), 'EI': (-3.538, 6.519), 'DMSP': (13.476, 26.697), 'SI': (0.777, 4.718)}
}

# Define fine-tuned scaling factors (cluster x variable x scenario x period)
adjustments = {
    'Cluster I': {
        'G': {'MS': [0.8, 0.6, 0.4], 'BAU': [1.0, 0.8, 0.6], 'HG': [1.2, 1.0, 0.8]},
        'SI': {'MS': [1, 0.8, 0.6], 'BAU': [1.2, 1, 0.8], 'HG': [1.4, 1.2, 1]},
        'EI': {'MS': [1.2, 1.25, 1.3], 'BAU': [1, 1.1, 1.2], 'HG': [0.8, 0.85, 0.9]},
        'ES': {'MS': [1.2, 1.25, 1.3], 'BAU': [1, 1.1, 1.2], 'HG': [0.8, 0.85, 0.9]},
        'DMSP': {'MS': [0.8, 0.7, 0.65], 'BAU': [0.9, 0.85, 0.8], 'HG': [1.1, 1.05, 1.0]},
    },
    'Cluster II': {
        'G': {'MS': [0.78, 0.58, 0.4], 'BAU': [0.98, 0.78, 0.68], 'HG': [1.22, 1.08, 0.82]},
        'SI': {'MS': [0.78, 0.6, 0.5], 'BAU': [0.88, 0.7, 0.55], 'HG': [1.28, 1.04, 1]},
        'EI': {'MS': [1.2, 1.25, 1.3], 'BAU': [1, 1.1, 1.2], 'HG': [0.8, 0.85, 0.9]},
        'ES': {'MS': [1.2, 1.25, 1.3], 'BAU': [1, 1.1, 1.2], 'HG': [0.8, 0.85, 0.9]},
        'DMSP': {'MS': [0.78, 0.7, 0.6], 'BAU': [0.88, 0.83, 0.78], 'HG': [1.15, 1.1, 1.05]},
    },
    'Cluster III': {
        'G': {'MS': [0.65, 0.55, 0.45], 'BAU': [0.85, 0.75, 0.65], 'HG': [1.15, 1.1, 1.05]},
        'SI': {'MS': [0.75, 0.68, 0.6], 'BAU': [0.85, 0.8, 0.75], 'HG': [1.1, 1.05, 1.0]},
        'EI': {'MS': [1.2, 1.25, 1.3], 'BAU': [1, 1.1, 1.2], 'HG': [0.8, 0.85, 0.9]},
        'ES': {'MS': [1.2, 1.25, 1.3], 'BAU': [1, 1.1, 1.2], 'HG': [0.8, 0.85, 0.9]},
        'DMSP': {'MS': [0.75, 0.68, 0.62], 'BAU': [0.88, 0.8, 0.75], 'HG': [1.12, 1.08, 1.02]},
    },
    'Cluster IV': {
        'G': {'MS': [0.7, 0.6, 0.5], 'BAU': [0.92, 0.85, 0.78], 'HG': [1.18, 1.1, 1.05]},
        'SI': {'MS': [0.8, 0.72, 0.65], 'BAU': [0.9, 0.85, 0.8], 'HG': [1.1, 1.05, 1.02]},
        'EI': {'MS': [1.1, 1.15, 1.2], 'BAU': [0.9, 1, 1.05], 'HG': [0.8, 0.85, 0.9]},
        'ES': {'MS': [1.1, 1.15, 1.2], 'BAU': [0.9, 1, 1.05], 'HG': [0.8, 0.85, 0.9]},
        'DMSP': {'MS': [0.82, 0.75, 0.68], 'BAU': [0.94, 0.88, 0.83], 'HG': [1.12, 1.06, 1.01]},
    }
}

# Build the simulation parameters table
simulation_data = []
for cluster in clusters:
    for scenario in scenarios:
        for i, period in enumerate(periods):
            row = {
                'City Type': cluster,
                'Scenario': scenario,
                'Time Period': period
            }
            for var in variables:
                mean, var_ = raw_data[cluster][var]
                m = adjustments[cluster][var][scenario][i]
                row[f'{var}_Mean'] = round(mean * m, 3)
                row[f'{var}_Variance'] = round(var_ * ((-m + 2) ** 1), 3)
            simulation_data.append(row)

# Convert to DataFrame and save
df_simulation = pd.DataFrame(simulation_data)
df_simulation.to_excel("Simulation_Parameters_Final.xlsx", index=False)