import os
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# File read in
parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data_files")

hot_100_path = os.path.join(data_dir, 'hot_100_current.csv')
rolling_stone_path = os.path.join(data_dir, 'rolling_stone.csv')

# Check file existence
for file_path in [hot_100_path, rolling_stone_path]:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        print(f"File found: {file_path}")

# Load CSV files
hot_100 = pd.read_csv(hot_100_path)
rolling_stone = pd.read_csv(rolling_stone_path)

# Ensure necessary columns exist
if 'Spotify Popularity' not in rolling_stone.columns:
    raise KeyError("Column 'Spotify Popularity' not found in rolling_stone dataset")
if 'wks_on_chart' not in hot_100.columns:
    raise KeyError("Column 'wks_on_chart' not found in hot_100 dataset")

# Select relevant columns
x = rolling_stone[['Spotify Popularity']]
y = hot_100[['wks_on_chart']]

# Convert data to numeric, forcing errors to NaN (handles incorrect formats)
x = pd.to_numeric(x.squeeze(), errors='coerce')
y = pd.to_numeric(y.squeeze(), errors='coerce')

# Drop missing values while keeping row alignment
df_cleaned = pd.concat([x, y], axis=1).dropna()


# Extract cleaned x and y
x = df_cleaned['Spotify Popularity']
y = df_cleaned['wks_on_chart']


print(df_cleaned.head(50))

# Ensure x and y have the same length
print(f"Cleaned x size: {len(x)}, Cleaned y size: {len(y)}")
print(f"x data type: {x.dtype}, y data type: {y.dtype}")  # Debugging check

# Compute correlation
correlation, p_value = pearsonr(x, y)
print('Correlation:', correlation)
print("p_value:", p_value)

# Reshape for sklearn
X = x.values.reshape(-1, 1)  # Independent variable
Y = y.values  # Target variable (1D)

# Fit Linear Regression Model
model = LinearRegression().fit(X, Y)

# Predict values
predictions = model.predict(X)

# Scatter plot with regression line
plt.figure(figsize=(8,6))
plt.scatter(X, Y, alpha=0.6, label='Actual Data')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.title('Spotify Popularity vs. Weeks on Chart')
plt.xlabel('Spotify Popularity Score')
plt.ylabel('Weeks on Billboard Hot 100')
plt.legend()
plt.grid(True)
plt.show()


'''
Look at spotify popularity and predict weeks on chart of a song
For each song get peak weeks on chart to give us how long a song can possibly be on the chart
'''