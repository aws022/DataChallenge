'''
Can use as lin reg template code

Shows prediction of a song's status on the 2003 list
versus the 2012 list
'''

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
rolling_stone_path = os.path.join(data_dir, 'rolling_stone.csv')

if not os.path.exists(rolling_stone_path):
    print(f"File not found: {rolling_stone_path}")
else:
    print("File found, proceeding with reading.")

# Load dataset
rolling_stone = pd.read_csv(rolling_stone_path)

# Select relevant columns
x = rolling_stone['2003 Rank']
y = rolling_stone['2012 Rank']

# Data cleaning - Drop null values
df_cleaned = rolling_stone[['2003 Rank', '2012 Rank']].dropna()
x = df_cleaned['2003 Rank']
y = df_cleaned['2012 Rank']

# Compute correlation
correlation, p_value = pearsonr(x, y)
print('Correlation:', correlation)
print("p_value:", p_value)

# Reshape for sklearn
X = x.values.reshape(-1, 1)  # Independent variable
Y = y.values  # Target variable (1D)

# Compute mutual information
mi = mutual_info_regression(X, Y)  # `Y` should be 1D, `X` should be 2D
print('Mutual Information:', mi)

# Scatter plot
plt.scatter(X, Y, alpha=0.6)
plt.title('Rolling Stone: 2003 Rank vs 2012 Rank')
plt.xlabel('2003 Rank')
plt.ylabel('2012 Rank')
plt.grid(True)

# Fit linear regression model
model = LinearRegression().fit(X, Y)

# Predict Y values
predictions = model.predict(X)

# Plot regression line
plt.plot(X, predictions, color='red', label='Regression Line')
plt.legend()
plt.show()
