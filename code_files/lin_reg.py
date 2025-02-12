
import os
from traceback import print_tb
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
love_songs_path = os.path.join(data_dir, 'Love song categories for Billboard Top 10 hits, 1958 - September 2023, from The Pudding.csv')
rolling_stone_path = os.path.join(data_dir, 'rolling_stone.csv')


if not os.path.exists(hot_100_path):
    print(f"File not found: {hot_100_path}")
else:
    print("File found, proceeding with reading.")

if not os.path.exists(love_songs_path):
    print(f"File not found: {love_songs_path}")
else:
    print("File found, proceeding with reading.")

if not os.path.exists(rolling_stone_path):
    print(f"File not found: {rolling_stone_path}")
else:
    print("File found, proceeding with reading.")


hot_100 = pd.read_csv(hot_100_path)
love_songs = pd.read_csv(love_songs_path)
rolling_stone = pd.read_csv(rolling_stone_path)


#print(hot_100.columns)

df = rolling_stone

''' Columns

Regression for weeks on chart of various attributes

Index(['chart_week', 'current_week', 'title', 'performer', 'last_week',
       'peak_pos', 'wks_on_chart'],
      dtype='object')
'''

print(df.columns)

x = df['2003 Rank']
y = df['2012 Rank']

# Data cleaning - Drop null values
x = x.dropna(how='any',axis=0)
y = y.dropna(how='any',axis=0)





# Compute correlation
correlation, p_value = pearsonr(x,y)
print('Correlation:', correlation)
print("p_value:", p_value)

y = np.array(y).reshape(-1,1)
mi = mutual_info_regression(y,x)

print('Mi:', mi)

plt.scatter(y,x,alpha=0.6)
plt.title('X on Y')
plt.xlabel('X variable')
plt.ylabel('Y variable')
plt.grid(True)


# Fit model
model = LinearRegression().fit(y,x)

# Predict the house value
predictions = model.predict(y)
plt.plot(x,y, color = 'red', label = 'Predicted Regression Line')
plt.show()


