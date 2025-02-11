
import os
from traceback import print_tb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


#print(rolling_stone.columns)


# Define 5-year age bins from 15 to 55
age_bins = list(range(15, 56, 5))  # 15-19, 20-24, ..., 50-54
age_labels = [f"{i}-{i+4}" for i in range(15, 55, 5)]  # "15-19", "20-24", ..., "50-54"

# Assign each artist to an age group
rolling_stone['Age Group'] = pd.cut(rolling_stone['Avg. Age at Top 500 Album'], bins=age_bins, labels=age_labels, right=False)

# Define decade bins for album release year
bins = list(range(1950, 2030, 10))  # 1950s to 2020s
labels = [f"{decade}s" for decade in range(1950, 2020, 10)]

# Assign each album to a release decade
rolling_stone['Decade'] = pd.cut(rolling_stone['Release Year'], bins=bins, labels=labels, right=False)

# Count the number of artists per (Decade, Age Group)
age_trend = rolling_stone.groupby(['Decade', 'Age Group']).size().unstack().fillna(0)

# Plot the heatmap to show trends over time
plt.figure(figsize=(12, 6))
sns.heatmap(age_trend, annot=True, fmt=".0f", cmap="viridis", linewidths=0.5)

# Formatting
plt.xlabel("Age Group (Years)")
plt.ylabel("Decade of Album Release")
plt.title("Change in Artist Age Distribution Over Time (Top 500 Albums)")

# Show plot
plt.show()
