import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from traceback import print_tb


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







# Convert 'chart_week' to datetime if it's not already
hot_100['chart_week'] = pd.to_datetime(hot_100['chart_week'])

# Extract decade from 'chart_week'
hot_100['Decade'] = (hot_100['chart_week'].dt.year // 10) * 10  # Converts to 1960, 1970, etc.
hot_100['Decade'] = hot_100['Decade'].astype(str) + "s"  # Format as "1960s", "1970s", etc.

# Define bins for 'wks_on_chart'
wks_bins = list(range(0, 101, 10))  # Bins from 0-9, 10-19, ..., 90-99 weeks
wks_labels = [f"{i}-{i+9}" for i in range(0, 100, 10)]  # Labels like "0-9", "10-19", etc.


# Unique top ranking songs
#print('hot_100.columns:',hot_100.columns)
#hot_100 = hot_100.drop_duplicates(subset=['title', 'performer'], keep='first')


# Assign each song to a weeks-on-chart bin
hot_100['Weeks Bin'] = pd.cut(hot_100['wks_on_chart'], bins=wks_bins, labels=wks_labels, right=False)

# Count the number of songs per (Decade, Weeks Bin)
weeks_trend = hot_100.groupby(['Decade', 'Weeks Bin']).size().unstack().fillna(0)

# Plot the heatmap to show trends over time
plt.figure(figsize=(12, 6))
sns.heatmap(weeks_trend, annot=True, fmt=".0f", cmap="coolwarm", linewidths=0.5)

# Formatting
plt.xlabel("Weeks on Chart")
plt.ylabel("Decade of Chart Appearance")
plt.title("Number of Songs on Hot 100 by week")

# Show plot
plt.show()


#Line plot of each bin over each decade
# Reset index so 'Decade' is a column
weeks_trend_reset = weeks_trend.reset_index()

# Melt the DataFrame to long format for plotting
weeks_trend_long = weeks_trend_reset.melt(id_vars="Decade", var_name="Weeks Bin", value_name="Count")

# Ensure the 'Weeks Bin' is in the correct order
weeks_trend_long["Weeks Bin"] = pd.Categorical(weeks_trend_long["Weeks Bin"], categories=wks_labels, ordered=True)

# Create the line plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=weeks_trend_long, x="Weeks Bin", y="Count", hue="Decade", marker="o")

# Formatting
plt.xlabel("Weeks on Chart Bin")
plt.ylabel("Number of Songs")
plt.title("Number of Songs on Hot 100 by Weeks on Chart Across Decades")
plt.xticks(rotation=45)  # Rotate for better readability
plt.legend(title="Decade", bbox_to_anchor=(1.05, 1), loc="upper left")  # Move legend outside

# Show the plot
plt.show()
