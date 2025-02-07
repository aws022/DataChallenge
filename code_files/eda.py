print("Hello world")

import os
from traceback import print_tb
import pandas as pd

# Read in the file
parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data_files")
hot_100_raw_file = os.path.join(data_dir, 'hot_100_current.csv')
love_songs_raw_file = os.path.join(data_dir, 'Love song categories for Billboard Top 10 hits, 1958 - September 2023, from The Pudding.csv')
rolling_stone_raw_file = os.path.join(data_dir, 'rolling_stone.csv')


if not os.path.exists(hot_100_raw_file):
    print(f"File not found: {hot_100_raw_file}")
else:
    print("File found, proceeding with reading.")


df_test = pd.read_csv(hot_100_raw_file, nrows=5)  # Load first 5 rows without filtering

print(df_test.columns)



# Get artists with the most weeks on chart
