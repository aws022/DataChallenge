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

''' Ideas

1. Artists with the most songs on the Billboard


2. Best years for music
    Years in Rolling stone 500 with the most songs from that year


3. Nostalgia - Is Generation Z causing older songs to be added to the 
    Billboard 100 and the Rolling Stone 500
    
    * Looking at 'Years Between Debut and Top 500 Album' in Rolling Stone 500
    and '2020-2003 Differential'
    
4. Changes in popularity of genres in these lists over time
    THIS COLUMN IS INCOMPLETE AND DOES NOT GIVE USEFUL DATA
    
5. Average age at top 500 album over time






Peak position in spotify -> weeks on chart



'''
print("Hot 100")
print(hot_100.columns)

print("Love songs")
print(love_songs.columns)

print("Rolling stone")
print(rolling_stone.columns)
