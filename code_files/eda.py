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


hot_100 = pd.read_csv(hot_100_raw_file)  # Load first 5 rows without filtering
love_songs = pd.read_csv(love_songs_raw_file)
rolling_stone = pd.read_csv(rolling_stone_raw_file)

''' Ideas

1. Artists with the most songs on the Billboard


2. Best years for music
    Years in Rolling stone 500 with the most songs from that year


3. Nostalgia - Is Generation Z causing older songs to be added to the 
    Billboard 100 and the Rolling Stone 500
    
    * Looking at 'Years Between Debut and Top 500 Album' in Rolling Stone 500
    and '2020-2003 Differential'
'''
