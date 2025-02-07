print("Hello world")

import os
from traceback import print_tb
import pandas as pd

# Read in the file
# Read the Tasks.md and answer all the questions using Python, Pandas, Numpy, Matplotlib, Seaboarn libraries.

parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data_files")
compas_raw_file = os.path.join(data_dir, 'hot_100_current.csv')

if not os.path.exists(compas_raw_file):
    print(f"File not found: {compas_raw_file}")
else:
    print("File found, proceeding with reading.")


df_test = pd.read_csv(compas_raw_file, nrows=5)  # Load first 5 rows without filtering

print(df_test.columns)