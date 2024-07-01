import pandas as pd
import os

def load_csv(file_path):
    # Construct the absolute path using os.path.join
    abs_file_path = os.path.join(os.path.dirname(__file__), file_path)
    return pd.read_csv(abs_file_path)

# Example usage:
data = load_csv('../merged_data/final_data.csv')



