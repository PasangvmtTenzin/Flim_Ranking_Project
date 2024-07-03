import pandas as pd
import os

def load_csv(file_path):
    abs_file_path = os.path.join(os.path.dirname(__file__), '..', 'merged_data', file_path)
    return pd.read_csv(abs_file_path)
