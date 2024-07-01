# tests/test_data_loader.py

import pytest
import pandas as pd
from src.data_loader import load_csv

def test_load_csv_valid(mock_pandas_read_csv):
    # Setup mock to return a valid DataFrame
    mock_pandas_read_csv.return_value = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    
    df = load_csv('some_path.csv')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)

def test_load_csv_non_existent(mock_pandas_read_csv):
    # Setup mock to raise FileNotFoundError
    mock_pandas_read_csv.side_effect = FileNotFoundError
    
    with pytest.raises(FileNotFoundError):
        load_csv('non_existent_file.csv')

def test_load_csv_empty(mock_pandas_read_csv):
    # Setup mock to raise EmptyDataError
    mock_pandas_read_csv.side_effect = pd.errors.EmptyDataError
    
    with pytest.raises(pd.errors.EmptyDataError):
        load_csv('empty_file.csv')

def test_load_csv_invalid(mock_pandas_read_csv):
    # Setup mock to raise ParserError
    mock_pandas_read_csv.side_effect = pd.errors.ParserError
    
    with pytest.raises(pd.errors.ParserError):
        load_csv('invalid_file.csv')
