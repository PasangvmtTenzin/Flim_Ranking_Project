import pytest
import pandas as pd
from io import StringIO
from data_cleaning.gdp_population_cleaning import load_population_data, load_economic_data, load_merge_data

# Sample data for testing
population_csv = '../data/population/cleaned_population_data.csv'
economic_csv = '../data/economic/cleaned_economic_data.csv'

# Test loading population data
def test_load_population_data():
    population_data = StringIO(population_csv)
    df = load_population_data(population_data)
    assert not df.empty
    assert 'Country Name' in df.columns
    assert 'Country Code' in df.columns
    assert 'Year' in df.columns
    assert 'Population' in df.columns
    assert df['Population'].iloc[0] == 1000000

# Test loading economic data
def test_load_economic_data():
    economic_data = StringIO(economic_csv)
    df = load_economic_data(economic_data)
    assert not df.empty
    assert 'Country Name' in df.columns
    assert 'Country Code' in df.columns
    assert 'Year' in df.columns
    assert 'GDP' in df.columns
    assert df['GDP'].iloc[0] == 500000000

# Test merging data
def test_merge_data():
    population_data = StringIO(population_csv)
    economic_data = StringIO(economic_csv)
    
    pop_df = load_population_data(population_data)
    eco_df = load_economic_data(economic_data)
    
    pop_df = pop_df.rename(columns={'Country Name': 'Country_Name', 'Country Code': 'Country_Code'})
    eco_df = eco_df.rename(columns={'Country Name': 'Country_Name', 'Country Code': 'Country_Code'})
    
    pop_df.to_csv('cleaned_population_data.csv', index=False)
    eco_df.to_csv('cleaned_economic_data.csv', index=False)
    
    merged_df = load_merge_data('cleaned_population_data.csv', 'cleaned_economic_data.csv', 'merged_data.csv')
    assert not merged_df.empty
    assert 'Country_Name' in merged_df.columns
    assert 'Country_Code' in merged_df.columns
    assert 'Year' in merged_df.columns
    assert 'Population' in merged_df.columns
    assert 'GDP' in merged_df.columns
    assert 'GDP per Capita' in merged_df.columns
    assert merged_df['GDP per Capita'].iloc[0] == 500.0

if __name__ == "__main__":
    pytest.main()
