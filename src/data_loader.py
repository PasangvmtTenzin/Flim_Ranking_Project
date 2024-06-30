import pandas as pd
import gzip

def load_data(file_path):
    with gzip.open(file_path, 'rt', encoding='latin1', errors='ignore') as f:
        data = pd.read_csv(f, delimiter='\t', low_memory=False)
    return data

def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    data = data[['tconst', 'primaryTitle', 'titleType', 'startYear', 'genres']]
    data = data.dropna(subset=['startYear', 'genres'])
    data['startYear'] = pd.to_numeric(data['startYear'], errors='coerce')
    data = data.dropna(subset=['startYear'])
    return data

def filter_by_year_range(data, start_year, end_year):
    return data[(data['startYear'] >= start_year) & (data['startYear'] <= end_year)]

def combine_data(movie_data, gdp_data, population_data):
    combined_data = movie_data.merge(gdp_data, on='country').merge(population_data, on='country')
    return combined_data
