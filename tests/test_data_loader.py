import pytest
import pandas as pd
from data_loader import load_data, load_csv_data, clean_data, filter_by_year_range, combine_data

@pytest.fixture
def sample_ratings_data():
    data = {
        'tconst': ['tt0000001', 'tt0000002', 'tt0000003'],
        'averageRating': [5.6, 6.2, 7.1],
        'numVotes': [1500, 300, 2000]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_basics_data():
    data = {
        'tconst': ['tt0000001', 'tt0000002', 'tt0000003'],
        'primaryTitle': ['Movie1 USA', 'Movie2 ITA', 'Movie3 USA'],
        'titleType': ['movie', 'movie', 'movie'],
        'startYear': [2020, 2019, 2018],
        'genres': ['Drama', 'Comedy', 'Action']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_gdp_data():
    data = {
        'country': ['USA', 'ITA'],
        'GDP': [21000000, 2000000]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_population_data():
    data = {
        'country': ['USA', 'ITA'],
        'population': [331000000, 60000000]
    }
    return pd.DataFrame(data)

def test_load_data(sample_ratings_data):
    assert not sample_ratings_data.empt
