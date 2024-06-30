# tests/test_model.py
import pytest
import pandas as pd
from src.model import measure_cinematic_impact, analyze_quality

@pytest.fixture
def sample_data():
    data = {
        'title': ['Movie1', 'Movie2', 'Movie3'],
        'startYear': [2020, 2019, 2018],
        'averageRating': [8.0, 7.5, 9.0],
        'numVotes': [1000, 1500, 500],
        'country': ['USA', 'France', 'USA']
    }
    return pd.DataFrame(data)

def test_measure_cinematic_impact(sample_data):
    impact = measure_cinematic_impact(sample_data)
    assert 'USA' in impact.index
    assert impact.loc['USA'] > impact.loc['France']

def test_analyze_quality(sample_data):
    top_movies = analyze_quality(sample_data, 2)
    assert len(top_movies) == 2
    assert top_movies.iloc[0]['title'] == 'Movie3'  # Highest rating
