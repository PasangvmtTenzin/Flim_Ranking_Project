import pytest
import pandas as pd
from src import analysis

# Sample data for testing
sample_data = pd.DataFrame({
    'Year': [2000, 2000, 2001, 2001, 2002, 2002],
    'Country_Name': ['USA', 'UK', 'USA', 'UK', 'USA', 'UK'],
    'total_votes': [1000000, 1500000, 2000000, 2500000, 3000000, 3500000],
    'average_quality_score': [8.0, 7.5, 8.5, 7.0, 9.0, 6.5]
})

@pytest.fixture
def data():
    return sample_data

def test_load_data(mocker):
    mocker.patch('pandas.read_csv', return_value=sample_data)
    data = analysis.load_data('dummy_path.csv')
    assert not data.empty
    assert 'Year' in data.columns
    assert 'Country_Name' in data.columns

def test_calculate_weak_impact(data):
    weak_impact = analysis.calculate_weak_impact(data)
    assert 'weakImpact' in weak_impact.columns
    assert len(weak_impact) == len(data['Country_Name'].unique()) * len(data['Year'].unique())

def test_calculate_strong_impact(data):
    strong_impact = analysis.calculate_strong_impact(data)
    assert 'strongImpact' in strong_impact.columns
    assert len(strong_impact) == len(data['Country_Name'].unique()) * len(data['Year'].unique())

def test_analyze_quality_by_country(data):
    quality_by_country = analysis.analyze_quality_by_country(data, top_n=2)
    assert 'average_quality_score' in quality_by_country.columns
    assert quality_by_country['average_quality_score'].max() <= 9.0
    assert quality_by_country['average_quality_score'].min() >= 6.5

def test_millions_formatter():
    formatted = analysis.millions_formatter(1500000, 0)
    assert formatted == '1.5M'

def test_plot_and_save_weak_impact(mocker, data):
    mocker.patch('matplotlib.animation.FuncAnimation.save')
    analysis.plot_and_save_weak_impact(data, 'test_weak_impact.gif')

def test_plot_and_save_strong_impact(mocker, data):
    mocker.patch('matplotlib.animation.FuncAnimation.save')
    analysis.plot_and_save_strong_impact(data, 'test_strong_impact.gif')

def test_plot_and_save_quality_by_country(mocker, data):
    mocker.patch('matplotlib.animation.FuncAnimation.save')
    analysis.plot_and_save_quality_by_country(data, 'test_quality_by_country.gif')

def test_perform_analysis(mocker):
    mocker.patch('src.analysis.load_data', return_value=sample_data)
    mocker.patch('src.analysis.plot_and_save_weak_impact')
    mocker.patch('src.analysis.plot_and_save_strong_impact')
    mocker.patch('src.analysis.plot_and_save_quality_by_country')
    
    analysis.perform_analysis('dummy_path.csv', start_year=1960, end_year=2002)
    
    analysis.plot_and_save_weak_impact.assert_called_once()
    analysis.plot_and_save_strong_impact.assert_called_once()
    analysis.plot_and_save_quality_by_country.assert_called_once()
