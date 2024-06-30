# cinematic_impact_lib/analysis.py
import pandas as pd

def load_and_clean_imdb_data():
    # Load and clean IMDb data
    basics = pd.read_csv('data/title.basics.tsv.gz', sep='\t', na_values='\\N', low_memory=False)
    ratings = pd.read_csv('data/title.ratings.tsv.gz', sep='\t', na_values='\\N', low_memory=False)
    
    # Merge basics and ratings
    data = pd.merge(basics, ratings, on='tconst')
    
    # Keep relevant columns
    data = data[['tconst', 'primaryTitle', 'startYear', 'genres', 'averageRating', 'numVotes']]
    
    # Filter out missing values and non-movies
    data = data.dropna(subset=['startYear', 'genres', 'averageRating', 'numVotes'])
    data = data[data['titleType'] == 'movie']
    
    return data

def load_gdp_data(file_path):
    # Load and clean GDP data
    gdp_data = pd.read_csv(file_path)
    return gdp_data

def load_population_data(file_path):
    # Load and clean population data
    population_data = pd.read_csv(file_path)
    return population_data

def perform_analysis(gdp_file, population_file, start_year=None, end_year=None):
    imdb_data = load_and_clean_imdb_data()
    gdp_data = load_gdp_data(gdp_file)
    population_data = load_population_data(population_file)
    
    # Filter IMDb data based on years
    if start_year and end_year:
        imdb_data = imdb_data[(imdb_data['startYear'] >= start_year) & (imdb_data['startYear'] <= end_year)]
    
    # Implement further analysis...
    
def calculate_weak_impact(data):
    country_votes = data.groupby('originCountry')['numVotes'].sum().reset_index()
    country_votes = country_votes.rename(columns={'numVotes': 'weakImpact'})
    return country_votes

def calculate_strong_impact(data):
    country_ratings = data.groupby('originCountry')['averageRating'].mean().reset_index()
    country_ratings = country_ratings.rename(columns={'averageRating': 'strongImpact'})
    return country_ratings

def analyze_quality_by_country(data, top_n=100):
    top_movies = data.nlargest(top_n, 'averageRating')
    country_quality = top_movies.groupby('originCountry')['averageRating'].mean().reset_index()
    return country_quality.sort_values(by='averageRating', ascending=False)

def perform_analysis(gdp_file, population_file, start_year=None, end_year=None):
    imdb_data = load_and_clean_imdb_data()
    gdp_data = load_gdp_data(gdp_file)
    population_data = load_population_data(population_file)
    
    # Filter IMDb data based on years
    if start_year and end_year:
        imdb_data = imdb_data[(imdb_data['startYear'] >= start_year) & (imdb_data['startYear'] <= end_year)]
    
    # Weak and strong impact analysis
    weak_impact = calculate_weak_impact(imdb_data)
    strong_impact = calculate_strong_impact(imdb_data)
    
    # Quality of movies by country
    quality_by_country = analyze_quality_by_country(imdb_data, top_n=100)
    
    # Output results
    print("Weak Impact by Country:")
    print(weak_impact.head(10))
    
    print("\nStrong Impact by Country:")
    print(strong_impact.head(10))
    
    print("\nQuality of Movies by Country (Top 100):")
    print(quality_by_country.head(10))
