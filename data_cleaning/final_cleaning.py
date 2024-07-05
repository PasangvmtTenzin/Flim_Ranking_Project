import pandas as pd 
from collections import Counter


cinematic_data_path = 'merged_data/merged_cinematic_data.csv'
population_economic_data_path = 'merged_data/population_economic_data.csv'

# Function to calculate the mode
def mode(series):
    return series.mode().iloc[0] if not series.mode().empty else None

# Function to find the least occurring value
def least_occurring_value(series):
    counts = Counter(series.dropna())
    if counts:
        least_occurrence = counts.most_common()[-1][0]
        return least_occurrence
    return None

def load_cinematic_data(cinematic_data_path):
    cinematic_data = pd.read_csv(cinematic_data_path)

    filtered_data = cinematic_data[cinematic_data['startYear'] >= 1960].dropna(subset=['startYear'])

    # convert 'startYear' to integers
    filtered_data['startYear'] = filtered_data['startYear'].astype('int64')

    # Fill null values with the least occurring value
    columns_with_nulls = filtered_data.columns[filtered_data.isnull().any()].tolist()
    for col in columns_with_nulls:
        least_occur_value = least_occurring_value(filtered_data[col])
        filtered_data[col].fillna(least_occur_value, inplace=True)


    # Group by 'region' and 'startYear'
    grouped = filtered_data.groupby(['region', 'startYear'])

    # Aggregate data
    cinematic_aggregated = grouped.agg({
        'averageRating': 'mean',    # Mean of average ratings
        'numVotes': 'sum',          # Sum of votes
        'primaryTitle': mode,       # Mode of primaryTitle
        'genres': mode,             # mode of genres
        'primaryName': mode         # Mode of primaryName
    }).reset_index()

    return cinematic_aggregated

cinematic = load_cinematic_data(cinematic_data_path)
cinematic.to_csv('merged_data/merged_cinematic_data.csv')
# print(cinematic.head(5))

def load_population_economic(population_economic_data_path):
    population_economic_data = pd.read_csv(population_economic_data_path)
    eco_grouped = population_economic_data.groupby(['Country_Code', 'Year'])

    # Aggregate data
    eco_aggregated = eco_grouped.agg({
        'GDP': 'sum',    # Mean of average ratings
        'Population': 'sum',          # Sum of votes
        'Country_Name': mode       # Mode of primaryTitle
    }).reset_index()

    # Calculate GDP per capita
    eco_aggregated['GDP_per_Capital'] = eco_aggregated['GDP'] / eco_aggregated['Population']


    return eco_aggregated

# population_economic = load_population_economic(population_economic_data_path)
# population_economic.to_csv('merged_data/population_economic_data.csv')
# print(population_economic.head())
