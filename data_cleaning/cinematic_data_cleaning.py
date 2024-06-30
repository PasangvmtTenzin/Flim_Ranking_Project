import os
import dask.dataframe as dd

# Define the subdirectory containing your data files
data_directory = os.path.join(os.getcwd(), 'raw_cinematic_data')

# Define columns to use for each file
usecols = {
    'title.ratings': ['tconst', 'averageRating', 'numVotes'],
    'title.basics': ['tconst', 'primaryTitle', 'startYear', 'genres'],
    'title.crew': ['tconst', 'directors'],
    'title.akas': ['titleId', 'region'],
    'name.basics': ['nconst', 'primaryName']
}

# Helper function to check if file exists
def check_file_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

# Paths to the data files
ratings_path = os.path.join(data_directory, 'title.ratings.tsv.gz')
basics_path = os.path.join(data_directory, 'title.basics.tsv.gz')
crew_path = os.path.join(data_directory, 'title.crew.tsv.gz')
akas_path = os.path.join(data_directory, 'title.akas.tsv.gz')
names_path = os.path.join(data_directory, 'name.basics.tsv.gz')

# Check if all files exist
check_file_exists(ratings_path)
check_file_exists(basics_path)
check_file_exists(crew_path)
check_file_exists(akas_path)
check_file_exists(names_path)

# Load data using Dask for parallel processing
ratings_df = dd.read_csv(ratings_path, sep='\t', compression='gzip', usecols=usecols['title.ratings'], blocksize=None, na_values='\\N')
basics_df = dd.read_csv(basics_path, sep='\t', compression='gzip', usecols=usecols['title.basics'], blocksize=None, na_values='\\N', dtype={'startYear': 'float64'})
crew_df = dd.read_csv(crew_path, sep='\t', compression='gzip', usecols=usecols['title.crew'], blocksize=None, na_values='\\N')
akas_df = dd.read_csv(akas_path, sep='\t', compression='gzip', usecols=usecols['title.akas'], blocksize=None, na_values='\\N')
names_df = dd.read_csv(names_path, sep='\t', compression='gzip', usecols=usecols['name.basics'], blocksize=None, na_values='\\N')

# Clean data to ensure correct types and remove invalid entries
basics_df['startYear'] = dd.to_numeric(basics_df['startYear'], errors='coerce')

# Ensure consistent data types for merging
crew_df['directors'] = crew_df['directors'].astype('object')
names_df['nconst'] = names_df['nconst'].astype('object')

# Merge ratings_df and basics_df on 'tconst'
merged_df = dd.merge(ratings_df, basics_df, on='tconst')

# Merge crew_df with names_df to replace 'directors' with 'primaryName'
merged_df = dd.merge(merged_df, crew_df, on='tconst')
merged_df = dd.merge(merged_df, names_df, left_on='directors', right_on='nconst', suffixes=('_title', '_person'))

# Merge with akas_df to add 'region' and 'language'
merged_df = dd.merge(merged_df, akas_df[['titleId', 'region']], left_on='tconst', right_on='titleId')

# Ensure all columns are present in the final merged_df
columns_to_keep = ['tconst', 'averageRating', 'numVotes', 'primaryTitle', 'startYear', 'genres', 'region', 'primaryName']
merged_df = merged_df[columns_to_keep]

# Convert merged_df to pandas DataFrame to save as CSV
final_data_pd = merged_df.compute()

# Save the final DataFrame as CSV with null values properly handled
final_data_pd.to_csv('cleaned_data/merged_cinematic_data.csv', index=False, na_rep='NaN')
