import pandas as pd 

# Population Cleaning

def load_population_data(file_path):
    try:
        # Try reading the file with default parameters
        population_data = pd.read_csv(file_path)
    except pd.errors.ParserError:
        # If there's a parsing error, try reading with different delimiters
        try:
            population_data = pd.read_csv(file_path, delimiter=';')
        except pd.errors.ParserError:
            try:
                population_data = pd.read_csv(file_path, delimiter='\t')
            except pd.errors.ParserError:
                raise Exception("Unable to parse the CSV file. Please check the file format.")
    
    # Skip initial rows that might contain metadata
    population_data = pd.read_csv(file_path, skiprows=4)
    
    # Drop unnecessary columns if they exist
    columns_to_drop = ['Indicator Name', 'Indicator Code']
    population_data = population_data.drop(columns=columns_to_drop, axis=1, errors='ignore')
    
    # Keep only columns that are either 'Country Name', 'Country Code' or years
    population_data = population_data.loc[:, population_data.columns.str.match(r'Country Name|Country Code|\d{4}')]
    
    # Melt the dataframe to have years as rows
    population_data = population_data.melt(id_vars=['Country Name', 'Country Code'], 
                                           var_name='Year', 
                                           value_name='Population')
    
    # Convert the 'Year' column to integer, ignoring errors
    population_data['Year'] = pd.to_numeric(population_data['Year'], errors='coerce')
    
    # Drop rows with missing values in 'Year' or 'Population'
    population_data = population_data.dropna(subset=['Year', 'Population'])
    
    # Ensure 'Year' is an integer
    population_data['Year'] = population_data['Year'].astype(int)
    
    return population_data

population_path = 'data/population/population_data.csv'
# population_data = load_population_data(population_path)
# print(population_data.head(10), "\n")

# population = population_data.rename(columns={'Country Name': 'Country_Name', 'Country Code': 'Country_Code'})

# population.to_csv('data/population/cleaned_population_data.csv')


# GDP Cleaning

def load_economic_data(file_path):
    try:
        # Try reading the file with default parameters
        economic_data = pd.read_csv(file_path)
    except pd.errors.ParserError:
        # If there's a parsing error, try reading with different delimiters
        try:
            economic_data = pd.read_csv(file_path, delimiter=';')
        except pd.errors.ParserError:
            try:
                economic_data = pd.read_csv(file_path, delimiter='\t')
            except pd.errors.ParserError:
                raise Exception("Unable to parse the CSV file. Please check the file format.")
    
    # Skip initial rows that might contain metadata
    economic_data = pd.read_csv(file_path, skiprows=4)
    
    # Drop unnecessary columns if they exist
    # Assuming 'Indicator Name', 'Indicator Code', and 'Unnamed: 66' might be present
    columns_to_drop = ['Indicator Name', 'Indicator Code', 'Unnamed: 66']
    economic_data = economic_data.drop(columns=columns_to_drop, axis=1, errors='ignore')
    
    # Keep only columns that are either 'Country Name', 'Country Code' or years
    economic_data = economic_data.loc[:, economic_data.columns.str.match(r'Country Name|Country Code|\d{4}')]
    
    # Melt the dataframe to have years as rows
    economic_data = economic_data.melt(id_vars=['Country Name', 'Country Code'], 
                                       var_name='Year', 
                                       value_name='GDP')
    
    # Convert the 'Year' column to integer, ignoring errors
    economic_data['Year'] = pd.to_numeric(economic_data['Year'], errors='coerce')
    
    # Drop rows with missing values in 'Year' or 'GDP'
    economic_data = economic_data.dropna(subset=['Year', 'GDP'])
    
    # Ensure 'Year' is an integer
    economic_data['Year'] = economic_data['Year'].astype(int)
    
    return economic_data

economic_path = 'data/economic/economic_data.csv'
# economic_data = load_economic_data(economic_path)
# print(economic_data.head(10),"\n")

# economic = economic_data.rename(columns={'Country Name': 'Country_Name', 'Country Code': 'Country_Code'})

# economic.to_csv('data/economic/cleaned_economic_data.csv')


def load_merge_data(population_file_path, economic_file_path, output_file_path):
    # Read the population and economic data from the provided file paths
    population_data = pd.read_csv(population_file_path)
    economic_data = pd.read_csv(economic_file_path)

    # Merge GDP and Population data on Country Name, Country Code, and Year
    merged_data = pd.merge(economic_data, population_data, on=['Country_Name', 'Country_Code', 'Year'])

    # Format the GDP column to remove decimals
    merged_data['GDP'] = merged_data['GDP'].apply(lambda x: '{:.0f}'.format(x))

    # Drop unnecessary columns
    population_economic_data = merged_data.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])

    # Save the merged data to a CSV file
    population_economic_data.to_csv(output_file_path, index=False)

    return population_economic_data

population_file_path = 'data/population/cleaned_population_data.csv'
economic_file_path = 'data/economic/cleaned_economic_data.csv'
output_file_path = 'clean_data/eco_pop_data/population_economic_data.csv'

# merged_data = load_merge_data(population_file_path, economic_file_path, output_file_path)

# print(merged_data.head(6))

# print(merged_data.isnull().sum())