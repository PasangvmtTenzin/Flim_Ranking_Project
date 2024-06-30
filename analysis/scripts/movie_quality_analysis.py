import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import numpy as np

# Load the dataset
file_path = 'merged_data/merged_cinematic_data.csv'
movie_data = pd.read_csv(file_path)

# Sort the movies by average rating in descending order
sorted_movies_df = movie_data.sort_values(by='averageRating', ascending=False)

# Initialize dictionary to store rankings
rankings = {size: [] for size in range(10, 201, 10)}

# Calculate rankings for different set sizes
for size in range(10, 201, 10):
    # Select the top 'size' movies
    top_movies = sorted_movies_df.head(size)
    # Count the number of movies per country
    country_counts = top_movies['region'].value_counts()
    # Store the top 10 countries in the rankings dictionary
    rankings[size] = country_counts.nlargest(10).index.tolist()

# Convert rankings to DataFrame for better visualization
rankings_df = pd.DataFrame.from_dict(rankings, orient='index')


# Prepare data for Plotly
plotly_data = []

for size in rankings_df.index:
    for rank, country in enumerate(rankings_df.loc[size]):
        plotly_data.append({
            'Set Size': size,
            'Country': country,
            'Rank': rank + 1
        })

plotly_df = pd.DataFrame(plotly_data)

# Create interactive plot
fig = px.line(plotly_df, x='Set Size', y='Rank', color='Country', markers=True,
              title='Top 10 Countries by Representation in Top Rated Movies',
              labels={'Set Size': 'Set Size', 'Rank': 'Country Rank'},
              hover_name='Country', hover_data={'Set Size': True, 'Rank': True})

# Invert y-axis to have rank 1 at the top
fig.update_yaxes(autorange="reversed")

# Update layout for better readability
fig.update_layout(
    legend_title='Country',
    legend=dict(
        title_font_family="Arial",
        font=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    )
)

# Show plot
fig.show()

# Save the plot as an HTML file
# fig.write_html('analysis/plots/movie/top_10_countries_representation.html')


""" This code includes general analysis of the dataset """

# A function to format the y-axis labels
def thousands_formatter(x, pos):
    return f'{int(x/1000)}K'

# Set the style for the plots
sns.set(style='whitegrid')

def plot_distribution_of_average_ratings(data):
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data['averageRating'], bins=20, kde=True)
    plt.title('Distribution of Average Ratings')
    plt.xlabel('Average Rating')
    plt.ylabel('Frequency')
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.format_coord = lambda x, y: f'x={x:.2f}, y={y:.0f}'
    plt.savefig('analysis/plots/movie_quality/distribution_of_average_ratings.png', dpi=300)
    plt.show()

def plot_number_of_votes_per_year(data):
    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(x='startYear', y='numVotes', data=data)
    plt.title('Number of Votes Per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Votes')
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.savefig('analysis/plots/movie_quality/number_of_votes_per_year.png', dpi=300)
    plt.show()

def plot_average_rating_per_genre(data):
    # Explode the genres
    data_exploded = data.assign(genres=data['genres'].str.split(',')).explode('genres')
    
    # Order genres by median rating
    order = data_exploded.groupby('genres')['averageRating'].median().sort_values(ascending=False).index
    
    plt.figure(figsize=(16, 10))
    sns.set(style="whitegrid")
    
    # Create the box plot
    ax = sns.boxplot(x='genres', y='averageRating', data=data_exploded, order=order, palette='Set3')
    
    # Setting the title and labels with larger font sizes
    plt.title('Average Rating per Genre', fontsize=20)
    plt.xlabel('Genre', fontsize=16)
    plt.ylabel('Average Rating', fontsize=16)
    
    # Rotate x-ticks for better readability and increase their font size
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add annotations for median values
    medians = data_exploded.groupby('genres')['averageRating'].median().loc[order]
    for i, median in enumerate(medians):
        ax.text(i, median + 0.05, f'{median:.2f}', ha='center', fontsize=10, color='black')
    
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.savefig('analysis/plots/movie_quality/average_rating_per_genre.png', dpi=300)
    plt.show()


def plot_correlation_between_votes_and_ratings(data):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Create scatter plot with smaller and semi-transparent markers
    sns.scatterplot(x='numVotes', y='averageRating', data=data, alpha=0.5, s=10, color='blue')

    # Setting the title and labels
    plt.title('Correlation Between Number of Votes and Average Rating', fontsize=20, fontweight='bold')
    plt.xlabel('Number of Votes', fontsize=14, fontweight='bold')
    plt.ylabel('Average Rating', fontsize=14, fontweight='bold')
    
    # Setting the x-axis to logarithmic scale
    plt.xscale('log')
    
    # Add a trend line (using linear regression)
    sns.regplot(x='numVotes', y='averageRating', data=data, scatter=False, color='red', line_kws={"lw": 2})

    # Add gridlines and adjust tick parameters
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    
    # Adjusting the layout to prevent clipping
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('analysis/plots/movie_quality/correlation_between_votes_and_ratings.png', dpi=300)
    plt.show()

def correlation_matrix(data):
    # Select only numerical columns
    numerical_data = data.select_dtypes(include=[np.number])

    # Calculate the correlation matrix
    correlation_matrix = numerical_data.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Draw the heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, square=True)

    # Add title and adjust layout
    plt.title('Correlation Matrix of the Dataset', fontsize=20, fontweight='bold')
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('analysis/plots/movie_quality/correlation_matrix.png', dpi=300)
    plt.show()

# Calling the functions
# plot_distribution_of_average_ratings(movie_data)
# plot_number_of_votes_per_year(movie_data)
# plot_average_rating_per_genre(movie_data)
# plot_correlation_between_votes_and_ratings(movie_data)
# correlation_matrix(movie_data)