import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FuncFormatter


def load_data(file_path):
    # Load the provided data
    data = pd.read_csv(file_path)
    return data

def calculate_weak_impact(data):
    country_votes = data.groupby(['Year', 'Country_Name'])['total_votes'].sum().reset_index()
    country_votes = country_votes.rename(columns={'total_votes': 'weakImpact'})
    return country_votes

def calculate_strong_impact(data):
    country_ratings = data.groupby(['Year', 'Country_Name'])['average_quality_score'].mean().reset_index()
    country_ratings = country_ratings.rename(columns={'average_quality_score': 'strongImpact'})
    return country_ratings

def analyze_quality_by_country(data, top_n=100):
    top_movies = data.groupby('Year').apply(lambda x: x.nlargest(top_n, 'average_quality_score')).reset_index(drop=True)
    country_quality = top_movies.groupby(['Year', 'Country_Name'])['average_quality_score'].mean().reset_index()
    return country_quality.sort_values(by=['Year', 'average_quality_score'], ascending=[True, False])

def millions_formatter(x, pos):
    return f'{x / 1e6:.0f}M'

def plot_and_save_weak_impact(data, file_name):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define the formatter
    formatter = FuncFormatter(millions_formatter)

    def update(year):
        ax.clear()
        year_data = data[data['Year'] == year].nlargest(10, 'weakImpact')
        bars = ax.bar(year_data['Country_Name'], year_data['weakImpact'], color=plt.cm.tab20.colors)
        ax.set_xlabel('Country')
        ax.set_ylabel('Total Votes (Weak Impact)')
        ax.set_title(f"Weak Impact by Country ({year})")
        ax.set_xticklabels(year_data['Country_Name'], rotation=90)
        ax.legend(bars, year_data['Country_Name'], loc='upper left')
        
        # Apply the formatter to y-axis
        ax.yaxis.set_major_formatter(formatter)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height / 1e6:.1f}M',   # Format label in millions
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    years = data['Year'].unique()
    ani = FuncAnimation(fig, update, frames=years, repeat=False)
    ani.save(file_name, writer=PillowWriter(fps=2))

def plot_and_save_strong_impact(data, file_name):
    fig, ax = plt.subplots(figsize=(12, 8))

    def update(year):
        ax.clear()
        year_data = data[data['Year'] == year].nlargest(10, 'strongImpact')
        bars = ax.bar(year_data['Country_Name'], year_data['strongImpact'], color=plt.cm.tab20.colors)
        ax.set_xlabel('Country')
        ax.set_ylabel('Average Rating (Strong Impact)')
        ax.set_title(f"Strong Impact by Country ({year})")
        ax.set_xticklabels(year_data['Country_Name'], rotation=90)
        ax.legend(bars, year_data['Country_Name'], loc='upper left')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    years = data['Year'].unique()
    ani = FuncAnimation(fig, update, frames=years, repeat=False)
    ani.save(file_name, writer=PillowWriter(fps=2))

def plot_and_save_quality_by_country(data, file_name):
    fig, ax = plt.subplots(figsize=(12, 8))

    def update(year):
        ax.clear()
        year_data = data[data['Year'] == year].nlargest(10, 'average_quality_score')
        bars = ax.bar(year_data['Country_Name'], year_data['average_quality_score'], color=plt.cm.tab20.colors)
        ax.set_xlabel('Country')
        ax.set_ylabel('Average Quality Score')
        ax.set_title(f"Quality of Movies by Country (Top 100 Movies) ({year})")
        ax.set_xticklabels(year_data['Country_Name'], rotation=90)
        ax.legend(bars, year_data['Country_Name'], loc='upper left')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    years = data['Year'].unique()
    ani = FuncAnimation(fig, update, frames=years, repeat=False)
    ani.save(file_name, writer=PillowWriter(fps=2))

def perform_analysis(data_file, start_year=None, end_year=None):
    data = load_data(data_file)
    
    # Filter data based on years
    if start_year and end_year:
        data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
    
    # Weak and strong impact analysis
    weak_impact = calculate_weak_impact(data)
    strong_impact = calculate_strong_impact(data)
    
    # Quality of movies by country
    quality_by_country = analyze_quality_by_country(data, top_n=100)
    
    # Plot and save the animations
    plot_and_save_weak_impact(weak_impact, 'plots_src/analysis/weak_impact.gif')
    plot_and_save_strong_impact(strong_impact, 'plots_src/analysis/strong_impact.gif')
    plot_and_save_quality_by_country(quality_by_country, 'plots_src/analysis/quality_by_country.gif')

# Function call:
perform_analysis('merged_data/final_data.csv', start_year=1960, end_year=2020)
