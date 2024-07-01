import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

file_path = 'merged_data/final_data.csv'

merged_data = pd.read_csv(file_path)

# Filter top 5 countries with most hegemony
top_10_strong_hegemony_country = []


for year in merged_data['Year'].unique():
    # Compute hegemony
    merged_data['strong_hegemony'] = merged_data['gdp_rank'] - merged_data['average_quality_rank']
    merged_data['weak_hegemony'] = merged_data['gdp_rank'] - merged_data['total_votes_rank']
    top_10_hegemony_countries_year = merged_data[merged_data['Year'] == year].nlargest(5, 'strong_hegemony')
    top_10_strong_hegemony_country.append(top_10_hegemony_countries_year)

top_10_hegemony_countries = pd.concat(top_10_strong_hegemony_country)

# Calculate the correlation coefficient
correlation = top_10_hegemony_countries[['weak_hegemony', 'strong_hegemony']].corr().iloc[0, 1]

sns.set(style="whitegrid")
# Create a scatter plot with a regression line
plt.figure(figsize=(10, 6))
plt.scatter(top_10_hegemony_countries['weak_hegemony'], 
            top_10_hegemony_countries['strong_hegemony'],
            alpha=0.3, color='blue')
plt.title(f'Correlation between Weak and Strong Hegemony (Correlation: {correlation:.2f})', fontsize=15, fontweight='bold')

sns.regplot(x='weak_hegemony', y='strong_hegemony', data=top_10_hegemony_countries, scatter=False, color='red')

# Add labels
plt.xlabel('Weak Hegemony')
plt.ylabel('Strong Hegemony')

plt.xticks([-2000, 0, 2000, 4000, 6000],
           ['-2k', '0', '2k', '4k', '6k'])
plt.yticks([0, 2000, 4000, 6000, 8000, 10000],
           ['0', '2k', '4k', '6k', '8k', '10k'])

plt.grid(True, linestyle='--', color='grey', linewidth=0.5)
plt.tight_layout()

# Show the plot
#plt.show()
plt.savefig('analysis/plots/flim_hegemony/correlation.png')