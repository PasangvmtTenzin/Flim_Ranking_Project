import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the datasets
cinematic_data_path = 'merged_data/merged_cinematic_data.csv'
economic_data_path = 'merged_data/population_economic_data.csv'

cinematic_data = pd.read_csv(cinematic_data_path)
economic_data = pd.read_csv(economic_data_path)

# Country code mapping 
country_code_mapping = {
    'AD': 'AND', 'AE': 'ARE', 'AF': 'AFG', 'AG': 'ATG', 'AI': 'AIA',
    'AL': 'ALB', 'AM': 'ARM', 'AN': 'ANT', 'AO': 'AGO', 'AQ': 'ATA',
    'AR': 'ARG', 'AS': 'ASM', 'AT': 'AUT', 'AU': 'AUS', 'AW': 'ABW',
    'AZ': 'AZE', 'BA': 'BIH', 'BB': 'BRB', 'BD': 'BGD', 'BE': 'BEL',
    'BF': 'BFA', 'BG': 'BGR', 'BH': 'BHR', 'BI': 'BDI', 'BJ': 'BEN',
    'BM': 'BMU', 'BN': 'BRN', 'BO': 'BOL', 'BR': 'BRA', 'BS': 'BHS',
    'BT': 'BTN', 'BW': 'BWA', 'BY': 'BLR', 'BZ': 'BLZ', 'CA': 'CAN',
    'CD': 'COD', 'CF': 'CAF', 'CG': 'COG', 'CH': 'CHE', 'CI': 'CIV',
    'CK': 'COK', 'CL': 'CHL', 'CM': 'CMR', 'CN': 'CHN', 'CO': 'COL',
    'CR': 'CRI', 'CU': 'CUB', 'CV': 'CPV', 'CW': 'CUW', 'CY': 'CYP',
    'CZ': 'CZE', 'DE': 'DEU', 'DJ': 'DJI', 'DK': 'DNK', 'DM': 'DMA',
    'DO': 'DOM', 'DZ': 'DZA', 'EC': 'ECU', 'EE': 'EST', 'EG': 'EGY',
    'ER': 'ERI', 'ES': 'ESP', 'ET': 'ETH', 'FI': 'FIN', 'FJ': 'FJI',
    'FO': 'FRO', 'FR': 'FRA', 'GA': 'GAB', 'GB': 'GBR', 'GD': 'GRD',
    'GE': 'GEO', 'GF': 'GUF', 'GH': 'GHA', 'GI': 'GIB', 'GL': 'GRL',
    'GM': 'GMB', 'GN': 'GIN', 'GP': 'GLP', 'GQ': 'GNQ', 'GR': 'GRC',
    'GT': 'GTM', 'GU': 'GUM', 'GW': 'GNB', 'GY': 'GUY', 'HK': 'HKG',
    'HN': 'HND', 'HR': 'HRV', 'HT': 'HTI', 'HU': 'HUN', 'ID': 'IDN',
    'IE': 'IRL', 'IL': 'ISR', 'IN': 'IND', 'IQ': 'IRQ', 'IR': 'IRN',
    'IS': 'ISL', 'IT': 'ITA', 'JM': 'JAM', 'JO': 'JOR', 'JP': 'JPN',
    'KE': 'KEN', 'KG': 'KGZ', 'KH': 'KHM', 'KM': 'COM', 'KN': 'KNA',
    'KP': 'PRK', 'KR': 'KOR', 'KW': 'KWT', 'KY': 'CYM', 'KZ': 'KAZ',
    'LA': 'LAO', 'LB': 'LBN', 'LC': 'LCA', 'LI': 'LIE', 'LK': 'LKA',
    'LR': 'LBR', 'LS': 'LSO', 'LT': 'LTU', 'LU': 'LUX', 'LV': 'LVA',
    'LY': 'LBY', 'MA': 'MAR', 'MC': 'MCO', 'MD': 'MDA', 'ME': 'MNE',
    'MG': 'MDG', 'MH': 'MHL', 'MK': 'MKD', 'ML': 'MLI', 'MM': 'MMR',
    'MN': 'MNG', 'MO': 'MAC', 'MQ': 'MTQ', 'MR': 'MRT', 'MT': 'MLT',
    'MU': 'MUS', 'MV': 'MDV', 'MW': 'MWI', 'MX': 'MEX', 'MY': 'MYS',
    'MZ': 'MOZ', 'NA': 'NAM', 'NC': 'NCL', 'NE': 'NER', 'NG': 'NGA',
    'NI': 'NIC', 'NL': 'NLD', 'NO': 'NOR', 'NP': 'NPL', 'NR': 'NRU',
    'NZ': 'NZL', 'OM': 'OMN', 'PA': 'PAN', 'PE': 'PER', 'PF': 'PYF',
    'PG': 'PNG', 'PH': 'PHL', 'PK': 'PAK', 'PL': 'POL', 'PT': 'PRT',
    'PW': 'PLW', 'PY': 'PRY', 'QA': 'QAT', 'RO': 'ROU', 'RS': 'SRB',
    'RU': 'RUS', 'RW': 'RWA', 'SA': 'SAU', 'SB': 'SLB', 'SC': 'SYC',
    'SD': 'SDN', 'SE': 'SWE', 'SG': 'SGP', 'SI': 'SVN', 'SK': 'SVK',
    'SL': 'SLE', 'SM': 'SMR', 'SN': 'SEN', 'SO': 'SOM', 'SR': 'SUR',
    'ST': 'STP', 'SV': 'SLV', 'SY': 'SYR', 'SZ': 'SWZ', 'TC': 'TCA',
    'TD': 'TCD', 'TG': 'TGO', 'TH': 'THA', 'TJ': 'TJK', 'TM': 'TKM',
    'TN': 'TUN', 'TO': 'TON', 'TR': 'TUR', 'TT': 'TTO', 'TV': 'TUV',
    'TZ': 'TZA', 'UA': 'UKR', 'UG': 'UGA', 'US': 'USA', 'UY': 'URY',
    'UZ': 'UZB', 'VA': 'VAT', 'VC': 'VCT', 'VE': 'VEN', 'VG': 'VGB',
    'VI': 'VIR', 'VN': 'VNM', 'VU': 'VUT', 'WS': 'WSM', 'YE': 'YEM',
    'ZA': 'ZAF', 'ZM': 'ZMB', 'ZW': 'ZWE'
}

# Apply the mapping to the cinematic data
cinematic_data['Country_Code'] = cinematic_data['region'].map(country_code_mapping)

# Drop rows where the country code mapping was not found
cinematic_data = cinematic_data.dropna(subset=['Country_Code'])

# Re-group by the new country code and aggregate the votes and quality scores
cinematic_data_grouped = cinematic_data.groupby('Country_Code').agg(
    total_votes=('numVotes', 'sum'),
    average_quality_score=('averageRating', 'mean')
).reset_index()

# Merge the Data
merged_data = pd.merge(cinematic_data_grouped, economic_data, on='Country_Code')

# Compute Ranks and Hegemony
# Calculate the ranks
merged_data['population_rank'] = merged_data['Population'].rank(ascending=False)
merged_data['gdp_rank'] = merged_data['GDP'].rank(ascending=False)
merged_data['gdp_per_capita_rank'] = merged_data['GDP_per_Capital'].rank(ascending=False)
merged_data['total_votes_rank'] = merged_data['total_votes'].rank(ascending=False)
merged_data['average_quality_rank'] = merged_data['average_quality_score'].rank(ascending=False)

# Compute hegemony
merged_data['strong_hegemony'] = merged_data['gdp_rank'] - merged_data['average_quality_rank']
merged_data['weak_hegemony'] = merged_data['gdp_rank'] - merged_data['total_votes_rank']
data = merged_data.sort_values(by=['Year'], ascending=True)

merged_data.to_csv('merged_data/final_data.csv')

# Filter out rows with negative values in the strong_hegemony column
filtered_data = data[data['strong_hegemony'] >= 0]

# Extract the top 10 countries by GDP for each year
top_10_gdp_countries = (data.groupby('Year', as_index=False, group_keys=False)
                        .apply(lambda x: x.nlargest(10, 'GDP')))

# Manually filter top 10 countries by total votes for each year
top_10_votes_countries_list = []
for year in data['Year'].unique():
    top_10_votes_countries_year = data[data['Year'] == year].nlargest(11, 'total_votes')
    top_10_votes_countries_list.append(top_10_votes_countries_year)
top_10_votes_countries = pd.concat(top_10_votes_countries_list)


# Filter top 5 countries with most hegemony
top_10_strong_hegemony_country = []


for year in merged_data['Year'].unique():
    # Compute hegemony
    merged_data['strong_hegemony'] = merged_data['gdp_rank'] - merged_data['average_quality_rank']
    merged_data['weak_hegemony'] = merged_data['gdp_rank'] - merged_data['total_votes_rank']
    top_10_hegemony_countries_year = merged_data[merged_data['Year'] == year].nlargest(5, 'strong_hegemony')
    top_10_strong_hegemony_country.append(top_10_hegemony_countries_year)

top_10_hegemony_countries = pd.concat(top_10_strong_hegemony_country)

# Function to handle annimation speed
def speed_animate(fig, animation_speed=1000):
    """
    Adjust the animation speed of a Plotly figure.

    Parameters:
    - fig: The Plotly figure object to be animated.
    - animation_speed: Duration in milliseconds between frames.
    """
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": animation_speed, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ]
        }]
    )
    return fig

# Bubble plot for GDP vs Total Votes with only top 10 countries
fig1 = px.scatter(top_10_gdp_countries, 
                 x='GDP', 
                 y='total_votes', 
                 animation_frame='Year', 
                 size='Population', 
                 color='gdp_rank', 
                 hover_name='Country_Code', 
                 log_x=True, 
                 size_max=80, 
                 range_x=[top_10_gdp_countries['GDP'].min(), top_10_gdp_countries['GDP'].max()], 
                 range_y=[top_10_gdp_countries['total_votes'].min(), top_10_gdp_countries['total_votes'].max()],
                 title='GDP vs Total Votes Over Year (Top 10 Countries)',
                 labels={'Year': 'Year'})
    
# fig1 = speed_animate(fig1, animation_speed=500)
#fig1.show()
#fig1.write_html('analysis/plots/flim_hegemony/gdp_vs_votes.html')

# Bubble plot for total votes vs average quality score with only top 10 countries
fig2 = px.scatter(top_10_votes_countries, 
                  x='total_votes', 
                  y='average_quality_score', 
                  animation_frame='Year', 
                  size='Population', 
                  color='Country_Code', 
                  hover_name='Country_Name', 
                  title='Total Votes vs Average Quality Score (Top 10 Countries)',
                  labels={'total_votes': 'Total Votes', 'average_quality_score': 'Average Quality Score', 'Year': 'Year'},
                  size_max=150, 
                  log_x=True, 
                  range_x=[top_10_votes_countries['total_votes'].min(), top_10_votes_countries['total_votes'].max()], 
                  range_y=[top_10_votes_countries['average_quality_score'].min(), top_10_votes_countries['average_quality_score'].max()])

# Display the plot
# fig2 = speed_animate(fig2, animation_speed=1000)
# fig2.show()
# fig2.write_html('analysis/plots/flim_hegemony/gdp_vs_avg_quality_score.html')


fig3 = px.scatter_matrix(
    top_10_hegemony_countries,
    dimensions=['population_rank', 'gdp_rank', 'gdp_per_capita_rank', 'total_votes_rank', 'average_quality_rank'],
    color='Country_Code',
    title='Scatter Plot Matrix of Ranks',
    size_max=5
)

fig3.update_layout(
    title=dict(x=0.5),
    width=1700,
    height=850,
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode='closest'
)

fig3.update_xaxes(tickangle=45, automargin=True)
fig3.update_yaxes(automargin=True)

#fig3.show()
fig3.write_html('analysis/plots/flim_hegemony/Scatter_Plot_Matrix.html')


# top 10 hegenom countries plot
fig5 = px.line(top_10_hegemony_countries, x='Year', y='strong_hegemony', color='Country_Code',
               title='Strong Hegemony Over Year',
               labels={'Year': 'Year', 'strong_hegemony': 'Hegemony'})

fig5.update_traces(mode='lines+markers')
#fig5.show()
#fig5.write_html('analysis/plots/flim_hegemony/strong_hegenomy_countries.html')


# Weak hegonomy vs strong hegenomy country over year
fig6 = px.scatter(filtered_data, x='weak_hegemony', y='strong_hegemony', size='strong_hegemony', color='Country_Code',
                  hover_name='Country_Code', animation_frame='Year',
                  title='Weak Hegemony vs Strong Hegemony',
                  labels={'weak_hegemony': 'Weak Hegemony', 'strong_hegemony': 'Strong Hegemony', 'gdp_rank': 'GDP Rank'},
                  size_max=80)

fig6.update_layout(showlegend=False)  # Hides the legend for better animation clarity
#fig6.show()
#fig6.write_html('analysis/plots/flim_hegemony/hegenomy_weak_vs_strong.html')

