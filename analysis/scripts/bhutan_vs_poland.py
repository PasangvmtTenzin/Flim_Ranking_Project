import pandas as pd 
import matplotlib.pyplot as plt
file_path = 'merged_data/final_data.csv'

data = pd.read_csv(file_path)

# Filter data for Poland and Bhutan
poland_data = data[data['Country_Name'] == 'Poland']
bhutan_data = data[data['Country_Name'] == 'Bhutan']

# Select relevant columns
relevant_columns = ['Year', 'strong_hegemony', 'weak_hegemony']
poland_data_filtered = poland_data[relevant_columns]
bhutan_data_filtered = bhutan_data[relevant_columns]

# Plotting the strong hegemony over the years
plt.figure(figsize=(14, 7))

plt.plot(poland_data_filtered['Year'], poland_data_filtered['strong_hegemony'], label='Poland - Strong Hegemony', color='blue')
plt.plot(bhutan_data_filtered['Year'], bhutan_data_filtered['strong_hegemony'], label='Bhutan - Strong Hegemony', color='red')

plt.xlabel('Year')
plt.ylabel('Strong Hegemony')
plt.yticks([-7500, -5000, -2500, 0, 2500, 5000, 7500, 10000],
           ['-7.5k', '-5k', '-2.5k', '0', '2.5k', '5k', '7.5k', '10k'])
plt.title('Strong Hegemony Comparison between Poland and Bhutan')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('analysis/plots/bhutan_vs_poland/strong_hegemony_bt_vs_pl.png')

# Plotting the weak hegemony over the years
# plt.figure(figsize=(14, 7))

plt.plot(poland_data_filtered['Year'], poland_data_filtered['weak_hegemony'], label='Poland - Weak Hegemony', color='blue')
plt.plot(bhutan_data_filtered['Year'], bhutan_data_filtered['weak_hegemony'], label='Bhutan - Weak Hegemony', color='red')

plt.xlabel('Year')
plt.ylabel('Weak Hegemony')
plt.yticks([0, 500, 1500, 2000, 2500, 3000],['0', '0.5k', '1.5k', '2k', '2.5k', '3k'])
plt.title('Weak Hegemony Comparison between Poland and Bhutan')
plt.legend()
plt.grid(True)
#plt.show()
# plt.savefig('analysis/plots/bhutan_vs_poland/weak_hegenomy_bt_vs_pl.png')