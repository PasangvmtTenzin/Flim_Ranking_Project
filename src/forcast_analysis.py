from data_loader import load_csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Function to clean data
def clean_data(data):
    # Drop unnecessary columns
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    
    # Remove duplicates based on Country_Name and Year
    data = data.drop_duplicates(subset=['Country_Name', 'Year'])
    
    # Create additional features
    data['CIS'] = data['average_quality_score'] * data['total_votes']
    data['GDP_Normalized_CIS'] = data['CIS'] / data['GDP']
    data['Population_Normalized_CIS'] = data['CIS'] / data['Population']
    data['OIS'] = data['CIS']  # Assuming OIS is the same as CIS for now
    
    return data

# Define Features and Targets
features = ['GDP', 'Population', 'CIS', 'GDP_Normalized_CIS', 'Population_Normalized_CIS']
targets = ['OIS', 'strong_hegemony', 'weak_hegemony']

# Function to make forecasts
def make_forecasts(models_to_use, scalers, features, start_year, end_year, data, economic_growth_rate, targets):
    forecasts = []
    
    # Generate years to forecast
    years = list(range(start_year, end_year + 1))
    
    for year in years:
        forecast_row = {}

        # Implement forecasting strategies for each feature
        for feature in features:
            if feature == 'GDP':
                # Forecast GDP using economic growth rate
                forecast_row['GDP'] = data['GDP'].iloc[-1] * (1 + economic_growth_rate / 100)
            elif feature == 'Population':
                # Example: Forecast Population using linear regression model (assuming it's already trained)
                X_train = data[['Year']]
                y_train = data['Population']
                model = LinearRegression().fit(X_train, y_train)
                forecast_row['Population'] = model.predict([[year]])[0]
            elif feature in data.columns:
                # Use historical average for features that exist in data
                forecast_row[feature] = data[feature].mean()
            else:
                # Handle missing features gracefully
                forecast_row[feature] = None

        # Create a dummy row DataFrame with forecast values
        dummy_row = pd.DataFrame([forecast_row], columns=features)

        for target in targets:
            if target in models_to_use:
                # Load best model for the target
                best_model = models_to_use[target]['Best_Model']['Pipeline']
                
                # Predict using the best model
                X_forecast = dummy_row[features]  # Ensure features are in the correct order
                X_forecast_scaled = scalers[target].transform(X_forecast)
                
                # Convert X_forecast_scaled to DataFrame with column names
                X_forecast_scaled_df = pd.DataFrame(X_forecast_scaled, columns=features)
                
                predicted_value = best_model.predict(X_forecast_scaled_df)
                
                forecast_row[f'{target}_Prediction'] = predicted_value[0]
            else:
                # If model for target not found, assign None to prediction columns
                forecast_row[f'{target}_Prediction'] = None
        
        # Add year information
        forecast_row['Year'] = year
        
        # Append forecast to forecasts list
        forecasts.append(forecast_row)
    
    # Create DataFrame from forecasts list
    forecast_data = pd.DataFrame(forecasts)
    
    return forecast_data

# Example usage
if __name__ == "__main__":
    # Define Features and Targets
    features = ['GDP', 'Population', 'CIS', 'GDP_Normalized_CIS', 'Population_Normalized_CIS']
    targets = ['OIS', 'strong_hegemony', 'weak_hegemony']
    
    # Load data
    data = load_csv('../merged_data/final_data.csv')
    
    # Clean data
    data = clean_data(data)
    
    # Ensure data has necessary columns for predictions
    required_columns = ['Model', 
                        'OIS_Prediction', 
                        'strong_hegemony_Prediction', 
                        'weak_hegemony_Prediction']
    
    for col in required_columns:
        if col not in data.columns:
            data[col] = None
    
    # Load saved models and scalers
    models_to_use = {}
    scalers = {}
    for target in targets:
        best_model = joblib.load(f"src/model/{target}_model.pkl")
        scaler = joblib.load(f"src/scaler/{target}_scaler.pkl")
        models_to_use[target] = {'Best_Model': {'Pipeline': best_model}}
        scalers[target] = scaler
    
    # Economic growth rate for GDP forecasting
    economic_growth_rate = 3.09  # Update this as needed
    
    # Make Forecasts from 2022 to 2040
    forecast_data = make_forecasts(models_to_use, scalers, features, 2018, 2045, data, economic_growth_rate, targets)
    
    # Save Forecast Data
    #forecast_data.to_csv('src/forecast_data.csv', index=False)