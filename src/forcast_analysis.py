from data_loader import load_csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import numpy as np

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
def make_forecasts(models_to_use, scalers, features, start_year, end_year, data):
    forecasts = []
    
    # Generate years to forecast
    years = list(range(start_year, end_year + 1))
    
    for year in years:
        forecast_row = {}

        for feature in features:
            forecast_row[feature] = data[feature].mean()  # Replace with appropriate forecasting logic

        # Create a dummy row DataFrame with forecast values
        dummy_row = pd.DataFrame(forecast_row, index=[0])

        for target, model_dict in models_to_use.items():
            for model_name, model_info in model_dict.items():
                preprocessor = scalers[target]  # Get the preprocessor (ColumnTransformer)
                model = model_info['Model']  # Get the model from the model_info
                
                # Transform the dummy row using the preprocessor
                scaled_row = preprocessor.transform(dummy_row[features])  # No need to convert to DataFrame
                
                # Predict using the model
                predicted_value = model.predict(scaled_row)
                forecast_row[f'{model_name}_Prediction'] = predicted_value[0]  # Assuming single prediction output
        
        # Add year information
        forecast_row['Year'] = year
        
        # Append forecast to forecasts list
        forecasts.append(forecast_row)
    
    # Create DataFrame from forecasts list
    forecast_data = pd.DataFrame(forecasts)
    
    return forecast_data

# Example usage
if __name__ == "__main__":
    # Load data
    data = load_csv('../merged_data/final_data.csv')
    
    # Clean data
    data = clean_data(data)
    
    # Ensure data has necessary columns for predictions
    required_columns = ['Model', 'Best_OIS_Prediction', 'Best_strong_hegemony_Prediction', 'Best_weak_hegemony_Prediction']
    for col in required_columns:
        if col not in data.columns:
            data[col] = None
    
    # Load saved models and scalers
    models_to_use = {}
    scalers = {}
    for target in targets:
        model = joblib.load(f"src/{target}_model.pkl")
        models_to_use[target] = {'Best_Model': {'Model': model}}
        scalers[target] = joblib.load(f"src/{target}_scaler.pkl")
    
    # Make Forecasts from 2022 to 2040
    forecast_data = make_forecasts(models_to_use, scalers, features, 2022, 2040, data)
    
    # Save Forecast Data
    forecast_data.to_csv('src/forecast_data.csv', index=False)
