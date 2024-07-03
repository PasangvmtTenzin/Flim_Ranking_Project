from data_loader import load_csv
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
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

# Function to train models
def train_models(data, features, targets, params):
    results = {}
    
    for target in targets:
        X = data[features]
        y = data[target]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models and parameters for GridSearchCV
        models = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': params.get('Linear Regression', {})
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor(),
                'params': params.get('Decision Tree', {'max_depth': [5, 10, 15]})
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(),
                'params': params.get('Gradient Boosting', {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]})
            },
            'XGBoost': {
                'model': XGBRegressor(objective='reg:squarederror'),
                'params': params.get('XGBoost', {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]})
            }
        }
        
        target_results = {}
        
        for model_name, model_info in models.items():
            # Perform GridSearchCV
            grid_search = GridSearchCV(estimator=model_info['model'], param_grid=model_info['params'], 
                                       scoring='neg_root_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
            grid_search.fit(X_scaled, y)
            
            # Get best model from GridSearchCV
            best_model = grid_search.best_estimator_
            
            # Cross-validation results
            cv_results = cross_val_score(best_model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
            cv_rmse_mean = np.mean(np.sqrt(-cv_results))
            cv_rmse_std = np.std(np.sqrt(-cv_results))
            
            # Save results
            target_results[model_name] = {
                'Model': best_model,
                'CV_RMSE_Mean': cv_rmse_mean,
                'CV_RMSE_Std': cv_rmse_std
            }
        
        results[target] = target_results
    
    return results, scaler

# Example function to make forecasts
def make_forecasts(models, scaler, features, start_year, end_year):
    forecast_data = []
    
    # Generate forecast years
    forecast_years = range(start_year, end_year + 1)
    
    for year in forecast_years:
        # Create dummy row for prediction (assuming missing data scenario)
        dummy_data = pd.DataFrame({'Year': [year], 'total_votes': [None], 'GDP': [None], 'Population': [None]})
        dummy_row = pd.DataFrame([[year] + [0] * len(features)], columns=['Year'] + features)
        
        # Scale dummy row using the fitted scaler
        scaled_row = pd.DataFrame(scaler.transform(dummy_row[features]), columns=features)
        
        for target, target_models in models.items():
            for model_name, model_info in target_models.items():
                model = model_info['Model']
                
                # Predict using scaled dummy row
                predicted_value = model.predict(scaled_row.values.reshape(1, -1))
                
                # Calculate CIS based on predicted_value and dummy_data
                CIS = None
                if not dummy_data.empty and not pd.isna(dummy_data['total_votes'].values[0]):
                    CIS = predicted_value * dummy_data['total_votes'].values[0]
                
                # Prepare forecast entry
                forecast_entry = {
                    'Year': year,
                    'Target': target,
                    'Model': model_name,
                    'CIS': CIS,
                    'GDP_Normalized_CIS': CIS / dummy_data['GDP'].values[0] if CIS is not None else None,
                    'Population_Normalized_CIS': CIS / dummy_data['Population'].values[0] if CIS is not None else None,
                    'OIS': CIS,
                    'strong_hegemony': predicted_value[0],
                    'weak_hegemony': predicted_value[0],
                    'Prediction': predicted_value[0]
                }
                
                forecast_data.append(forecast_entry)
    
    forecast_data_df = pd.DataFrame(forecast_data)
    return forecast_data_df

# Example function to save best models
def save_best_model(results, model_path='src/best_model.pkl'):
    best_models = {}
    for target, target_results in results.items():
        best_model = min(target_results.items(), key=lambda x: x[1]['CV_RMSE_Mean'])  # Use CV_RMSE_Mean for finding best model
        best_models[target] = best_model[1]['Model']
        joblib.dump(best_model[1]['Model'], f"{model_path}_{target}.pkl")
    
    return best_models

# Example function to plot model performance
def plot_model_performance(results):
    all_data = []
    for target, target_results in results.items():
        for model_name, metrics in target_results.items():
            all_data.append({
                'Target': target,
                'Model': model_name,
                'CV_RMSE_Mean': metrics['CV_RMSE_Mean'],
                'CV_RMSE_Std': metrics['CV_RMSE_Std']
            })
    
    df = pd.DataFrame(all_data)
    fig = px.line(df, x='Model', y='CV_RMSE_Mean', color='Target', error_y='CV_RMSE_Std', markers=True, title='Model Performance')
    fig.write_html('plots_src/model_performance.html')

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
    
    # Define Features and Targets
    features = ['GDP', 'Population', 'CIS', 'GDP_Normalized_CIS', 'Population_Normalized_CIS']
    targets = ['OIS', 'strong_hegemony', 'weak_hegemony']
    
    # Define Model Parameters (removed params dictionary)
    params = {
    'Linear Regression': {},
    'Decision Tree': {'max_depth': [5, 10, 15]},
    'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}
    }
    
    # Train Models
    results, scaler = train_models(data, features, targets, params)
    
    # Print Results (modified for CV results)
    for target, target_results in results.items():
        print(f"\nResults for {target}:")
        for model_name, result in target_results.items():
            print(f"{model_name}: CV_RMSE_Mean={result['CV_RMSE_Mean']}, CV_RMSE_Std={result['CV_RMSE_Std']}")
    
    # Save Best Models
    save_best_model(results)
    
    # Make Forecasts from 1960 to 2035
    models_to_use = {target: {model_name: info for model_name, info in target_results.items()} for target, target_results in results.items()}
    forecast_data = make_forecasts(models_to_use, scaler, features, 1960, 2035)
    
    # Save Forecast Data
    forecast_data.to_csv('src/forecast_data.csv', index=False)
    
    # Update data with best predictions
    for _, row in forecast_data.iterrows():
        year, target, model, prediction = row['Year'], row['Target'], row['Model'], row['Prediction']
        data.loc[(data['Year'] == year) & (data['Model'] == model), f'Best_{target}_Prediction'] = prediction
    
    # Save updated data
    data.to_csv('src/updated_data.csv', index=False)
    
    # Plot Model Performance
    plot_model_performance(results)
