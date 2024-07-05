from data_loader import load_csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
import numpy as np
import plotly.express as px

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

# Function to train models
def train_models(data, features, targets, params):
    results_train = {}
    scalers = {}

    for target in targets:
        X = data[features]
        y = data[target]

        # Standardize features using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), features)
            ]
        )
        
        # Fit the preprocessor on X
        X_train = pd.DataFrame(preprocessor.fit_transform(X), columns=features)
        
        # Save the scaler for this target
        joblib.dump(preprocessor, f"src/{target}_scaler.pkl")
        scalers[target] = preprocessor
        
        # Split data into training and test/validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models and parameters for GridSearchCV
        models = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': params.get('Linear Regression', {})
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor(),
                'params': params.get('Decision Tree', {'regressor__max_depth': [5, 10, 15]})
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(),
                'params': params.get('Gradient Boosting', {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.1, 0.05]})
            },
            'XGBoost': {
                'model': XGBRegressor(objective='reg:squarederror'),
                'params': params.get('XGBoost', {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.1, 0.05]})
            }
        }
        
        target_results = {}
        
        for model_name, model in models.items():
            # Create a pipeline with the preprocessor and the model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model['model'])
            ])
            
            # Perform GridSearchCV on the pipeline
            grid_search = GridSearchCV(estimator=pipeline, param_grid=model['params'], 
                                       scoring='neg_root_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Get best model from GridSearchCV
            best_model = grid_search.best_estimator_
            
            # Cross-validation results on training data
            cv_results = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
            cv_rmse_mean = np.mean(np.sqrt(-cv_results))
            cv_rmse_std = np.std(np.sqrt(-cv_results))
            
            # Evaluate on test/validation data
            y_pred_test = best_model.predict(X_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Save results for the best model
            target_results[model_name] = {
                'Model': best_model,
                'CV_RMSE_Mean': cv_rmse_mean,
                'CV_RMSE_Std': cv_rmse_std,
                'RMSE': rmse_test
            }
        
        results_train[target] = target_results

    return results_train, scalers

# Function to save best models
def save_best_model(results, model_path='src/'):
    best_models = {}
    for target, target_results in results.items():
        best_model = min(target_results.items(), key=lambda x: x[1]['CV_RMSE_Mean'])  # Use CV_RMSE_Mean for finding best model
        best_models[target] = best_model[1]['Model']
        joblib.dump(best_model[1]['Model'], f"{model_path}{target}_model.pkl")
    
    return best_models

# Function to plot model performance
def plot_model_performance(results_train):
    all_data = []
    for target, target_results in results_train.items():
        for model_name, metrics in target_results.items():
            all_data.append({
                'Target': target,
                'Model': model_name,
                'Metric': 'CV_RMSE_Mean',
                'Value': metrics['CV_RMSE_Mean']
            })
    
    df = pd.DataFrame(all_data)
    
    fig = px.bar(df, x='Model', y='Value', color='Target', barmode='group', title='Model Performance by Target')
    fig.update_layout(yaxis_type='log')
    fig.write_html('plots_src/model_performance.html')

# Function to plot combined performance (train and test/validation)
def plot_combined_performance(results_train, results_test):
    all_data = []
    
    # Append training (cross-validation) data
    for target, target_results in results_train.items():
        for model_name, metrics in target_results.items():
            if 'CV_RMSE_Mean' in metrics:  # Using cross-validation RMSE mean for training
                all_data.append({
                    'Target': target,
                    'Model': model_name,
                    'Metric': 'CV_RMSE_Mean',
                    'Value': metrics['CV_RMSE_Mean'],
                    'Dataset': 'Training'
                })
    
    # Create DataFrame for plotting
    df = pd.DataFrame(all_data)
    
    # Plot model performance
    fig = px.bar(df, x='Model', y='Value', color='Target', barmode='group', facet_col='Dataset', title='Combined Model Performance')
    fig.update_layout(yaxis_type='log')
    fig.write_html('plots_src/combined_performance.html')

# Main Function
if __name__ == "__main__":
    # Load data
    data = load_csv('../merged_data/final_data.csv')
    
    # Clean data
    data = clean_data(data)
    
    # Define Model Parameters
    params = {
        'Linear Regression': {},
        'Decision Tree': {'regressor__max_depth': [5, 10, 15]},
        'Gradient Boosting': {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.1, 0.05]},
        'XGBoost': {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.1, 0.05]}
    }
    
    # Train Models
    results_train, scalers = train_models(data, features, targets, params)
    
    # Print Results (modified for CV results)
    for target, target_results in results_train.items():
        print(f"\nResults for {target}:")
        for model_name, result in target_results.items():
            print(f"{model_name}: CV_RMSE_Mean={result['CV_RMSE_Mean']}, CV_RMSE_Std={result['CV_RMSE_Std']}, RMSE={result['RMSE']}")
    
    # Plot Model Performance
    plot_model_performance(results_train)
    
    # Save Best Models and Scalers
    save_best_model(results_train)
    
    # Evaluate on Test/Validation Set and Plot Combined Performance
    results_test = {}
    for target, target_results in results_train.items():
        target_test_results = {}
        for model_name, result in target_results.items():
            # Use the same model trained on X_train for evaluation on X_test
            model = result['Model']
            
            # Split data into training and test/validation sets
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Transform X_test using the preprocessor
            X_test_df = pd.DataFrame(X_test, columns=features)  # Ensure X_test is DataFrame
            X_test_transformed = scalers[target].transform(X_test_df)
            
            # Convert X_test_transformed to DataFrame
            X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=features)
            
            # Predict on X_test_transformed_df
            y_pred_test = model.predict(X_test_transformed_df)
            
            # Calculate RMSE on y_test and y_pred_test
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Save test results
            target_test_results[model_name] = {
                'Model': model,
                'RMSE': rmse_test
            }
        
        results_test[target] = target_test_results
    
    # Plot Combined Performance (Training vs Test/Validation)
    plot_combined_performance(results_train, results_test)
