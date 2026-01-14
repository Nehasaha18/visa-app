"""
Prediction Script for Visa Processing Days
This script loads the trained model and makes predictions for website integration.
"""

import os
import numpy as np
import pandas as pd
import joblib
import pickle
from datetime import datetime

def load_model_and_preprocessing():
    """Load the trained model and preprocessing information."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_dir, "visa_processing_model.pkl")
    preprocessing_path = os.path.join(base_dir, "preprocessing_info.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please run Milestone3.py first.")
    
    if not os.path.exists(preprocessing_path):
        raise FileNotFoundError(f"Preprocessing file not found: {preprocessing_path}. Please run Milestone3.py first.")
    
    model = joblib.load(model_path)
    
    with open(preprocessing_path, 'rb') as f:
        preprocessing_info = pickle.load(f)
    
    return model, preprocessing_info

def predict_processing_days(application_date, country, visa_type):
    """
    Predict visa processing days based on user inputs.
    
    Parameters:
    -----------
    application_date : str or datetime
        Application date in format 'YYYY-MM-DD'
    country : str
        Country name (e.g., 'India', 'United States', etc.)
    visa_type : str
        Visa type (e.g., 'Student', 'Tourist', 'Work')
    
    Returns:
    --------
    dict : Dictionary containing prediction and details
    """
    # Load model and preprocessing info
    model, prep_info = load_model_and_preprocessing()
    
    # Convert application_date to datetime
    if isinstance(application_date, str):
        app_date = pd.to_datetime(application_date)
    else:
        app_date = application_date
    
    # Extract features
    application_month = app_date.month
    season = "Peak" if application_month in [1, 2, 12] else "Off-Peak"
    
    # Get processing office
    office_map = prep_info['office_map']
    processing_office = office_map.get(country, "Unknown")
    
    # Get country and visa averages
    country_avg = prep_info['country_avg'].get(country, prep_info['mean_processing_days'])
    visa_avg = prep_info['visa_avg'].get(visa_type, prep_info['mean_processing_days'])
    
    # Create feature vector
    feature_names = prep_info['feature_names']
    feature_dict = {name: 0 for name in feature_names}
    
    # Set numeric features
    feature_dict['application_month'] = application_month
    feature_dict['country_avg'] = country_avg
    feature_dict['visa_avg'] = visa_avg
    
    # Set categorical features (one-hot encoding)
    country_col = f"country_{country}"
    visa_col = f"visa_type_{visa_type}"
    season_col = f"season_{season}"
    office_col = f"processing_office_{processing_office}"
    
    if country_col in feature_dict:
        feature_dict[country_col] = 1
    if visa_col in feature_dict:
        feature_dict[visa_col] = 1
    if season_col in feature_dict:
        feature_dict[season_col] = 1
    if office_col in feature_dict:
        feature_dict[office_col] = 1
    
    # Convert to DataFrame with correct column order
    feature_df = pd.DataFrame([feature_dict])[feature_names]
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    prediction = max(0, round(prediction, 1))  # Ensure non-negative and round to 1 decimal
    
    return {
        'predicted_days': prediction,
        'application_date': app_date.strftime('%Y-%m-%d'),
        'country': country,
        'visa_type': visa_type,
        'season': season,
        'processing_office': processing_office,
        'model_type': prep_info['model_type']
    }

# Example usage
if __name__ == "__main__":
    # Test prediction
    try:
        result = predict_processing_days(
            application_date="2024-06-15",
            country="India",
            visa_type="Student"
        )
        
        print("="*50)
        print("VISA PROCESSING DAYS PREDICTION")
        print("="*50)
        print(f"Application Date: {result['application_date']}")
        print(f"Country: {result['country']}")
        print(f"Visa Type: {result['visa_type']}")
        print(f"Season: {result['season']}")
        print(f"Processing Office: {result['processing_office']}")
        print(f"\nPredicted Processing Days: {result['predicted_days']} days")
        print(f"Model Used: {result['model_type']}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease make sure you have run Milestone3.py first to train and save the model.")

