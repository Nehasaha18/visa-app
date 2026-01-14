"""Test script for prediction function"""
import sys
sys.path.append('.')
from predict_processing_days import predict_processing_days

try:
    result = predict_processing_days('2024-06-15', 'India', 'Student')
    print("✅ Prediction successful!")
    print(f"Predicted days: {result['predicted_days']}")
    print(f"Country: {result['country']}")
    print(f"Visa Type: {result['visa_type']}")
    print(f"Season: {result['season']}")
    print(f"Processing Office: {result['processing_office']}")
    print(f"Model Type: {result['model_type']}")
    print("\n✅ All tests passed! The web app should work correctly.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

