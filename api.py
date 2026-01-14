import os
import sys

# ===== CRITICAL: Disable OpenMP threading BEFORE any imports =====
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['SKLEARN_THREADING_LAYER'] = 'sequential'
os.environ['OPENBLAS'] = 'USE_OPENMP=0'

# Ensure libgomp is not used
if 'LD_PRELOAD' not in os.environ:
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libgomp.so.1'

import pickle
import joblib
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set numpy to single-threaded
np.seterr(all='ignore')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "visa_processing_model.pkl")
PREPROCESS_PATH = os.path.join(BASE_DIR, "preprocessing_info.pkl")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESS_PATH):
        raise FileNotFoundError("Model or preprocessing info not found.")
    model = joblib.load(MODEL_PATH)
    with open(PREPROCESS_PATH, "rb") as f:
        prep = pickle.load(f)
    return model, prep


def build_feature_vector(prep, country, visa_type, application_date_str, processing_office=None):
    feature_names = prep["feature_names"]
    row = {c: 0 for c in feature_names}

    try:
        app_date = pd.to_datetime(application_date_str)
    except Exception:
        app_date = pd.to_datetime("today")

    application_month = int(app_date.month)
    season = "Peak" if application_month in [1, 2, 12] else "Off-Peak"

    if "application_month" in row:
        row["application_month"] = application_month
    if "country_avg" in row:
        row["country_avg"] = float(prep.get("country_avg", {}).get(country, prep.get("mean_processing_days", 0)))
    if "visa_avg" in row:
        row["visa_avg"] = float(prep.get("visa_avg", {}).get(visa_type, prep.get("mean_processing_days", 0)))

    country_col = f"country_{country}"
    if country_col in row:
        row[country_col] = 1

    visa_col = f"visa_type_{visa_type}"
    if visa_col in row:
        row[visa_col] = 1

    season_col = f"season_{season}"
    if season_col in row:
        row[season_col] = 1

    office_map = prep.get("office_map", {})
    mapped_office = processing_office or office_map.get(country, "Unknown")
    office_col = f"processing_office_{mapped_office}"
    if office_col in row:
        row[office_col] = 1

    df = pd.DataFrame([row], columns=feature_names).fillna(0)
    return df


def predict(model, prep, country, visa_type, application_date_str, processing_office=None):
    X = build_feature_vector(prep, country, visa_type, application_date_str, processing_office)
    pred = model.predict(X)[0]
    pred = max(0.0, float(pred))
    return round(pred, 1)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "message": "VisaAI Backend API is running"}, 200


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json()
        country = data.get("country", "Unknown")
        visa_type = data.get("visa_type", "Unknown")
        application_date = data.get("application_date", datetime.today().strftime("%Y-%m-%d"))
        processing_office = data.get("processing_office", None)
        
        model, prep = load_artifacts()
        days = predict(model, prep, country, visa_type, application_date, processing_office)
        
        return {
            "success": True,
            "country": country,
            "visa_type": visa_type,
            "application_date": application_date,
            "estimated_days": days
        }, 200
    except Exception as e:
        return {"success": False, "error": str(e)}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
