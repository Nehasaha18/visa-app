import os
import pickle
import joblib
from datetime import datetime
try:
    from flask import Flask, request, render_template_string, redirect, url_for
    HAS_FLASK = True
except Exception:
    HAS_FLASK = False
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "visa_processing_model.pkl")
PREPROCESS_PATH = os.path.join(BASE_DIR, "preprocessing_info.pkl")


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESS_PATH):
        raise FileNotFoundError("Model or preprocessing info not found. Run Milestone3 first.")
    model = joblib.load(MODEL_PATH)
    with open(PREPROCESS_PATH, "rb") as f:
        prep = pickle.load(f)
    return model, prep


def build_feature_vector(prep, country, visa_type, application_date_str, processing_office=None):
    feature_names = prep["feature_names"]

    # Base row: zeros
    row = {c: 0 for c in feature_names}

    # Parse date
    try:
        app_date = pd.to_datetime(application_date_str)
    except Exception:
        app_date = pd.to_datetime("today")

    application_month = int(app_date.month)
    season = "Peak" if application_month in [1, 2, 12] else "Off-Peak"

    # Numeric features
    if "application_month" in row:
        row["application_month"] = application_month
    if "country_avg" in row:
        row["country_avg"] = float(prep.get("country_avg", {}).get(country, prep.get("mean_processing_days", 0)))
    if "visa_avg" in row:
        row["visa_avg"] = float(prep.get("visa_avg", {}).get(visa_type, prep.get("mean_processing_days", 0)))

    # One-hot country
    country_col = f"country_{country}"
    if country_col in row:
        row[country_col] = 1

    # One-hot visa type
    visa_col = f"visa_type_{visa_type}"
    if visa_col in row:
        row[visa_col] = 1

    # Season
    season_col = f"season_{season}"
    if season_col in row:
        row[season_col] = 1

    # Processing office (use mapping if not provided)
    office_map = prep.get("office_map", {})
    mapped_office = processing_office or office_map.get(country, "Unknown")
    office_col = f"processing_office_{mapped_office}"
    if office_col in row:
        row[office_col] = 1

    # Ensure order and return DataFrame
    df = pd.DataFrame([row], columns=feature_names).fillna(0)
    return df


def predict(model, prep, country, visa_type, application_date_str, processing_office=None):
    X = build_feature_vector(prep, country, visa_type, application_date_str, processing_office)
    pred = model.predict(X)[0]
    # sanity: clip negatives and round to 1 decimal
    pred = max(0.0, float(pred))
    return round(pred, 1)


INDEX_HTML = """
<h2>Visa Processing Time Estimator</h2>
<form method="post" action="/predict">
  Country: <input name="country" value="India"><br>
  Visa Type: <input name="visa_type" value="Student"><br>
  Application Date (YYYY-MM-DD): <input name="application_date" value="2024-09-01"><br>
  Processing Office (optional): <input name="processing_office"><br>
  <input type="submit" value="Predict">
</form>
"""

# Create Flask app (required for gunicorn)
APP = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'), static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))

if HAS_FLASK:
    @APP.route("/health", methods=["GET"])
    def health():
        return {"status": "ok", "message": "VisaAI API is running"}, 200

    @APP.route("/", methods=["GET"])
    def index():
        try:
            html_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'index.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                return render_template_string(f.read())

        except Exception as e:
            return f"<h1>Error loading page: {e}</h1>", 500


    @APP.route("/predict", methods=["POST", "GET"])
    def predict_route():
        try:
            country = request.form.get("country") or request.args.get("country", "Unknown")
            visa_type = request.form.get("visa_type") or request.args.get("visa_type", "Unknown")
            application_date = request.form.get("application_date") or request.args.get("application_date", datetime.today().strftime("%Y-%m-%d"))
            processing_office = request.form.get("processing_office") or request.args.get("processing_office", None)
            
            model, prep = load_artifacts()
            days = predict(model, prep, country, visa_type, application_date, processing_office)
        except Exception as e:
            return f"<html><body style='font-family:Inter, Poppins, sans-serif;background:#07104a;color:#eaf0ff;padding:20px'><h2>Error during prediction:</h2><p>{str(e)}</p><p><a href='/'>Back</a></p></body></html>", 500
        
        return render_template_string(f"<html><body style='font-family:Inter, Poppins, sans-serif;background:#07104a;color:#eaf0ff;display:flex;align-items:center;justify-content:center;height:100vh'><div style='background:rgba(255,255,255,0.02);padding:24px;border-radius:12px;box-shadow:0 20px 40px rgba(0,0,0,0.6)'><h2>Estimated processing days: {days}</h2><p><a href='/'>Back</a></p></div></body></html>")


def run_tests():
    model, prep = load_artifacts()
    samples = [
        {"country": "India", "visa_type": "Student", "application_date": "2024-09-02", "office": "New Delhi"},
        {"country": "United Kingdom", "visa_type": "Work", "application_date": "2024-04-29", "office": "London"},
        {"country": "Germany", "visa_type": "Tourist", "application_date": "2023-11-27", "office": "Berlin"},
        {"country": "India", "visa_type": "Student", "application_date": "2024-12-15", "office": "New Delhi"},
    ]
    print("Running sample predictions:")
    for s in samples:
        days = predict(model, prep, s["country"], s["visa_type"], s["application_date"], s.get("office"))
        print(f"{s['country']} | {s['visa_type']} | {s['application_date']} -> {days} days")


if __name__ == "__main__":
    import sys
    import os
    import traceback
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "runserver":
        if not HAS_FLASK:
            print("Flask is not available in this environment. Install Flask to run the server.")
            sys.exit(1)
        try:
            # Bind to 0.0.0.0 on Heroku port (or localhost for development)
            port = int(os.environ.get("PORT", 5000))
            print(f"Starting Flask server on port {port}...")
            APP.run(host="0.0.0.0", port=port, debug=False)
        except Exception as e:
            print(f"Error starting Flask server: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        run_tests()
