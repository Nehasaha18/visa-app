# IMPORT PACKAGES
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

pd.set_option("display.max_columns", None)

# LOAD FULL VISA DATASET FROM CSV (visa_dataset.csv created in Milestone 1)
print("\n===== MILESTONE 3: PREDICTIVE MODELING =====\n")
csv_path = os.path.join(os.path.dirname(__file__), "..", "visa_dataset.csv")
df = pd.read_csv(csv_path)
print("Original DataFrame loaded from visa_dataset.csv:\n", df.head())

# HANDLE MISSING VALUES
# Fill missing dates with mode
df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])
df["application_date"] = df["application_date"].fillna(df["application_date"].mode()[0])
df["decision_date"] = df["decision_date"].fillna(df["decision_date"].mode()[0])

# Fill missing categorical values with 'Unknown'
df["country"] = df["country"].fillna("Unknown")
df["visa_type"] = df["visa_type"].fillna("Unknown")

# CALCULATE PROCESSING DAYS
df["processing_days"] = (df["decision_date"] - df["application_date"]).dt.days
# Any negative processing days -> set as NaN
df.loc[df["processing_days"] < 0, "processing_days"] = np.nan

# ADD PROCESSING OFFICE (based on country)
office_map = {
    "India": "New Delhi",
    "United States": "Washington DC",
    "United Kingdom": "London",
    "Canada": "Ottawa",
    "Australia": "Canberra",
    "Germany": "Berlin",
    "France": "Paris",
    "Japan": "Tokyo",
    "Brazil": "Brasilia",
    "Italy": "Rome",
    "China": "Beijing",
    "Netherlands": "Amsterdam",
    "Spain": "Madrid",
    "Mexico": "Mexico City",
    "South Korea": "Seoul",
    "Unknown": "Unknown"
}
df["processing_office"] = df["country"].map(office_map).fillna("Unknown")

# FEATURE ENGINEERING
# Application month
df["application_month"] = df["application_date"].dt.month

# Season: Peak vs Off-Peak
df["season"] = df["application_month"].apply(lambda x: "Peak" if x in [1,2,12] else "Off-Peak")

# Country average processing days
country_avg = df.groupby("country")["processing_days"].mean()
df["country_avg"] = df["country"].map(country_avg)
# Fill NaN in country_avg with overall mean
df["country_avg"] = df["country_avg"].fillna(df["processing_days"].mean())

# Visa type average processing days
visa_avg = df.groupby("visa_type")["processing_days"].mean()
df["visa_avg"] = df["visa_type"].map(visa_avg)
# Fill NaN in visa_avg with overall mean
df["visa_avg"] = df["visa_avg"].fillna(df["processing_days"].mean())

print("\nDataFrame after feature engineering:\n", df.head())

# ENCODING CATEGORICAL FEATURES
df_encoded = pd.get_dummies(df, columns=["country", "visa_type", "season", "processing_office"], drop_first=True)
print("\nEncoded DataFrame ready for ML:\n", df_encoded.head())

# PREPARE DATA FOR MODELING
# Drop rows with missing processing_days (target variable)
df_ml = df_encoded.dropna(subset=["processing_days"])

X = df_ml.drop(columns=["processing_days", "application_date", "decision_date"])
y = df_ml["processing_days"]

# Fill any remaining NaN values in features with 0
X = X.fillna(0)

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


# MODEL 1: LINEAR REGRESSION (BASELINE)

print("\n" + "="*50)
print("MODEL 1: LINEAR REGRESSION (BASELINE)")
print("="*50)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"MAE: {mae_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R² Score: {r2_lr:.4f}")


# MODEL 2: RANDOM FOREST REGRESSOR

print("\n" + "="*50)
print("MODEL 2: RANDOM FOREST REGRESSOR")
print("="*50)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"MAE: {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R² Score: {r2_rf:.4f}")


# MODEL 3: GRADIENT BOOSTING REGRESSOR

print("\n" + "="*50)
print("MODEL 3: GRADIENT BOOSTING REGRESSOR")
print("="*50)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)

print(f"MAE: {mae_gb:.2f}")
print(f"RMSE: {rmse_gb:.2f}")
print(f"R² Score: {r2_gb:.4f}")


# MODEL COMPARISON

print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)

comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MAE': [mae_lr, mae_rf, mae_gb],
    'RMSE': [rmse_lr, rmse_rf, rmse_gb],
    'R² Score': [r2_lr, r2_rf, r2_gb]
})

print(comparison_df.to_string(index=False))

# Select best model based on lowest RMSE (or highest R²)
best_model_idx = comparison_df['RMSE'].idxmin()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
print(f"\nBest Model (Lowest RMSE): {best_model_name}")


# HYPERPARAMETER TUNING FOR BEST MODEL

print("\n" + "="*50)
print(f"HYPERPARAMETER TUNING: {best_model_name}")
print("="*50)

if best_model_name == "Random Forest":
    # Tune Random Forest
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search_rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search_rf.fit(X_train, y_train)
    
    best_rf_model = grid_search_rf.best_estimator_
    y_pred_tuned = best_rf_model.predict(X_test)
    
    print(f"Best Parameters: {grid_search_rf.best_params_}")
    
elif best_model_name == "Gradient Boosting":
    # Tune Gradient Boosting
    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search_gb = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid_gb,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search_gb.fit(X_train, y_train)
    
    best_gb_model = grid_search_gb.best_estimator_
    y_pred_tuned = best_gb_model.predict(X_test)
    
    print(f"Best Parameters: {grid_search_gb.best_params_}")
    
else:
    # Linear Regression doesn't need hyperparameter tuning
    best_lr_model = lr_model
    y_pred_tuned = y_pred_lr
    print("Linear Regression - No hyperparameters to tune")

# Evaluate tuned model
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"\nTuned Model Performance:")
print(f"MAE: {mae_tuned:.2f}")
print(f"RMSE: {rmse_tuned:.2f}")
print(f"R² Score: {r2_tuned:.4f}")

# ============================================
# SAVE MODEL FOR DEPLOYMENT
# ============================================
print("\n" + "="*50)
print("SAVING MODEL FOR DEPLOYMENT")
print("="*50)

# Determine the best tuned model
if best_model_name == "Random Forest":
    final_model = best_rf_model
elif best_model_name == "Gradient Boosting":
    final_model = best_gb_model
else:
    final_model = lr_model

# Save the model
model_path = os.path.join(os.path.dirname(__file__), "..", "visa_processing_model.pkl")
joblib.dump(final_model, model_path)
print(f"Model saved to: {model_path}")

# Save preprocessing information (feature names, office map, etc.)
preprocessing_info = {
    'feature_names': list(X.columns),
    'office_map': office_map,
    'model_type': best_model_name,
    'mean_processing_days': df['processing_days'].mean(),
    'country_avg': country_avg.to_dict(),
    'visa_avg': visa_avg.to_dict()
}

preprocessing_path = os.path.join(os.path.dirname(__file__), "..", "preprocessing_info.pkl")
with open(preprocessing_path, 'wb') as f:
    pickle.dump(preprocessing_info, f)
print(f"Preprocessing info saved to: {preprocessing_path}")

# ============================================
# VISUALIZATIONS
# ============================================
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# 1. Model Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(comparison_df['Model'], comparison_df['MAE'])
axes[0].set_title('MAE Comparison')
axes[0].set_ylabel('Mean Absolute Error')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(comparison_df['Model'], comparison_df['RMSE'])
axes[1].set_title('RMSE Comparison')
axes[1].set_ylabel('Root Mean Squared Error')
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(comparison_df['Model'], comparison_df['R² Score'])
axes[2].set_title('R² Score Comparison')
axes[2].set_ylabel('R² Score')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 2. Actual vs Predicted (Best Model - Tuned)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_tuned, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Processing Days')
plt.ylabel('Predicted Processing Days')
plt.title(f'Actual vs Predicted - {best_model_name} (Tuned)')
plt.show()

# 3. Residual Plot
residuals = y_test - y_pred_tuned
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_tuned, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Processing Days')
plt.ylabel('Residuals')
plt.title(f'Residual Plot - {best_model_name} (Tuned)')
plt.show()

print("\n" + "="*50)
print("MILESTONE 3 COMPLETED!")
print("="*50)