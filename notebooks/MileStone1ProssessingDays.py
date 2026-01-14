# IMPORT PACKAGES
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.set_option("display.max_columns", None)

# PART 1: LARGE VISA DATASET (LOAD FROM CSV INSTEAD OF HARD-CODED DATA)
print("\n===== PART 1: LARGE VISA DATASET (FROM visa_dataset.csv) =====\n")

csv_path = os.path.join(os.path.dirname(__file__), "..", "visa_dataset.csv")
df = pd.read_csv(csv_path)
print("Original DataFrame loaded from visa_dataset.csv:\n", df.head())


# DATE CONVERSION
df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])

print("\nAfter Date Conversion:\n", df)

# PROCESSING DAYS
df["processing_days"] = (
    df["decision_date"] - df["application_date"]
).dt.days

print("\nAfter Calculating Processing Days:\n", df)

# ENCODING
df_encoded = pd.get_dummies(df, columns=["country", "visa_type"])

print("\nEncoded DataFrame:\n", df_encoded)

# MACHINE LEARNING
X = df_encoded.drop(
    columns=["processing_days", "application_date", "decision_date"]
)
y = df_encoded["processing_days"]

model = LinearRegression()
model.fit(X, y)


# PREDICTION SAMPLE
sample_input = pd.DataFrame(
    np.zeros((1, len(X.columns))),
    columns=X.columns
)

# Example: India + Student Visa
sample_input.loc[0, "country_India"] = 1
sample_input.loc[0, "visa_type_Student"] = 1

predicted_days = model.predict(sample_input)

print("\nPredicted Processing Time (India + Student):",
      round(predicted_days[0], 2), "days")
