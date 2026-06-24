"""
train_model.py
----------------
Loads placement_data.csv, trains a Linear Regression model, evaluates it,
and saves the trained model + feature names to disk using joblib so the
Streamlit app can load it instantly without retraining every time.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load data
df = pd.read_csv("placement_data.csv")

FEATURES = ["CGPA", "Internships", "Projects", "Certifications",
            "Coding_Score", "Communication_Score", "Backlogs"]
TARGET = "Package_LPA"

X = df[FEATURES]
y = df[TARGET]

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("===== Model Evaluation =====")
print(f"R² Score : {r2:.4f}  (closer to 1.0 = better fit)")
print(f"MAE      : {mae:.2f} LPA")
print(f"RMSE     : {rmse:.2f} LPA")
print()
print("===== Learned Coefficients (feature importance) =====")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"{feat:22s}: {coef:+.3f}")
print(f"{'Intercept':22s}: {model.intercept_:+.3f}")

# 5. Save model + feature list together
joblib.dump({"model": model, "features": FEATURES, "r2": r2, "mae": mae}, "placify_model.pkl")
print("\nModel saved to placify_model.pkl")