import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("../data/cleaned_student_data.csv")

# ==========================================================
# Features & Target
# ==========================================================

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

# ==========================================================
# Categorical & Numerical Columns
# ==========================================================

categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# ==========================================================
# Preprocessing
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# ==========================================================
# Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Linear Regression Pipeline
# ==========================================================

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

# ==========================================================
# Linear Metrics
# ==========================================================

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_rmse = linear_mse ** 0.5
linear_r2 = r2_score(y_test, linear_predictions)

print("=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

print(f"MAE  : {linear_mae:.4f}")
print(f"MSE  : {linear_mse:.4f}")
print(f"RMSE : {linear_rmse:.4f}")
print(f"R²   : {linear_r2:.4f}")

# ==========================================================
# Ridge Regression
# ==========================================================

ridge_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

ridge_model.fit(X_train, y_train)

ridge_predictions = ridge_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_predictions)
ridge_r2 = r2_score(y_test, ridge_predictions)

print("\n" + "=" * 60)
print("RIDGE REGRESSION")
print("=" * 60)

print(f"MSE : {ridge_mse:.4f}")
print(f"R²  : {ridge_r2:.4f}")

# ==========================================================
# Comparison Table
# ==========================================================

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression"],
    "MSE": [linear_mse, ridge_mse],
    "R²": [linear_r2, ridge_r2]
})

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(comparison)

# ==========================================================
# Feature Coefficients
# ==========================================================

feature_names = linear_model.named_steps["preprocessor"].get_feature_names_out()

coefficients = linear_model.named_steps["regressor"].coef_

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coef_df["Absolute"] = coef_df["Coefficient"].abs()

coef_df = coef_df.sort_values("Absolute", ascending=False)

print("\n")
print("=" * 60)
print("TOP 3 FEATURES")
print("=" * 60)

print(coef_df.head(3)[["Feature", "Coefficient"]])

# ==========================================================
# Save Predictions
# ==========================================================

os.makedirs("results", exist_ok=True)

prediction_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": linear_predictions
})

prediction_df.to_csv(
    "results/predictions.csv",
    index=False
)

# ==========================================================
# Save Model
# ==========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    linear_model,
    "models/student_performance_model.pkl"
)

print("\nPipeline model saved successfully!")