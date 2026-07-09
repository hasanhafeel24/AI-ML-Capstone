import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("../data/cleaned_student_data.csv")

# Convert all categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop(columns=["Exam_Score"])
y = df["Exam_Score"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Save prediction results
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

os.makedirs("results", exist_ok=True)
results.to_csv("results/predictions.csv", index=False)

print("Predictions saved successfully!")

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("Model Performance")
print("-----------------")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.2f}")

# Create models folder
os.makedirs("models", exist_ok=True)

# Save Model
joblib.dump(model, "models/student_performance_model.pkl")

print("\nModel saved successfully!")