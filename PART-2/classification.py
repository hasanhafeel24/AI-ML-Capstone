import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("../data/cleaned_student_data.csv")

# ==========================================================
# Create Binary Target
# ==========================================================

df["Target"] = (df["Exam_Score"] > df["Exam_Score"].median()).astype(int)

X = df.drop(["Exam_Score", "Target"], axis=1)
y = df["Target"]

print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
print(y.value_counts())

# ==========================================================
# Columns
# ==========================================================

categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns
numerical_cols = X.select_dtypes(exclude=["object", "category", "string"]).columns

# ==========================================================
# Preprocessing
# ==========================================================

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

# ==========================================================
# Logistic Regression Pipeline
# ==========================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Train
# ==========================================================

model.fit(X_train, y_train)

# ==========================================================
# Predictions
# ==========================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions))

# ==========================================================
# ROC Curve
# ==========================================================

fpr, tpr, _ = roc_curve(y_test, probabilities)

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()

plt.savefig("results/roc_curve.png")
plt.show()

print("\nROC Curve saved to results/roc_curve.png")

# ==========================================================
# Threshold Sensitivity Analysis
# ==========================================================

print("\n" + "=" * 60)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 60)

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

threshold_results = []

for threshold in thresholds:

    preds = (probabilities >= threshold).astype(int)

    p = precision_score(y_test, preds)
    r = recall_score(y_test, preds)
    f = f1_score(y_test, preds)

    threshold_results.append([threshold, p, r, f])

threshold_df = pd.DataFrame(
    threshold_results,
    columns=["Threshold", "Precision", "Recall", "F1 Score"]
)

print(threshold_df)

best_threshold = threshold_df.loc[
    threshold_df["F1 Score"].idxmax()
]

print("\nBest Threshold based on F1 Score")
print(best_threshold)

# ==========================================================
# Logistic Regression Regularization Comparison
# ==========================================================

print("\n" + "=" * 60)
print("REGULARIZATION COMPARISON")
print("=" * 60)

regularized_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(C=0.01, max_iter=1000))
])

regularized_model.fit(X_train, y_train)

reg_predictions = regularized_model.predict(X_test)

reg_probabilities = regularized_model.predict_proba(X_test)[:,1]

reg_precision = precision_score(y_test, reg_predictions)
reg_recall = recall_score(y_test, reg_predictions)
reg_auc = roc_auc_score(y_test, reg_probabilities)

comparison = pd.DataFrame({
    "Model":[
        "Logistic (C=1.0)",
        "Logistic (C=0.01)"
    ],
    "Precision":[
        precision,
        reg_precision
    ],
    "Recall":[
        recall,
        reg_recall
    ],
    "AUC":[
        auc,
        reg_auc
    ]
})

print(comparison)