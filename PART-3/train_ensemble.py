import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("../data/cleaned_student_data.csv")

# ==========================================================
# CREATE TARGET
# ==========================================================

y = (df["Exam_Score"] > df["Exam_Score"].median()).astype(int)

X = df.drop("Exam_Score", axis=1)

categorical_cols = X.select_dtypes(include=["object", "string"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns

# ==========================================================
# PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            categorical_cols,
        ),
        (
            "num",
            StandardScaler(),
            numeric_cols,
        ),
    ]
)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Fit preprocessor

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("=" * 60)
print("DATA LOADED SUCCESSFULLY")
print("=" * 60)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print("Features         :", X.shape[1])


# ==========================================================
# DECISION TREE BASELINE
# ==========================================================

print("\n" + "=" * 60)
print("DECISION TREE (UNCONSTRAINED)")
print("=" * 60)

tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train_processed, y_train)

train_pred = tree.predict(X_train_processed)
test_pred = tree.predict(X_test_processed)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Training Accuracy : {train_acc:.4f}")
print(f"Testing Accuracy  : {test_acc:.4f}")

if train_acc - test_acc > 0.05:
    print("\nObservation:")
    print("The model appears to be overfitting.")
else:
    print("\nObservation:")
    print("The model generalizes well.")

    # ==========================================================
# CONTROLLED DECISION TREE
# ==========================================================

print("\n" + "=" * 60)
print("CONTROLLED DECISION TREE")
print("=" * 60)

controlled_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

controlled_tree.fit(X_train_processed, y_train)

train_pred_controlled = controlled_tree.predict(X_train_processed)
test_pred_controlled = controlled_tree.predict(X_test_processed)

train_acc_controlled = accuracy_score(y_train, train_pred_controlled)
test_acc_controlled = accuracy_score(y_test, test_pred_controlled)

print(f"Training Accuracy : {train_acc_controlled:.4f}")
print(f"Testing Accuracy  : {test_acc_controlled:.4f}")

print("\nTrain/Test Gap Comparison")
print(f"Unconstrained Gap : {train_acc - test_acc:.4f}")
print(f"Controlled Gap    : {train_acc_controlled - test_acc_controlled:.4f}")

if (train_acc_controlled - test_acc_controlled) < (train_acc - test_acc):
    print("\nObservation:")
    print("The controlled tree reduces overfitting and generalizes better.")
else:
    print("\nObservation:")
    print("The controlled tree does not reduce overfitting.")

    # ==========================================================
# GINI vs ENTROPY COMPARISON
# ==========================================================

print("\n" + "=" * 60)
print("GINI vs ENTROPY")
print("=" * 60)

gini_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

entropy_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

gini_tree.fit(X_train_processed, y_train)
entropy_tree.fit(X_train_processed, y_train)

gini_pred = gini_tree.predict(X_test_processed)
entropy_pred = entropy_tree.predict(X_test_processed)

gini_acc = accuracy_score(y_test, gini_pred)
entropy_acc = accuracy_score(y_test, entropy_pred)

comparison = pd.DataFrame({
    "Criterion": ["Gini", "Entropy"],
    "Test Accuracy": [gini_acc, entropy_acc]
})

print(comparison)

# ==========================================================
# RANDOM FOREST
# ==========================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf.fit(X_train_processed, y_train)

rf_train_pred = rf.predict(X_train_processed)
rf_test_pred = rf.predict(X_test_processed)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

rf_prob = rf.predict_proba(X_test_processed)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob)

print(f"Training Accuracy : {rf_train_acc:.4f}")
print(f"Testing Accuracy  : {rf_test_acc:.4f}")
print(f"ROC-AUC           : {rf_auc:.4f}")

# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 5 Important Features")
print(importance_df.head(5))

# ==========================================================
# GRADIENT BOOSTING
# ==========================================================

print("\n" + "=" * 60)
print("GRADIENT BOOSTING")
print("=" * 60)

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train_processed, y_train)

gb_train_pred = gb.predict(X_train_processed)
gb_test_pred = gb.predict(X_test_processed)

gb_train_acc = accuracy_score(y_train, gb_train_pred)
gb_test_acc = accuracy_score(y_test, gb_test_pred)

gb_prob = gb.predict_proba(X_test_processed)[:, 1]
gb_auc = roc_auc_score(y_test, gb_prob)

print(f"Training Accuracy : {gb_train_acc:.4f}")
print(f"Testing Accuracy  : {gb_test_acc:.4f}")
print(f"ROC-AUC           : {gb_auc:.4f}")

# ==========================================================
# FEATURE ABLATION STUDY
# ==========================================================

print("\n" + "=" * 60)
print("FEATURE ABLATION STUDY")
print("=" * 60)

# Get 5 least important transformed features
least_features = importance_df.tail(5)["Feature"].tolist()

print("\n5 Least Important Features")
print(pd.DataFrame({"Feature": least_features}))

# Map transformed names back to original column names
remove_original = []

for feature in least_features:
    if feature.startswith("num__"):
        remove_original.append(feature.replace("num__", ""))
    elif feature.startswith("cat__"):
        original = feature.replace("cat__", "").split("_")[0]
        remove_original.append(original)

remove_original = list(set(remove_original))

print("\nOriginal Features Removed")
print(remove_original)

X_reduced = X.drop(columns=remove_original, errors="ignore")

cat_cols_reduced = X_reduced.select_dtypes(include=["object", "string"]).columns
num_cols_reduced = X_reduced.select_dtypes(exclude=["object", "string"]).columns

preprocessor_reduced = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            cat_cols_reduced,
        ),
        (
            "num",
            StandardScaler(),
            num_cols_reduced,
        ),
    ]
)

X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train_red = preprocessor_reduced.fit_transform(X_train_red)
X_test_red = preprocessor_reduced.transform(X_test_red)

rf_reduced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
)

rf_reduced.fit(X_train_red, y_train_red)

reduced_prob = rf_reduced.predict_proba(X_test_red)[:, 1]

reduced_auc = roc_auc_score(y_test_red, reduced_prob)

print(f"\nFull Model ROC-AUC    : {rf_auc:.4f}")
print(f"Reduced Model ROC-AUC : {reduced_auc:.4f}")

# ==========================================================
# CROSS-VALIDATED MODEL COMPARISON
# ==========================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train_processed,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    results.append([
        name,
        scores.mean(),
        scores.std()
    ])

cv_results = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Mean AUC",
        "Std AUC"
    ]
)

print(cv_results)

# ==========================================================
# GRID SEARCH + PIPELINE
# ==========================================================

print("\n" + "=" * 60)
print("GRID SEARCH CV")
print("=" * 60)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------------------
# Detect feature types
# ----------------------------------------

categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns
numerical_cols = X.select_dtypes(exclude=["object", "category", "string"]).columns

# ----------------------------------------
# Numeric Pipeline
# ----------------------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ----------------------------------------
# Categorical Pipeline
# ----------------------------------------

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ----------------------------------------
# Full Preprocessor
# ----------------------------------------

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ----------------------------------------
# Final Pipeline
# ----------------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("randomforestclassifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters")
print(grid.best_params_)

print("\nBest Cross Validation AUC")
print(f"{grid.best_score_:.4f}")

best_pipeline = grid.best_estimator_

print(type(best_pipeline))
print(best_pipeline)

joblib.dump(best_pipeline, "best_model.pkl")

print("\nModel saved as best_model.pkl")

total_models = (
    len(param_grid["randomforestclassifier__n_estimators"])
    * len(param_grid["randomforestclassifier__max_depth"])
    * len(param_grid["randomforestclassifier__min_samples_leaf"])
)

print(f"\nTotal Parameter Combinations : {total_models}")
print(f"Total Models Evaluated       : {total_models * 5}")

# ============================================================
# MANUAL LEARNING CURVE
# ============================================================

print("\n============================================================")
print("MANUAL LEARNING CURVE")
print("============================================================")

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

learning_results = []

for f in fractions:

    size = int(f * len(X_train))

    X_small = X_train.iloc[:size]
    y_small = y_train.iloc[:size]

    best_pipeline.fit(X_small, y_small)

    train_prob = best_pipeline.predict_proba(X_small)[:, 1]
    test_prob = best_pipeline.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_small, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)

    learning_results.append([
        f,
        train_auc,
        test_auc
    ])

learning_df = pd.DataFrame(
    learning_results,
    columns=[
        "Training Fraction",
        "Training AUC",
        "Test AUC"
    ]
)

print(learning_df)

# ============================================================
# LOAD SAVED MODEL
# ============================================================

print("\n============================================================")
print("LOADING SAVED MODEL")
print("============================================================")

loaded_model = joblib.load("best_model.pkl")

sample_rows = X_test.iloc[:2]

predictions = loaded_model.predict(sample_rows)

print("Predictions for two sample rows:")
print(predictions)