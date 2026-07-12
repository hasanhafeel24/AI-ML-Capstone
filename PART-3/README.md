# Part 3 – Advanced Modeling, Ensembles, Hyperparameter Tuning, and ML Pipeline

## Objective

The objective of Part 3 was to compare multiple machine learning models, evaluate their robustness using cross-validation, optimize model performance through hyperparameter tuning, and build a reusable machine learning pipeline for deployment.

---

# Dataset

**Dataset Used:** Student Performance Factors Dataset

## Target Variable

A binary classification target was created from the Exam Score.

- Students scoring **above the median Exam Score** were labeled as **1**
- Students scoring **below or equal to the median Exam Score** were labeled as **0**

---

# 1. Decision Tree Baseline

A default `DecisionTreeClassifier` was trained without restricting its depth.

## Results

- **Training Accuracy:** 100.00%
- **Testing Accuracy:** 80.71%

### Interpretation

The baseline Decision Tree clearly shows signs of overfitting.

Although it perfectly memorized the training dataset, its performance dropped significantly on unseen test data.

Decision Trees are considered **high-variance models** because they greedily choose the best split at every node without revisiting previous decisions. This allows them to fit training data extremely well but often reduces generalization performance.

---

# 2. Controlled Decision Tree

A second Decision Tree was trained using:

- **max_depth = 5**
- **min_samples_split = 20**

## Results

- **Training Accuracy:** 84.60%
- **Testing Accuracy:** 81.62%

### Interpretation

Restricting the tree depth reduced overfitting.

The gap between training and testing accuracy became much smaller, indicating improved generalization.

### Parameter Explanation

### max_depth

Limits how deep the decision tree is allowed to grow.

Reducing the maximum depth decreases model variance but may introduce a small amount of bias.

### min_samples_split

Specifies the minimum number of samples required before a node can be split.

This prevents the model from creating branches based on very small noisy subsets.

---

# 3. Gini vs Entropy

Two Decision Tree models were trained using different splitting criteria.

| Criterion | Test Accuracy |
|-----------|--------------:|
| Gini | 81.62% |
| Entropy | 82.38% |

## Gini Formula

**Gini = 1 − Σ(pi²)**

## Entropy Formula

**Entropy = −Σ(pi log₂(pi))**

### Interpretation

A **Gini value of 0** means every sample inside the node belongs to exactly one class.

Both criteria produced very similar performance, although Entropy performed slightly better on this dataset.

---

# 4. Random Forest

Random Forest was trained using:

- **n_estimators = 100**
- **max_depth = 10**
- **random_state = 42**

## Results

- **Training Accuracy:** 97.65%
- **Testing Accuracy:** 88.58%
- **ROC-AUC:** 95.90%

---

## Top Five Important Features

| Feature | Importance |
|---------|-----------:|
| Attendance | 0.4333 |
| Hours Studied | 0.2077 |
| Previous Scores | 0.0700 |
| Tutoring Sessions | 0.0362 |
| Sleep Hours | 0.0229 |

### Feature Importance Interpretation

Random Forest computes feature importance by measuring the **average reduction in Gini impurity** contributed by each feature across all trees.

Unlike Linear Regression coefficients, feature importance values do **not** indicate whether a feature has a positive or negative relationship with the target.

Instead, they measure how useful a feature is for making accurate predictions.

---

# Bagging Concept

Random Forest uses **Bootstrap Aggregating (Bagging).**

Each tree is trained on a random sample of the training data selected **with replacement**.

At every split, only a **random subset of features** is considered.

Combining predictions from many independent trees reduces variance and improves generalization compared to a single Decision Tree.

---

# 4a. Gradient Boosting

Gradient Boosting was trained using:

- **n_estimators = 100**
- **learning_rate = 0.1**
- **max_depth = 3**

## Results

- **Training Accuracy:** 94.47%
- **Testing Accuracy:** 90.85%
- **ROC-AUC:** 97.38%

Gradient Boosting achieved the highest testing accuracy among all ensemble models.

---

# 4b. Feature Ablation Study

The five least important features identified by the Random Forest model were removed, and a second Random Forest model was trained.

## Results

| Model | ROC-AUC |
|-------|---------:|
| Full Model | **0.9590** |
| Reduced Model | **0.9590** |

### Interpretation

Removing the least important features produced **no reduction in ROC-AUC**.

This indicates that these features contributed very little predictive information.

A simpler model with fewer features can therefore reduce computational cost and maintenance effort without sacrificing predictive performance.

---

# 5. Cross Validation

Five-fold Stratified Cross Validation was performed using ROC-AUC.

| Model | Mean AUC | Std AUC |
|-------|---------:|--------:|
| Logistic Regression | **0.9926** | 0.0031 |
| Decision Tree | 0.9011 | 0.0142 |
| Random Forest | 0.9611 | 0.0080 |
| Gradient Boosting | 0.9736 | 0.0056 |

### Interpretation

Cross-validation provides a more reliable estimate of model performance than a single train-test split because every observation is used for both training and validation across multiple folds.

---

# 6. Hyperparameter Tuning

GridSearchCV was applied using a Random Forest Pipeline.

## Pipeline Components

- SimpleImputer (Median)
- StandardScaler
- RandomForestClassifier

## Parameter Grid

- n_estimators = [50, 100, 200]
- max_depth = [5, 10, None]
- min_samples_leaf = [1, 5]

### Total Parameter Combinations

18

### Total Model Evaluations

90

## Best Parameters

- max_depth = None
- min_samples_leaf = 1
- n_estimators = 200

## Best Cross Validation AUC

**0.9719**

### Grid Search vs Random Search

Grid Search evaluates every possible parameter combination and guarantees the best configuration within the defined search space.

Randomized Search evaluates only randomly selected combinations, making it faster but without guaranteeing the global optimum.

---

# 7. Manual Learning Curve

| Training Fraction | Training AUC | Test AUC |
|------------------|-------------:|---------:|
| 20% | 1.0000 | 0.9562 |
| 40% | 1.0000 | 0.9601 |
| 60% | 1.0000 | 0.9675 |
| 80% | 1.0000 | 0.9691 |
| 100% | 1.0000 | 0.9693 |

### Interpretation

Training AUC remained at **1.0000** across all training fractions, indicating that the Random Forest perfectly fit each training subset.

Test AUC steadily increased as more training data became available, demonstrating improved generalization.

Since the increase begins to plateau near the full training set, the model appears only slightly data-limited and is approaching its maximum achievable performance.

---

# 8. Model Serialization

The best pipeline obtained from GridSearchCV was saved using Joblib.

## Saved Model

```
best_model.pkl
```

The saved model was successfully reloaded using `joblib.load()` and used to predict two unseen test samples without errors.

---

# Final Model Comparison

| Model | Mean CV AUC | Std CV AUC | Test AUC |
|-------|------------:|-----------:|---------:|
| Logistic Regression | 0.9926 | 0.0031 | **0.9951** |
| Decision Tree | 0.9011 | 0.0142 | — |
| Random Forest | 0.9611 | 0.0080 | 0.9590 |
| Gradient Boosting | 0.9736 | 0.0056 | **0.9738** |

---

# Final Recommendation

Based on all experiments, **Gradient Boosting** is recommended for deployment.

Although Logistic Regression achieved the highest ROC-AUC on this dataset, Gradient Boosting provides an excellent balance between predictive performance, robustness, and its ability to model complex non-linear relationships.

Its cross-validation performance also remained consistently high, making it a strong choice for real-world student performance prediction.

---

# Repository Structure

```
PART-3/
│── train_ensemble.py
│── best_model.pkl
│── README.md

data/
│── cleaned_student_data.csv
```

---

# Files Included

- train_ensemble.py
- best_model.pkl
- README.md