# Part 2 – Supervised Machine Learning Model

## Objective

The objective of this part is to build and evaluate two supervised machine learning models using the cleaned student performance dataset.

The regression model predicts students' exam scores, while the classification model predicts whether a student's score is above the dataset median.

---

# Dataset

Dataset Used:

Student Performance Factors Dataset

Input Features:

All columns except Exam_Score.

Regression Target:

Exam_Score

Classification Target:

Students with Exam_Score greater than the dataset median were labeled as **1**, while the remaining students were labeled as **0**.

---

# Data Preprocessing

## Encoding

The dataset contains several categorical variables.

### Ordinal Features

Features such as:

- Parental Involvement
- Teacher Quality
- Motivation Level

have natural ordering (Low < Medium < High). These can be represented numerically while preserving their order.

### Nominal Features

Features without any natural ordering were encoded using One-Hot Encoding.

One-Hot Encoding avoids introducing false ordinal relationships that Label Encoding would create.

---

# Train-Test Split

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

Random State:

42

---

# Data Leakage Prevention

The scaler was fitted **only on the training dataset**.

The fitted scaler was then used to transform both the training and testing datasets.

This avoids data leakage because fitting the scaler on the complete dataset would expose the model to statistics from the test data before evaluation.

---

# Linear Regression

Linear Regression was trained to predict Exam_Score.

## Results

| Metric | Value |
|---------|-------|
| MAE | 0.4499 |
| MSE | 3.2521 |
| RMSE | 1.8034 |
| R² | 0.7699 |

---

# Ridge Regression

Ridge Regression was trained using:

Alpha = 1.0

## Results

| Metric | Value |
|---------|-------|
| MSE | 3.2519 |
| R² | 0.7699 |

---

# Linear Regression vs Ridge Regression

| Model | MSE | R² |
|--------|-----|-----|
| Linear Regression | 3.2521 | 0.7699 |
| Ridge Regression | 3.2519 | 0.7699 |

### Interpretation

Ridge Regression applies L2 regularization to reduce excessively large coefficients.

The alpha parameter controls the amount of regularization.

A higher alpha shrinks coefficients more aggressively.

On this dataset, Ridge Regression produced almost identical performance because multicollinearity was not severe.

---

# Most Important Features

Top three features based on absolute coefficient values:

| Feature | Coefficient |
|----------|-------------|
| Access_to_Resources_Low | -1.0558 |
| Access_to_Resources_High | 1.0444 |
| Parental_Involvement_High | 1.0247 |

### Interpretation

A large positive coefficient means increasing that standardized feature increases the predicted exam score.

A large negative coefficient means increasing that feature decreases the predicted exam score.

---

# Binary Classification

Students were classified into two categories:

- 1 → Above Median Score
- 0 → Below or Equal to Median Score

## Class Distribution

Class 0:

3599

Class 1:

3008

The dataset is reasonably balanced.

Therefore, additional balancing methods such as SMOTE or class_weight were not required.

---

# Logistic Regression

Logistic Regression was trained using:

max_iter = 1000

---

## Classification Results

| Metric | Value |
|---------|-------|
| Accuracy | 98.34% |
| Precision | 97.70% |
| Recall | 98.67% |
| F1 Score | 98.18% |
| ROC-AUC | 99.51% |

---

# Confusion Matrix

```
[[706 14]
 [ 8 594]]
```

---

# Precision and Recall

Precision Formula

Precision = TP / (TP + FP)

Recall Formula

Recall = TP / (TP + FN)

### Interpretation

For student performance prediction, Recall is particularly important because identifying students likely to perform well or poorly consistently is more valuable than missing them.

However, the model achieved both high Precision and Recall, indicating excellent overall performance.

---

# ROC Curve

The ROC Curve evaluates the classifier across all classification thresholds.

The Area Under the Curve (AUC) obtained was:

**0.9951**

### Interpretation

An AUC close to 1.0 indicates outstanding discrimination between the two classes.

The classifier can effectively distinguish students above and below the median score.

---

# Threshold Sensitivity Analysis

Thresholds evaluated:

- 0.30
- 0.40
- 0.50
- 0.60
- 0.70

| Threshold | Precision | Recall | F1 Score |
|------------|-----------|--------|----------|
|0.30|0.9127|0.9900|0.9498|
|0.40|0.9429|0.9884|0.9651|
|0.50|0.9770|0.9867|0.9818|
|0.60|0.9914|0.9552|0.9729|
|0.70|0.9982|0.9286|0.9621|

### Best Threshold

0.50

The default threshold produced the highest F1 Score.

Lowering the threshold increases Recall but decreases Precision.

Increasing the threshold improves Precision but may reduce Recall.

---

# Logistic Regression Regularization

Two Logistic Regression models were compared.

| Model | Precision | Recall | AUC |
|--------|-----------|--------|-----|
| C = 1.0 | 0.9770 | 0.9867 | 0.9951 |
| C = 0.01 | 0.9447 | 0.9369 | 0.9896 |

### Interpretation

The parameter C controls the strength of regularization.

Smaller values of C apply stronger L2 regularization.

Reducing C slightly reduced predictive performance on this dataset.

---

# Bootstrap Confidence Interval

500 bootstrap samples were generated.

Mean AUC Difference

0.005399

95% Confidence Interval

Lower Bound:

0.003489

Upper Bound:

0.007910

### Interpretation

The confidence interval does not include zero.

Therefore, the Logistic Regression model using C = 1.0 consistently outperformed the more heavily regularized model across repeated samples.

---

# Files Included

- train_model.py
- classification.py
- student_performance_model.pkl
- predictions.csv
- roc_curve.png
- README.md
