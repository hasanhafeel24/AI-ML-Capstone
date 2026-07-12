import pandas as pd
import numpy as np

# ==========================================================
# LOAD CLEANED DATASET
# ==========================================================

df = pd.read_csv("data/cleaned_student_data.csv")

print("=" * 60)
print("ADVANCED ANALYSIS")
print("=" * 60)

# ==========================================================
# SKEWNESS
# ==========================================================

numeric_cols = df.select_dtypes(include=np.number).columns

skewness = pd.DataFrame({
    "Column": numeric_cols,
    "Skewness": [df[col].skew() for col in numeric_cols]
})

skewness["Absolute"] = skewness["Skewness"].abs()

skewness = skewness.sort_values("Absolute", ascending=False)

print("\nSkewness")
print(skewness)

# ==========================================================
# MEAN vs MEDIAN
# ==========================================================

print("\n" + "=" * 60)
print("MEAN vs MEDIAN")
print("=" * 60)

top2 = skewness.head(2)["Column"].tolist()

for col in top2:

    print(f"\n{col}")

    print(f"Mean   : {df[col].mean():.4f}")
    print(f"Median : {df[col].median():.4f}")

    df[col] = df[col].fillna(df[col].median())

print("\nRemaining Null Values")

print(df[top2].isnull().sum())

# ==========================================================
# PEARSON
# ==========================================================

pearson = df[numeric_cols].corr()

# ==========================================================
# SPEARMAN
# ==========================================================

spearman = df[numeric_cols].corr(method="spearman")

print("\nPearson Correlation")
print(pearson)

print("\nSpearman Correlation")
print(spearman)

# ==========================================================
# DIFFERENCE TABLE
# ==========================================================

difference = []

for i in range(len(numeric_cols)):

    for j in range(i + 1, len(numeric_cols)):

        c1 = numeric_cols[i]
        c2 = numeric_cols[j]

        p = pearson.loc[c1, c2]
        s = spearman.loc[c1, c2]

        difference.append([
            c1,
            c2,
            p,
            s,
            abs(s - p)
        ])

difference_df = pd.DataFrame(
    difference,
    columns=[
        "Column 1",
        "Column 2",
        "Pearson",
        "Spearman",
        "|Difference|"
    ]
)

difference_df = difference_df.sort_values(
    "|Difference|",
    ascending=False
)

print("\nTop 3 Pearson vs Spearman Differences")

print(difference_df.head(3))

# ==========================================================
# GROUPED AGGREGATION
# ==========================================================

print("\n" + "=" * 60)
print("GROUPED AGGREGATION")
print("=" * 60)

group = df.groupby("Gender")["Exam_Score"].agg([
    "mean",
    "std",
    "count"
])

print(group)

highest_mean = group["mean"].idxmax()
highest_std = group["std"].idxmax()

ratio = group["mean"].max() / group["mean"].min()

print("\nHighest Mean Group :", highest_mean)
print("Highest Std Group  :", highest_std)
print(f"Mean Ratio         : {ratio:.3f}")

print("\nAdvanced Analysis Completed Successfully.")