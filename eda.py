import pandas as pd

# Load the dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n")

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\n")

print("=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns)

print("\n")

print("=" * 50)
print("DATA INFORMATION")
print("=" * 50)
print(df.info())

print("\n")

print("=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)
print(df.describe())

print("\n")

print("=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())