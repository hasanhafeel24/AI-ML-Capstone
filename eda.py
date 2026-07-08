import pandas as pd

# Load dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n")

print("=" * 60)
print("LAST 5 ROWS")
print("=" * 60)
print(df.tail())

print("\n")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n")

print("=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
df.info()

print("\n")

print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(df.describe())

print("\n")

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n")

print("=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())