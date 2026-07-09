import pandas as pd

# Load dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

print("=" * 60)
print("BEFORE CLEANING")
print("=" * 60)

print(f"Shape: {df.shape}")
print(f"Missing Values:\n{df.isnull().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")

# Remove duplicate rows
df = df.drop_duplicates()

print("\n")

print("=" * 60)
print("AFTER REMOVING DUPLICATES")
print("=" * 60)

print(f"Shape: {df.shape}")
print(f"Duplicate Rows: {df.duplicated().sum()}")

# Save cleaned dataset
df.to_csv("data/cleaned_student_data.csv", index=False)

print("\nCleaned dataset saved successfully!")