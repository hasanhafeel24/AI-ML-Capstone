import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 60)

# Load cleaned dataset from PART-1
df = pd.read_csv("../PART-1/data/cleaned_student_data.csv")

print("\n✅ Dataset loaded successfully!")

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\n" + "=" * 60)
print("ENCODING CATEGORICAL COLUMNS")
print("=" * 60)

label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = label_encoder.fit_transform(df[column])

print("✅ Encoding completed successfully!")

print(f"\nRows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumn Names:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 Rows:")
print(df.head())

print("\nUpdated Data Types:")
print(df.dtypes)