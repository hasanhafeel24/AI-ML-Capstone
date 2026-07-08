import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("data/cleaned_student_data.csv")

# Convert categorical columns to numeric
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()

label_encoder = LabelEncoder()

for column in df_encoded.columns:
    if df_encoded[column].dtype == "object":
        df_encoded[column] = label_encoder.fit_transform(df_encoded[column].astype(str))

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Correlation matrix
correlation = df_encoded.corr(numeric_only=True)

# Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("plots/correlation_heatmap.png")

plt.show()