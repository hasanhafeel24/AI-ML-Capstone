import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_student_data.csv")

# Set plot style
sns.set_style("whitegrid")

# -------------------------------
# 1. Histogram - Exam Score
# -------------------------------
plt.figure(figsize=(8,5))
plt.hist(df["Exam_Score"], bins=20)
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.savefig("plots/exam_score_distribution.png")
plt.close()

# -------------------------------
# 2. Attendance Distribution
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Attendance"], bins=20)
plt.title("Attendance Distribution")
plt.savefig("plots/attendance_distribution.png")
plt.close()

# -------------------------------
# 3. Gender Count
# -------------------------------
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
plt.savefig("plots/gender_distribution.png")
plt.close()

# -------------------------------
# 4. Box Plot
# -------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Exam_Score"])
plt.title("Exam Score Box Plot")
plt.savefig("plots/exam_score_boxplot.png")
plt.close()

# -------------------------------
# 5. Scatter Plot
# -------------------------------
plt.figure(figsize=(8,5))
plt.scatter(df["Hours_Studied"], df["Exam_Score"])
plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.savefig("plots/hours_vs_exam_score.png")
plt.close()

print("✅ All plots generated successfully!")