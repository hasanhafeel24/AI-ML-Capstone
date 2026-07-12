# Part 1 – Data Acquisition, Cleaning, and Exploratory Data Analysis

## Objective

This part focuses on preparing the Student Performance dataset for machine learning by performing data cleaning, exploratory data analysis (EDA), statistical analysis, and visualization. The goal is to produce a clean, reliable dataset and understand the relationships between different features before model development.

---

## Dataset

- **Dataset:** Student Performance Factors Dataset
- **Source:** CSV file (`StudentPerformanceFactors.csv`)
- **Target Variable:** Exam_Score

---

## Tasks Completed

### Data Preparation
- Loaded the dataset using Pandas
- Displayed dataset shape, column names, and data types
- Performed missing value analysis
- Calculated null counts and null percentages
- Filled missing numerical values using the **median**
- Detected and removed duplicate records
- Converted appropriate columns to correct data types
- Saved the cleaned dataset as:
  - `cleaned_student_data.csv`

---

### Exploratory Data Analysis

Performed descriptive statistics and analyzed:

- Dataset summary statistics
- Feature distributions
- Skewness of numerical features
- Outlier detection using the IQR method
- Correlation analysis (Pearson)
- Rank correlation analysis (Spearman)

---

### Advanced Statistical Analysis

Completed the following analyses required by the project rubric:

- Mean vs Median comparison for the two most skewed features
- Pearson vs Spearman correlation comparison
- Top three feature pairs with the largest correlation differences
- Grouped aggregation using categorical features
- Mean, standard deviation, and count for each group
- Highest mean group identification
- Highest variance group identification
- Mean ratio calculation and interpretation

---

## Visualizations Generated

The following plots were created:

- Line Plot
- Bar Chart
- Histogram
- Scatter Plot
- Box Plot
- Correlation Heatmap

All visualization files are available in the **plots/** directory.

---

## Key Findings

- Median was selected for imputing missing numerical values because it is more robust to skewed distributions and outliers than the mean.
- Attendance and Hours Studied showed the strongest positive relationships with Exam Score.
- Spearman correlation identified stronger monotonic relationships for several feature pairs compared to Pearson correlation.
- Outliers were identified using the IQR method and retained for subsequent modeling because they may represent valid student performance variations rather than data errors.
- Grouped aggregation showed small performance differences between categories while highlighting variation within groups.

---

## Output Files

- `clean_data.py`
- `eda.py`
- `advanced_analysis.py`
- `visualize.py`
- `correlation.py`
- `cleaned_student_data.csv`
- `plots/`

---

## Conclusion

Part 1 successfully transformed the raw dataset into a clean and analysis-ready dataset. Comprehensive exploratory analysis, statistical evaluation, and visualization provided valuable insights into feature distributions, correlations, and potential predictive variables. The cleaned dataset serves as the foundation for the machine learning models developed in Parts 2, 3, and 4.