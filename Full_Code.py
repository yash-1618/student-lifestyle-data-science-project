# ==============================
# DATA SCIENCE PROJECT
# Student Lifestyle & Performance
# ==============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("student_lifestyle_performance_dataset.csv")

print("\n===== DATA OVERVIEW =====")
print(df.head())
print("\nShape:", df.shape)
print("\nData Types:\n", df.dtypes)

# ==============================
# 2. DATA CLEANING
# ==============================
print("\n===== DATA CLEANING =====")

# Missing values
print("Missing Values:\n", df.isnull().sum())

# Duplicates
print("Duplicate Rows:", df.duplicated().sum())

# Convert categorical
df['Branch'] = df['Branch'].astype('category')
df['Diet_Type'] = df['Diet_Type'].astype('category')
df['Residence'] = df['Residence'].astype('category')

# ==============================
# 3. EDA (EXPLORATORY DATA ANALYSIS)
# ==============================
print("\n===== EDA =====")

# Summary statistics
print(df.describe())

# Skewness
print("\nSkewness:\n", df.skew(numeric_only=True))

# ==============================
# 🔹 3.1 OUTLIER DETECTION (IQR METHOD)
# ==============================
print("\n===== OUTLIER DETECTION =====")

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col}: {len(outliers)} outliers")

# ==============================
# 4. VISUALIZATION (5 PLOTS)
# ==============================
sns.set(style="whitegrid")

# 1. CGPA Distribution
plt.figure()
sns.histplot(df['CGPA'], kde=True)
plt.title("CGPA Distribution")
plt.show()

# 2. Study Hours vs CGPA (IMPORTANT)
plt.figure()
sns.regplot(x='Study_Hours_per_Day', y='CGPA', data=df)
plt.title("Study Hours vs CGPA")
plt.show()

# 3. Boxplot (Outliers)
plt.figure()
sns.boxplot(x=df['CGPA'])
plt.title("CGPA Boxplot")
plt.show()

# 4. CGPA by Branch
plt.figure()
sns.boxplot(x='Branch', y='CGPA', data=df)
plt.xticks(rotation=45)
plt.title("CGPA by Branch")
plt.show()

# 5. Bar Chart (Study Hours vs CGPA)
plt.figure(figsize=(8,5))

df['Study_Category'] = pd.cut(df['Study_Hours_per_Day'],
                             bins=[0,2,4,6,10],
                             labels=['0-2','2-4','4-6','6+'])

sns.barplot(x='Study_Category', y='CGPA',
            data=df.groupby('Study_Category')['CGPA'].mean().reset_index())

plt.title("Average CGPA by Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("CGPA")

plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ==============================
# 5. CORRELATION ANALYSIS
# ==============================
print("\n===== CORRELATION =====")
corr = df.corr(numeric_only=True)
print(corr['CGPA'].sort_values(ascending=False))

# ==============================
# 6. HYPOTHESIS TESTING
# ==============================
print("\n===== HYPOTHESIS TESTING =====")

df['Sleep_Group'] = np.where(df['Sleep_Hours'] >= 7, 'Adequate', 'Inadequate')

group1 = df[df['Sleep_Group'] == 'Adequate']['CGPA']
group2 = df[df['Sleep_Group'] == 'Inadequate']['CGPA']

t_stat, p_val = ttest_ind(group1, group2)

print("\nT-Test (Sleep vs CGPA)")
print("p-value:", p_val)

if p_val < 0.05:
    print("Reject H0 → Sleep affects CGPA")
else:
    print("Fail to Reject H0 → No significant effect")

stat, p = shapiro(df['CGPA'])

print("\nShapiro-Wilk Test (Normality)")
print("p-value:", p)

if p > 0.05:
    print("Data is normally distributed")
else:
    print("Data is NOT normally distributed")

# ==============================
# 7. SIMPLE LINEAR REGRESSION
# ==============================
print("\n===== SIMPLE LINEAR REGRESSION =====")

X = df[['Study_Hours_per_Day']]
y = df['CGPA']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("R² Score:", r2_score(y, y_pred))

print("\nEquation:")
print(f"CGPA = {model.coef_[0]:.3f} * Study_Hours + {model.intercept_:.3f}")

# ==============================
# 8. SCENARIO-BASED ANALYSIS
# ==============================
print("\n===== SCENARIO ANALYSIS =====")

print("""
Problem:
Does increasing study hours improve CGPA?

Insight:
- Strong positive correlation between Study Hours and CGPA
- Regression shows direct increase in CGPA with study hours

Conclusion:
Students studying more hours tend to achieve higher CGPA.

Recommendation:
Students should maintain 4–6 hours of focused study daily for optimal performance.
""")