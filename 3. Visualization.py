import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("student_lifestyle_performance_dataset.csv")

# ==============================
# OUTLIER REMOVAL
# ==============================
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

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
