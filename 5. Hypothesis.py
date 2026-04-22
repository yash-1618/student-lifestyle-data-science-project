import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, shapiro

df = pd.read_csv("student_lifestyle_performance_dataset.csv")

# Outlier Removal
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

    

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

print("\nShapiro-Wilk Test")
print("p-value:", p)

if p > 0.05:
    print("Normal distribution")
else:
    print("Not normal")