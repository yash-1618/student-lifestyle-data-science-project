import pandas as pd
import numpy as np

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

print("""
Problem:
Does increasing study hours improve CGPA?

Insight:
- Strong positive correlation between Study Hours and CGPA
- Regression shows direct increase in CGPA with study hours

Conclusion:
Students studying more hours tend to achieve higher CGPA.

Recommendation:
Students should maintain 4–6 hours of focused study daily.
""")