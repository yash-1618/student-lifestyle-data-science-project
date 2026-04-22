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

print("\n===== CORRELATION =====")
corr = df.corr(numeric_only=True)
print(corr['CGPA'].sort_values(ascending=False))