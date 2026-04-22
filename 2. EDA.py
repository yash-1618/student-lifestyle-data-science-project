import pandas as pd
import numpy as np

df = pd.read_csv("student_lifestyle_performance_dataset.csv")

print("\n===== EDA =====")

print(df.describe())
print("\nSkewness:\n", df.skew(numeric_only=True))

# ==============================
# OUTLIER DETECTION + REMOVAL
# ==============================
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Shape After EDA Outlier Removal:", df.shape)