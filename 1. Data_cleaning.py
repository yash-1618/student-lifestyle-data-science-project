import pandas as pd
import numpy as np

df = pd.read_csv("student_lifestyle_performance_dataset.csv")

print("\n===== DATA OVERVIEW =====")
print(df.head())
print("\nShape:", df.shape)
print("\nData Types:\n", df.dtypes)

# ==============================
# DATA CLEANING
# ==============================
print("\n===== DATA CLEANING =====")

print("Missing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

# Convert categorical
df['Branch'] = df['Branch'].astype('category')
df['Diet_Type'] = df['Diet_Type'].astype('category')
df['Residence'] = df['Residence'].astype('category')

# ==============================
# OUTLIER DETECTION + REMOVAL
# ==============================
print("\n===== OUTLIER REMOVAL =====")

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("New Shape After Outlier Removal:", df.shape)
