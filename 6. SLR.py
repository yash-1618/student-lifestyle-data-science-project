import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

X = df[['Study_Hours_per_Day']]
y = df['CGPA']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R² Score:", r2_score(y, y_pred))

print(f"CGPA = {model.coef_[0]:.3f} * Study_Hours + {model.intercept_:.3f}")