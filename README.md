# 📊 Student Lifestyle & Performance Analysis

## 📌 Project Overview
This project focuses on analyzing how different lifestyle factors affect the academic performance (CGPA) of students using Data Science techniques.

The analysis is performed on a dataset of 1000 students containing information such as study hours, sleep, screen time, stress levels, attendance, and internal marks.

---

## 🎯 Objective
- Perform Exploratory Data Analysis (EDA)
- Visualize patterns and relationships
- Conduct statistical hypothesis testing
- Apply Simple Linear Regression
- Derive meaningful insights from data

---

## 🧠 Techniques Used
- Data Cleaning (Pandas, NumPy)
- Exploratory Data Analysis (EDA)
- Data Visualization (Matplotlib, Seaborn)
- Correlation Analysis
- Hypothesis Testing:
  - T-Test
  - Shapiro-Wilk Test
- Machine Learning:
  - Simple Linear Regression

---

## 📂 Dataset Information
- Total Records: 1000
- Features: 12
- Type: Mixed (Numerical + Categorical)

### Key Columns:
- Study_Hours_per_Day
- Sleep_Hours
- Screen_Time_Hours
- Attendance_Percentage
- Stress_Level
- Internal_Marks
- CGPA (Target Variable)

---

## 📊 Key Insights
- Study hours have a **positive impact** on CGPA
- Sleep plays an important role in academic performance
- Internal marks show **strong correlation** with CGPA
- Excessive screen time may negatively affect performance
- Academic success depends on multiple factors, not just study time

---

## 📈 Visualizations Included
- CGPA Distribution (Histogram)
- Study Hours vs CGPA (Regression Plot)
- CGPA Boxplot (Outliers)
- CGPA by Branch
- Correlation Heatmap
- Study Category vs CGPA (Bar Chart)

---

## 📉 Machine Learning Model
### Simple Linear Regression
- Predictor: Study Hours per Day
- Target: CGPA

**Equation:**
CGPA = m × Study Hours + c

**Insight:**
Increase in study hours leads to an increase in CGPA, but R² shows other factors also influence performance.

---

## 🧪 Statistical Testing
- **T-Test:** Checked impact of sleep on CGPA
- **Shapiro-Wilk Test:** Checked normality of CGPA distribution

---

## 🛠 Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
