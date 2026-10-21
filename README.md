# 📡 Customer Churn Prediction: Feature Engineering & Selection Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning study using the **Telco Customer Churn** dataset. This project investigates how domain-driven **feature engineering** and model-based **feature selection** impact model interpretability and predictive performance, especially when identifying the minority churn class.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Summary](#-dataset-summary)
- [Workflow & Methodology](#-workflow--methodology)
  - [1. Data Cleaning & Preprocessing](#1-data-cleaning--preprocessing)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Preprocessing & Modeling Pipelines](#3-preprocessing--modeling-pipelines)
  - [4. Feature Selection](#4-feature-selection)
- [Model Evaluation & Comparison](#-model-evaluation--comparison)
- [Key Insights & Feature Importances](#-key-insights--feature-importances)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)


---

## 🚀 Project Overview

Customer churn prediction is a critical task for telecommunications providers aiming to proactively retain revenue and subscribers. While complex models are common, data preprocessing and feature representation often drive real-world performance gains.

### Primary Objectives:
1. **Benchmark Performance**: Build a baseline classification pipeline using raw, cleaned features.
2. **Engineer Intelligent Features**: Formulate new metrics (e.g., tenure cohorts, service aggregation, payment velocity) grounded in business logic.
3. **Quantify Performance Gains**: Compare baseline and feature-engineered models, prioritizing minority class ($Churn = 1$) metrics like Precision, Recall, and F1-score.
4. **Evaluate Feature Selection**: Assess whether dimensionality reduction using `SelectFromModel` (Random Forest importance) preserves predictive power while pruning features.

---

## 📊 Dataset Summary

The dataset used is the standard **Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), containing **7,043 rows** and **21 columns**.

* **Target Variable**: `Churn` (Converted to binary: `1` for Yes, `0` for No).
* **Class Distribution**:
  * Retained (`0`): ~5,174 customers (~73.5%)
  * Churned (`1`): ~1,869 customers (~26.5%)
  * *Reflects a moderately imbalanced classification setting.*

---

## ⚙️ Workflow & Methodology

```
Raw Data (CSV)
   │
   ▼
[Data Cleaning] ──► Numeric Coercion & Median Imputation (TotalCharges)
   │
   ▼
[Feature Engineering] ──► Tenure Binning, Service Bundling, Charge Ratios
   │
   ▼
[Scikit-Learn Pipeline] ──► ColumnTransformer (StandardScaler + OneHotEncoder)
   │
   ├──► Baseline Model (Logistic Regression on Raw Data)
   ├──► Enhanced Model (Logistic Regression on Engineered Features)
   └──► Feature Selection Model (SelectFromModel via Random Forest)
```

### 1. Data Cleaning & Preprocessing
* **Type Normalization**: The `TotalCharges` field contained whitespace values stored as strings. These were coerced to float (`errors='coerce'`), creating 11 missing entries imputed using the column **median**.
* **Target Encoding**: Mapped binary labels: `Yes -> 1`, `No -> 0`.
* **Identifier Handling**: Dropped uninformative identifier columns (`customerID`) prior to training.

### 2. Feature Engineering
Four primary transformations were implemented:
1. **Tenure Discretization (`tenure_group`)**: Continuous `tenure` (0–72 months) was binned into 5 business-friendly cohorts:
   * `0-1 Year`, `1-2 Years`, `2-4 Years`, `4-5 Years`, and `5+ Years`.
2. **Category Simplification**: Collapsed sub-categories such as `"No phone service"` and `"No internet service"` into clean `"No"` values across relevant service columns.
3. **Service Bundling (`num_add_services`)**: Aggregated count of subscribed optional add-on services:
   * *Features evaluated*: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
4. **Monthly Charge Ratio (`monthly_charge_ratio`)**: Calculated expense relative to tenure:
   $$\text{monthly\_charge\_ratio} = \frac{\text{MonthlyCharges}}{\text{tenure} + 1}$$
   *(Added 1 to avoid zero-division for new sign-ups).*

### 3. Preprocessing & Modeling Pipelines
To prevent data leakage, all transformations were bundled into unified Scikit-Learn pipelines:
* **Numerical Pipeline**: `StandardScaler()`
* **Categorical Pipeline**: `OneHotEncoder(handle_unknown='ignore')`
* **Classifier**: `LogisticRegression(max_iter=1000, random_state=42)` evaluated via an 80/20 stratified split (`stratify=y`).

### 4. Feature Selection
* **Algorithm**: `SelectFromModel` using a `RandomForestClassifier(random_state=42)` as the base estimator.
* **Criterion**: Retained features whose importance exceeded the **median** importance threshold.

---

## 📈 Model Evaluation & Comparison

Performance on the test set ($n = 1,409$):

| Model Configuration | Overall Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Baseline Model** (Raw Cleaned) | **0.80** | 0.65 | **0.56** | **0.60** |
| **2. Enhanced Model** (With Engineered Features) | **0.81** | **0.68** | 0.52 | **0.59** |
| **3. Selected Features Model** (`SelectFromModel`) | **0.81** | **0.68** | 0.52 | **0.59** |

### Key Observations:
* **Precision Lift**: The engineered features increased precision on churners from **65% to 68%**, reducing false positive churn alerts.
* **Feature Selection Reality Check**: Pruning features using the Random Forest median importance threshold maintained the accuracy (81%) and F1-score (0.59), showing that model complexity can be controlled without sacrificing primary performance. However, aggressive pruning must be tuned carefully to avoid reducing recall.

---

## 🔍 Key Insights & Feature Importances

Using a `RandomForestClassifier` trained on the engineered set, feature importances were extracted and analyzed:

1. **Top Predictive Drivers**:
   * `TotalCharges` and `MonthlyCharges`: Overall monetary spend remains a primary churn indicator.
   * `monthly_charge_ratio`: Customers with high early charges relative to tenure show higher churn likelihood.
   * `Contract`: Month-to-month contracts strongly correlate with churn compared to one- or two-year commitments.
   * `tenure_group`: Newer customers (`0-1 Year`) represent the highest risk retention window.
   * `num_add_services`: Customers with multiple add-ons show higher switching barriers and lower churn rates.

---


## 💻 Installation & Setup

### Prerequisites
* Python 3.8+ 
* Jupyter Notebook or Google Colab

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/customer-churn-feature-engineering.git
   cd customer-churn-feature-engineering
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   *Or manually install the core libraries:*
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

4. **Launch the Notebook:**
   ```bash
   jupyter notebook notebooks/customer_churn_feature_engineering.ipynb
   ```

