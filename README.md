# Loan Approval Prediction: A Machine Learning Project  

## **PART 2 – LOAN APPROVAL QUESTION**  

### **INTRODUCTION**  
This study aims to address two key questions:  
1. Which machine learning models can improve classification performance in predicting loan approvals?  
2. What factors significantly influence a bank's decision to approve or reject a loan application?  

The experiment leverages advanced machine learning techniques to enhance loan approval prediction accuracy. Initial analyses are performed using various decision-tree-based models. A stacked ensemble approach is later applied, combining LightGBM, XGBoost, and CatBoost to achieve superior classification performance. A feature importance analysis was conducted to uncover the factors most critical to the decision-making process for loan approvals.  

This analysis offers actionable insights into financial credit evaluations, potentially improving both predictive models and the bank's decision-making framework.

---

### **RAW DATA**  
The following three datasets were used in this study:  
1. **train.csv**  
2. **test.csv**  
3. **credit_risk_dataset.csv**  

The **credit_risk_dataset.csv** serves as the original dataset, while the **train.csv** and **test.csv** were preprocessed versions created by the contest organizer. These preprocessed files feature identical characteristics to the original dataset. To maximize data usage and enhance prediction accuracy, we merged **credit_risk_dataset.csv** with **train.csv**.  

---

### **PROCESS**  

#### **1. Data Exploration**  
- Dataset size: **91,226 entries** with 12 features.  
- Features with missing values: **person_emp_length** and **loan_int_rate**.  
- Observations: Most features have **right-skewed distributions**, such as **person_income**, suggesting a presence of outliers. Others, like **loan_int_rate**, show near-normal distributions.  

Key categorical observations:  
| **Feature**             | **Key Observation**                                       |  
|--------------------------|----------------------------------------------------------|  
| **Home Ownership**       | Majority (47,040 records) are renters.                   |  
| **Loan Intent**          | Education is the most common loan purpose (18,724 records). |  
| **Loan Grade**           | Grade A dominates, indicating high applicant creditworthiness. |  
| **Default Status**       | Majority (76,779) have no default history (good creditworthiness). |  

These insights shaped further preprocessing, feature engineering, and modeling steps.

---

#### **2. Data Preprocessing**  
- **Encoding categorical variables**: Transformed object-type features using `LabelEncoding`.  
- **Outlier handling**: Addressed extreme values with IQR-based truncation to improve data quality.  
- **Missing value imputation**: Used **KNNImputer** to accurately fill missing values for continuous variables.  

---

#### **3. Feature Engineering**  
Created 36 new features by deriving information from seven perspectives critical to loan approval decisions:  

| **Category**                 | **Example Features Created**                         |  
|------------------------------|-----------------------------------------------------|  
| **Income**                   | Income-to-loan ratio, logarithmic transformation.   |  
| **Loan History**             | Historical loan frequency, new borrower flag.       |  
| **Reputation Score**         | Loan-to-history ratio, grade-to-history relationships. |  
| **Guarantees & Mortgages**   | Loan amount vs. house value.                        |  
| **Repayment Capacity**       | Loan amount vs. employment duration.                |  
| **Loan Reasonableness**      | Income mismatches (e.g., high income but renting).  |  
| **Interest Rate Risk**       | Deviation from average interest rate, rate-history ratio. |  

---

#### **4. Model Training and Evaluation**  
Baseline models evaluated:  
| **Model**                 | **Accuracy** | **ROC Score** | **Training Time (s)** |  
|---------------------------|--------------|---------------|-----------------------|  
| RandomForest              | 0.9417       | 0.9360        | 22.72                |  
| XGBoostClassifier         | 0.9464       | 0.9553        | 2.98                 |  
| GradientBoostingClassifier | 0.9398       | 0.9364        | 28.86                |  
| AdaBoostClassifier        | 0.9171       | 0.9196        | 14.53                |  
| BaggingClassifier         | 0.9437       | 0.9419        | 148.57               |  
| LightGBMClassifier        | 0.9479       | 0.9559        | 3.08                 |  
| CatBoostClassifier        | 0.9465       | 0.9536        | 1.97                 |  

The top-performing models, **XGBoost, LightGBM, and CatBoost**, were selected for stacking.  

#### **5. Stacked Meta-Model**  
- **Stacked approach**: Combined **LightGBM**, **XGBoost**, and **CatBoost** using `StackingClassifier`.  
- **Optimization**: Hyperparameters tuned with **Bayesian search** and evaluated with **K-fold cross-validation**.  
- Final model achieved: **ROC score = 0.9581** and ~20 min training time.  

---

### **RESULTS**  

#### Top Influential Features  
| **Feature**             | **Importance** |  
|--------------------------|----------------|  
| **Loan Grade**           | High           |  
| **Person Income**        | Significant    |  
| **Home Ownership**       | Significant    |  
| **Loan Intent**          | Moderate       |  
| **Loan Percent Income**  | Moderate       |  

#### Observations and Insights  
1. **Model Performance**:  
    - **LightGBM** contributed the most within the stacked ensemble.  
    - The ensemble model outperformed individual models on ROC metrics, showcasing the benefits of model stacking.  

2. **Decision Factors for Loan Approvals**:  
    - **Primary Factors**: Credit rating (Loan Grade), borrower financial stability (Person Income, Home Ownership).  
    - **Additional Indicators**: Loan Intent and affordability metrics like Loan Percent Income.  

These insights offer a basis for improving loan approval processes and model-driven decision-making.  

---

### **CONCLUSION**  

This project answers the posed questions:  
1. **Best Models**: LightGBM, XGBoost, and CatBoost achieved strong performance, with the stacking approach yielding the highest accuracy.  
2. **Key Decision Factors**: Borrower's credit history, financial stability, and repayment capacity are critical for loan approvals.  

The findings provide a roadmap for enhancing the precision and fairness of bank lending decisions by employing cutting-edge data science techniques.
