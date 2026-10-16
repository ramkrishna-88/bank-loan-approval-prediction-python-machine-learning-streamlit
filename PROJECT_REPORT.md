# Bank Loan Approval Prediction — Project Report

## 1. Introduction
This project builds a supervised classification system to predict a loan approval class from applicant information.

## 2. Objective
Clean categorical and numerical data, encode categorical variables, train a Random Forest classifier, evaluate it using a confusion matrix and classification metrics, and provide an interactive approval prediction interface.

## 3. Dataset
The Week 4 brief specifies the Kaggle Loan Prediction Dataset. The runnable project follows the common loan-prediction schema including Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area and Loan_Status.

## 4. Workflow
Data cleaning → missing-value handling → one-hot encoding → train/test split → Random Forest → confusion matrix and classification report → new application prediction.

## 5. Model
Random Forest is used as the primary implementation requested by the brief. The pipeline handles numerical and categorical fields separately.

## 6. Evaluation
The application reports accuracy and a confusion matrix. A classification report is also displayed for additional analysis.

## 7. Limitations
The included CSV is a runnable demo dataset, not a claim that it is the official Kaggle data. For an academic submission, upload/use the official Kaggle Loan Prediction Dataset. Real lending requires validated financial and regulatory processes.

## 8. Future Scope
Compare XGBoost, tune hyperparameters, use cross-validation, add explainability, handle class imbalance carefully, and integrate validated financial data.

## 9. Conclusion
The project implements the practical classification workflow specified for Week 4.
