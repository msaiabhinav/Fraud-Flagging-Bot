# 🚗 Auto Insurance Fraud Detection Bot

## 📌 Overview
This project builds an auto insurance fraud detection system enhanced with a conversational bot that explains model predictions in plain English. The system identifies high-risk insurance claims and provides interpretable insights to support claims adjusters in making informed investigation decisions.

The focus is not only on predictive accuracy, but also on explainability and real-world usability.

---

## 🎯 Business Objective
- Detect potentially fraudulent auto insurance claims
- Minimize missed fraud cases (high recall)
- Reduce manual review workload
- Provide transparent, explainable predictions for claims teams

---

## 🧠 Modeling Approach
- Binary classification (Fraud vs Non-Fraud)
- Multiple models evaluated and compared
- Logistic Regression selected as the final model due to strong performance and interpretability
- Threshold-based decisioning to support investigation prioritization

---

## 🤖 Bot Functionality
The bot translates model outputs into clear, human-readable explanations and can answer questions such as:
- Why was this claim flagged as high risk?
- Which factors contributed most to the fraud prediction?
- Should this claim be investigated manually?
- What evidence should an adjuster review first?

---

## 🔍 Key Fraud Indicators
- High claim payout relative to vehicle value
- Short duration between policy inception and claim
- Partial or disputed liability
- Inconsistent claim details
- Vehicle age and damage severity patterns

---

## 📈 Results

### Validation Results (Threshold = 0.5)
Best Model: Logistic Regression
- ROC-AUC: 0.8981
- Accuracy: 0.89
- Precision (Fraud = 1): 0.7188
- Recall (Fraud = 1): 0.92
- F1 Score (Fraud = 1): 0.8070
- Confusion Matrix: TP=23, FP=9, TN=66, FN=2

### Test Results
- ROC-AUC: 0.8627
- Accuracy: 0.8440
- Precision (Fraud = 1): 0.6575
- Recall (Fraud = 1): 0.7742
- F1 Score (Fraud = 1): 0.7111
- Confusion Matrix: TP=48, FP=25, TN=163, FN=14

These results demonstrate strong discriminatory power while prioritizing recall to reduce missed fraud cases.

---

## 🛠️ Tools & Technologies
- Python
- Scikit-learn
- Logistic Regression, Tree-based models
- Model evaluation using ROC-AUC, Precision, Recall, F1
- Jupyter Notebook for experimentation

---

## 💡 Business Impact
- Improves fraud triage by prioritizing high-risk claims
- Reduces dependency on manual rule-based screening
- Enhances trust in model decisions through explainability
- Supports faster and more consistent claims handling

---

## 🚀 Future Enhancements
- SHAP-based feature attribution
- LLM-powered natural language explanations
- API deployment for real-time scoring
- Integration with claims management systems

---


