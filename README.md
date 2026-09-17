# 📊 Customer Churn Prediction System

An end-to-end machine learning system that predicts customer churn using historical usage, billing, and contract data — with an interactive Streamlit dashboard for live and batch predictions.

· Built with Python, scikit-learn, XGBoost, and Streamlit


## Problem

Customer acquisition costs significantly more than retention. This project identifies at-risk customers early — using patterns in contract type, tenure, and billing — so businesses can act before they churn.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data, ~7,000 customer records, Kaggle)

## Approach

1. **Preprocessing** — handled missing `TotalCharges` values, one-hot encoded categorical features
2. **EDA** — analyzed churn patterns across contract types, tenure, and monthly charges
3. **Modeling** — trained and compared 5 model variants:

| Model | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.807 | 0.567 | 0.658 | 0.609 | 0.842 |
| Random Forest | 0.806 | 0.535 | 0.669 | 0.594 | 0.844 |
| Logistic Regression (Balanced) | 0.740 | 0.786 | 0.507 | 0.616 | 0.841 |
| **Random Forest (Balanced)** ⭐ | 0.759 | 0.781 | 0.532 | **0.633** | 0.841 |
| XGBoost (Balanced) | 0.756 | 0.751 | 0.528 | 0.620 | 0.831 |

4. **Final model**: Random Forest with `class_weight='balanced'` — chosen for the best F1-score and strong recall, since missing an actual churner (false negative) is costlier to the business than a false alarm.

## Key Insight

Month-to-month contracts, high monthly charges, and low tenure are the strongest predictors of churn — visualized via feature importance analysis.

## Tech Stack

- **Data & ML**: pandas, numpy, scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn, plotly
- **Frontend**: Streamlit
- **Environment**: Python venv

## Run Locally

```bash
git clone https://github.com/Shiba1856/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
cd app
streamlit run app.py
```

## Project Structure

```
├── data/               # Dataset
├── notebooks/          # EDA + model training notebook
├── models/             # Saved model, scaler, and comparison results
├── app/                # Streamlit frontend
└── README.md
```

## Author

**Shiba Parvin Meraj Ahmed**
 · [LinkedIn](https://www.linkedin.com/in/shiba-parvin-meraj-ahmed-a59465301) · [GitHub](https://github.com/Shiba1856)