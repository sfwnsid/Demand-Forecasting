# 📦 Retail Demand Forecasting & Stockout Risk Prediction

ML models that predict **(1) whether a product will stock out in the next 7 days** and **(2) how many units customers will demand**, served through an interactive Streamlit app.

## 📊 Results

| Task | Model | Result (unseen future data) |
|---|---|---|
| Stockout risk | Decision Tree | ~85.5% accuracy |
| Demand forecast | Random Forest | R² 0.84, MAE 3.34 (~31% better than the "yesterday's sales" baseline) |

## 🗂️ Dataset

8,772 rows × 45 columns of daily data (2024–2025) across 12 products, 8 categories and 5 suppliers.

## 🧹 Highlights

- Cleaned data, dropped unrelated review columns, filled lag gaps per product
- Removed leaky columns (revenue, profit, units sold)
- Time-based split: first 70% of dates to train, last 30% to test
- SMOTE and scaling applied to training data only
- Compared KNN, Naive Bayes, Decision Tree, Random Forest and Logistic Regression

## 🚀 Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · matplotlib · seaborn · Streamlit
