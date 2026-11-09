# PredictionSales — Machine Learning Sales Prediction

A functional **Streamlit** learning project that explores merchant transactions, aggregates daily sales, trains a Random Forest regression model and visualises holdout performance and illustrative future predictions.

> Sample transactions are **synthetic**, not real merchants or actual sales. The app is a technical portfolio demonstration, not a validated business forecasting product.

## Run locally

Python 3.10+:

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

## CSV format

Upload a CSV containing `date,merchant,category,amount`, where each row represents a transaction or sales record. Example:

```csv
date,merchant,category,amount
2026-01-01,Demo Cafe,Food,89.25
2026-01-02,Demo Cafe,Food,110.50
```

Use **at least 44 calendar days** to create a sufficient lagged dataset, preferably more than 90 days. Invalid dates/amounts, missing categories and negative amounts are removed. Daily sales are summed across all merchants; missing days are interpreted as zero sales. Do not mix currencies or refunds with sales without adapting the cleaning logic.

## Methodology

1. Parse, validate and clean transaction records.
2. Aggregate total daily sales and display category-level revenue.
3. Generate lags at 1/7/14 days, prior-period rolling means, and calendar features.
4. Train Random Forest regression using an **ordered chronological split**; no random shuffling.
5. Compare against a training-set mean predictor and report MAE, RMSE and R² on the held-out tail.
6. Generate a recursive 7–30 day scenario and export results to CSV.

**Important limitation:** The recursive future forecast uses predicted sales as inputs for later days. Error accumulates. This is a point forecast, not a calibrated prediction interval. Results may degrade on seasonal data, structural changes or very short datasets.

## Verify

```bash
python -m unittest discover -s tests -v
python -m compileall app.py sales.py
```

## Project files

- `app.py`: Streamlit interface and Plotly charts
- `sales.py`: data generation, cleaning, temporal features, evaluation and forecasting
- `tests/test_sales.py`: unit tests
- `requirements.txt`: dependencies

## Extension ideas

Use a time-series cross-validation scheme, incorporate holidays/prices/promotions, measure forecast bias by merchant category and compare multiple baselines. To adapt your KNIME Merchant & Sales Performance project, export a clean CSV in the schema above.

**AI use disclosure:** Code was produced with AI assistance. Understand, test and extend it yourself before presenting it as personal technical achievement.
