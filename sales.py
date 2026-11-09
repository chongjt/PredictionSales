"""Sales modelling helpers. Demo data is synthetic, not real merchant data."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED = {"date", "merchant", "category", "amount"}


def demo_transactions(seed=42, days=240):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    for i, day in enumerate(dates):
        for merchant, category, base in [("North Cafe", "Food", 550), ("Metro Tech", "Electronics", 950),
                                          ("City Shop", "Retail", 680), ("Green Market", "Groceries", 730)]:
            trend = 1 + i / days * (0.22 if category in ("Food", "Groceries") else 0.08)
            weekend = 1.30 if day.dayofweek >= 5 else 1
            amount = max(0, rng.normal(base * trend * weekend, base * 0.16))
            rows.append((day, merchant, category, round(amount, 2)))
    return pd.DataFrame(rows, columns=["date", "merchant", "category", "amount"])


def clean_transactions(data):
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    result = data[list(REQUIRED)].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["amount"] = pd.to_numeric(result["amount"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    result["merchant"] = result["merchant"].astype("string").str.strip()
    result["category"] = result["category"].astype("string").str.strip()
    result = result.dropna(subset=["date", "amount", "merchant", "category"])
    result = result[(result["amount"] >= 0) & (result["merchant"] != "") & (result["category"] != "")]
    return result.sort_values("date").reset_index(drop=True)


def daily_sales(transactions):
    if transactions.empty:
        raise ValueError("No valid transactions found")
    series = (transactions.set_index("date")["amount"]
              .resample("D").sum().asfreq("D", fill_value=0))
    if len(series) < 28:
        raise ValueError("At least 28 calendar days of data are required")
    return series


def create_features(series, horizon=14):
    """Lags and rolling averages use only data strictly before each target day."""
    df = pd.DataFrame({"sales": series.astype(float)})
    for lag in (1, 7, 14):
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    df["rolling_7"] = df["sales"].shift(1).rolling(7).mean()
    df["rolling_14"] = df["sales"].shift(1).rolling(14).mean()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    return df.dropna()


def train_evaluate(series, test_days=28, seed=42):
    dataset = create_features(series)
    if len(dataset) < 30:
        raise ValueError("Need at least 44 calendar days to train and evaluate")
    test_days = min(int(test_days), max(7, len(dataset) // 3))
    train, test = dataset.iloc[:-test_days], dataset.iloc[-test_days:]
    columns = [c for c in dataset if c != "sales"]
    model = RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    baseline = DummyRegressor(strategy="mean")
    model.fit(train[columns], train["sales"])
    baseline.fit(train[columns], train["sales"])
    predicted = model.predict(test[columns])
    base_pred = baseline.predict(test[columns])
    results = test[["sales"]].rename(columns={"sales": "actual"}).copy()
    results["predicted"] = predicted
    results["baseline"] = base_pred
    metrics = {
        "MAE": float(mean_absolute_error(results["actual"], predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(results["actual"], predicted))),
        "R2": float(r2_score(results["actual"], predicted)),
        "Baseline MAE": float(mean_absolute_error(results["actual"], base_pred)),
    }
    return model, results, metrics, columns


def forecast_recursive(series, model, columns, days=14):
    """Illustrative recursive forecast; later lags include earlier predictions."""
    history = series.astype(float).copy()
    predictions = []
    for day in pd.date_range(history.index.max() + pd.Timedelta(days=1), periods=days, freq="D"):
        sample = {
            "lag_1": float(history.iloc[-1]),
            "lag_7": float(history.iloc[-7]),
            "lag_14": float(history.iloc[-14]),
            "rolling_7": float(history.iloc[-7:].mean()),
            "rolling_14": float(history.iloc[-14:].mean()),
            "day_of_week": day.dayofweek,
            "month": day.month,
        }
        value = max(0.0, float(model.predict(pd.DataFrame([sample], columns=columns))[0]))
        history.loc[day] = value
        predictions.append({"date": day, "forecast": value})
    return pd.DataFrame(predictions).set_index("date")
