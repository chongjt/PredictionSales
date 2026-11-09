import streamlit as st
import pandas as pd
import plotly.express as px
from sales import demo_transactions, clean_transactions, daily_sales, train_evaluate, forecast_recursive

st.set_page_config(page_title="PredictionSales | Forecast Lab", layout="wide")
st.title("PredictionSales")
st.caption("Machine learning sales prediction · illustrative educational portfolio, not financial advice")
st.info("Demo data is synthetic. Uploaded CSV files are processed in this Streamlit session, not saved by this application.")

with st.sidebar:
    st.header("Data source")
    file = st.file_uploader("Upload transaction CSV", type="csv")
    st.caption("Required columns: date, merchant, category, amount. Amount must be non-negative.")
    test_days = st.slider("Holdout days", 7, 60, 28)
    future_days = st.slider("Forecast horizon (days)", 7, 30, 14)
try:
    raw = pd.read_csv(file) if file else demo_transactions()
    df = clean_transactions(raw)
    if df.empty:
        st.error("No valid data rows found.")
        st.stop()
    st.subheader("Transaction overview")
    a, b, c = st.columns(3)
    a.metric("Total sales", f"${df['amount'].sum():,.2f}")
    b.metric("Transactions / daily records", f"{len(df):,}")
    c.metric("Merchants", df["merchant"].nunique())
    st.plotly_chart(px.bar(df.groupby("category", as_index=False)["amount"].sum(),
                           x="category", y="amount", title="Sales by category"), use_container_width=True)
    st.dataframe(df.head(20), use_container_width=True)

    daily = daily_sales(df)
    model, results, metrics, features = train_evaluate(daily, test_days)
    forecast = forecast_recursive(daily, model, features, future_days)
    st.subheader("Chronological holdout evaluation")
    st.caption("Last observations are held out as test data. No random shuffling; lag features use previous days only.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model MAE", f"${metrics['MAE']:,.2f}")
    m2.metric("Model RMSE", f"${metrics['RMSE']:,.2f}")
    m3.metric("Model R²", f"{metrics['R2']:.3f}")
    m4.metric("Baseline MAE", f"${metrics['Baseline MAE']:,.2f}")
    compare = results.reset_index().melt(id_vars="date", value_vars=["actual", "predicted", "baseline"],
                                          var_name="series", value_name="sales")
    st.plotly_chart(px.line(compare, x="date", y="sales", color="series", title="Test-period predictions vs actuals"),
                    use_container_width=True)
    st.subheader("Future sales scenario")
    st.warning("Recursive point forecasts are uncertain and are not prediction intervals. Performance may degrade with longer horizons.")
    st.plotly_chart(px.line(forecast.reset_index(), x="date", y="forecast",
                            title="Forecast of aggregate daily sales"), use_container_width=True)
    st.download_button("Download forecast CSV", forecast.reset_index().to_csv(index=False),
                       file_name="sales_forecast.csv", mime="text/csv")
    with st.expander("Model and methodology"):
        st.markdown("**Algorithm:** Random Forest regressor with lags (1, 7, 14 days), trailing averages and calendar features. "
                    "**Evaluation:** chronological holdout compared to training-set mean baseline. "
                    "**Limitation:** predictions depend on historical patterns and are not calibrated probabilities.")
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Could not analyse this dataset: {exc}")
