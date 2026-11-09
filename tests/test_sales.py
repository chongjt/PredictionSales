import unittest
import numpy as np
import pandas as pd
from sales import demo_transactions, clean_transactions, daily_sales, train_evaluate, forecast_recursive

class TestSales(unittest.TestCase):
    def test_demo_and_clean(self):
        df = clean_transactions(demo_transactions())
        self.assertEqual(df["merchant"].nunique(), 4)
        self.assertFalse(df.empty)

    def test_validation(self):
        with self.assertRaises(ValueError):
            clean_transactions(pd.DataFrame({"date": ["2026-01-01"]}))

    def test_invalid_values(self):
        raw = pd.DataFrame({"date": ["2026-01-01", "bad", "2026-01-02"],
                            "merchant": ["A", "A", "A"], "category": ["Food"] * 3,
                            "amount": [10, 50, -10]})
        self.assertEqual(len(clean_transactions(raw)), 1)

    def test_holdout_and_forecast(self):
        series = daily_sales(clean_transactions(demo_transactions(days=100)))
        model, result, metric, columns = train_evaluate(series, test_days=14)
        self.assertEqual(len(result), 14)
        self.assertTrue(np.isfinite(metric["MAE"]))
        future = forecast_recursive(series, model, columns, 7)
        self.assertEqual(len(future), 7)
        self.assertTrue((future["forecast"] >= 0).all())

if __name__ == "__main__":
    unittest.main()
