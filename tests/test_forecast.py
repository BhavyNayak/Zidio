import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.forecast import calculate_wape, calculate_bias, build_seasonal_naive_forecast

def test_metrics_calculation():
    # Simple arrays
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    forecast = np.array([12.0, 18.0, 35.0, 38.0])
    
    # Absolute errors: 2, 2, 5, 2 -> sum = 11
    # Actual sum = 100
    # WAPE = 11 / 100 = 0.11
    wape = calculate_wape(actual, forecast)
    assert np.isclose(wape, 0.11), f"WAPE metric calculation is incorrect: {wape}"
    
    # Net errors: +2, -2, +5, -2 -> sum = +3
    # Bias = 3 / 100 = 0.03
    bias = calculate_bias(actual, forecast)
    assert np.isclose(bias, 0.03), f"Bias metric calculation is incorrect: {bias}"

def test_seasonal_naive_baseline():
    # Mock data
    train_records = []
    # Generate 4 weeks of training data for a single SKU
    for day in range(28):
        dow = day % 7
        # Monday sales = 10, other days = 2
        sales = 10.0 if dow == 0 else 2.0
        train_records.append({
            "sku_id": "SKU001",
            "date": (datetime(2026, 1, 1) + timedelta(days=day)).strftime("%Y-%m-%d"),
            "day_of_week": dow,
            "units_sold": sales
        })
    df_train = pd.DataFrame(train_records)
    
    # Test data: 1 week (7 days)
    test_records = []
    for day in range(28, 35):
        test_records.append({
            "sku_id": "SKU001",
            "date": (datetime(2026, 1, 1) + timedelta(days=day)).strftime("%Y-%m-%d"),
            "day_of_week": day % 7,
            "units_sold": 0.0 # placeholder
        })
    df_test = pd.DataFrame(test_records)
    
    # Run baseline forecast
    forecasts = build_seasonal_naive_forecast(df_train, df_test)
    
    # Predictions should repeat: Monday (index 0 in test, which is day 28 -> 28%7 = 0) is 10.0, others are 2.0
    assert np.isclose(forecasts[0], 10.0)
    assert np.isclose(forecasts[1], 2.0)
    assert np.isclose(forecasts[6], 2.0)

def test_forecast_output_integrity():
    # Check if forecast outputs were generated correctly
    forecast_path = "data/processed/future_forecast.csv"
    artifacts_path = "src/models/forecast_artifacts.pkl"
    
    assert os.path.exists(forecast_path), "future_forecast.csv is missing!"
    assert os.path.exists(artifacts_path), "forecast_artifacts.pkl is missing!"
    
    # Check forecast columns
    df_fc = pd.read_csv(forecast_path)
    assert "date" in df_fc.columns
    assert "sku_id" in df_fc.columns
    assert "forecast_units" in df_fc.columns
    assert "forecast_lower_80" in df_fc.columns
    assert "forecast_upper_80" in df_fc.columns
    
    # Verify forecast values logic (lower bound <= forecast <= upper bound)
    assert (df_fc["forecast_lower_80"] <= df_fc["forecast_units"]).all(), "Forecast lower bounds are not <= forecast point estimates!"
    assert (df_fc["forecast_units"] <= df_fc["forecast_upper_80"]).all(), "Forecast point estimates are not <= forecast upper bounds!"
    
    # Load model artifacts
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
        
    assert "model" in artifacts
    assert "use_baseline" in artifacts
    assert "std_residuals" in artifacts
    assert isinstance(artifacts["std_residuals"], float)
