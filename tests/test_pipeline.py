import os
import pandas as pd
import numpy as np
from src.pipeline import run_pipeline

def test_pipeline_execution_and_schema():
    # 1. Execute pipeline (creates processed files)
    run_pipeline()
    
    processed_path = "data/processed/clean_sales.csv"
    assert os.path.exists(processed_path), "Pipeline output file does not exist!"
    
    # 2. Load dataset
    df = pd.read_csv(processed_path)
    
    # Check shape
    assert len(df) > 0, "Clean sales file is empty!"
    assert df.shape[1] >= 20, "Clean sales file is missing columns!"
    
    # 3. Check critical columns
    expected_cols = [
        "sku_id", "date", "units_sold", "revenue", "unit_price", "promo_flag",
        "units_sold_lag_1", "units_sold_lag_7", "units_sold_lag_28",
        "units_sold_roll_mean_7", "units_sold_roll_std_7",
        "units_sold_roll_mean_28", "units_sold_roll_std_28",
        "category", "subcategory", "unit_cost", "list_price",
        "day_of_week", "day_of_month", "month", "quarter"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col} is missing from processed file!"
        
    # 4. Verify no NaN values in critical features
    features_to_check = [
        "units_sold_lag_1", "units_sold_lag_7", "units_sold_lag_28",
        "units_sold_roll_mean_7", "units_sold_roll_std_7",
        "units_sold_roll_mean_28", "units_sold_roll_std_28"
    ]
    for feat in features_to_check:
        assert df[feat].isnull().sum() == 0, f"Feature {feat} contains null values!"
        
    # 5. Verify no duplicate entries for (sku, date)
    duplicate_keys = df.duplicated(subset=["sku_id", "date"]).sum()
    assert duplicate_keys == 0, f"Deduplication failed! Found {duplicate_keys} duplicate sku-date pairs."

def test_no_data_leakage():
    processed_path = "data/processed/clean_sales.csv"
    df = pd.read_csv(processed_path)
    
    # Test a specific SKU to ensure lag_1 equals units_sold shifted by 1
    sample_sku = df["sku_id"].iloc[0]
    sku_df = df[df["sku_id"] == sample_sku].sort_values("date").reset_index(drop=True)
    
    for i in range(1, len(sku_df)):
        actual_prev_sales = sku_df.loc[i - 1, "units_sold"]
        engineered_lag_1 = sku_df.loc[i, "units_sold_lag_1"]
        assert actual_prev_sales == engineered_lag_1, f"Data leakage or lag mismatch: Row {i} lag_1 is {engineered_lag_1}, but previous units_sold was {actual_prev_sales}"
        
        # Test rolling mean of last 7 days
        if i >= 7:
            recent_sales = sku_df.loc[i-7:i-1, "units_sold"].values
            expected_mean_7 = np.mean(recent_sales)
            engineered_mean_7 = sku_df.loc[i, "units_sold_roll_mean_7"]
            # Assert close due to floating point precision
            assert np.isclose(expected_mean_7, engineered_mean_7), f"Data leakage or rolling mean mismatch: Row {i} mean_7 is {engineered_mean_7}, expected {expected_mean_7}"
