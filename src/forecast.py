import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingRegressor

# Primary Evaluation Metrics
def calculate_wape(actual, forecast):
    sum_actual = np.sum(actual)
    if sum_actual == 0:
        return 0.0
    return np.sum(np.abs(actual - forecast)) / sum_actual

def calculate_bias(actual, forecast):
    sum_actual = np.sum(actual)
    if sum_actual == 0:
        return 0.0
    return np.sum(forecast - actual) / sum_actual

def build_seasonal_naive_forecast(train_df, test_df):
    """
    Seasonal-Naive baseline: For each SKU and date, forecast is the average of units_sold
    on the same day of week over the last 4 weeks in training data.
    """
    forecasts = []
    # Pre-calculate day_of_week means for each SKU
    # Group by sku_id and day_of_week on the last 28 days of training data to capture recent seasonality
    train_df = train_df.copy()
    train_df["date_dt"] = pd.to_datetime([datetime.strptime(x, "%Y-%m-%d") for x in train_df["date"].astype(str)])
    max_train_date = train_df["date_dt"].max()
    recent_train = train_df[train_df["date_dt"] > (max_train_date - timedelta(days=28))]
    
    # Baseline dict: sku_id -> day_of_week -> average sales
    baseline_lookup = recent_train.groupby(["sku_id", "day_of_week"])["units_sold"].mean().to_dict()
    # Overall SKU average as fallback
    sku_lookup = recent_train.groupby("sku_id")["units_sold"].mean().to_dict()
    # Global average as final fallback
    global_mean = recent_train["units_sold"].mean() if len(recent_train) > 0 else 0.0
    
    for idx, row in test_df.iterrows():
        sku = row["sku_id"]
        dow = row["day_of_week"]
        
        val = baseline_lookup.get((sku, dow), sku_lookup.get(sku, global_mean))
        forecasts.append(max(0.0, val))
        
    return np.array(forecasts)

def prepare_ml_features(df, is_train=True):
    """
    Prepares features for the HistGradientBoostingRegressor.
    Handles categorical columns by one-hot encoding or integer mapping.
    """
    # Features list
    feature_cols = [
        "promo_flag", "is_holiday", "day_of_week", "day_of_month", "quarter", "month",
        "units_sold_lag_1", "units_sold_lag_7", "units_sold_lag_28",
        "units_sold_roll_mean_7", "units_sold_roll_std_7",
        "units_sold_roll_mean_28", "units_sold_roll_std_28"
    ]
    
    # Categorical columns to encode: category, subcategory, season, promo_event
    categorical_cols = ["category", "subcategory", "season", "promo_event"]
    
    df_encoded = df.copy()
    # For HistGradientBoostingRegressor, we can convert categories to category type
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype("category")
            
    # We will return the dataframe with categorical types and numerical features
    all_features = feature_cols + [c for c in categorical_cols if c in df_encoded.columns]
    
    if is_train:
        return df_encoded[all_features], df_encoded["units_sold"]
    else:
        return df_encoded[all_features]

def run_backtest(df):
    """
    Rolling-Origin Cross-Validation.
    We will use 3 rolling folds of 6 weeks (42 days) each.
    """
    print("\nStarting rolling-origin cross-validation (backtesting)...")
    df = df.copy()
    df["date_dt"] = pd.to_datetime([datetime.strptime(x, "%Y-%m-%d") for x in df["date"].astype(str)])
    
    # Unique dates sorted
    unique_dates = sorted(df["date_dt"].unique())
    n_days = len(unique_dates)
    
    # Define test folds
    fold_horizon = 42 # 6 weeks
    folds = [
        # Fold 1: train days 0 to n_days - 3*fold_horizon, test next fold_horizon days
        (n_days - 3 * fold_horizon, n_days - 2 * fold_horizon),
        # Fold 2: train days 0 to n_days - 2*fold_horizon, test next fold_horizon days
        (n_days - 2 * fold_horizon, n_days - fold_horizon),
        # Fold 3: train days 0 to n_days - fold_horizon, test next fold_horizon days
        (n_days - fold_horizon, n_days)
    ]
    
    results = []
    
    for fold_idx, (train_cutoff_idx, test_end_idx) in enumerate(folds):
        train_cutoff_date = unique_dates[train_cutoff_idx]
        test_end_date = unique_dates[test_end_idx - 1]
        
        train_df = df[df["date_dt"] < train_cutoff_date]
        test_df = df[(df["date_dt"] >= train_cutoff_date) & (df["date_dt"] <= test_end_date)]
        
        print(f"Fold {fold_idx + 1}: Train up to {train_cutoff_date.date()}, Test {train_cutoff_date.date()} to {test_end_date.date()}")
        
        # 1. Baseline
        baseline_forecast = build_seasonal_naive_forecast(train_df, test_df)
        
        # 2. ML Model
        X_train, y_train = prepare_ml_features(train_df)
        X_test, y_test = prepare_ml_features(test_df)
        
        # Train Scikit-Learn HistGradientBoostingRegressor
        # We specify categorical features for optimal performance
        cat_features_indices = [X_train.columns.get_loc(col) for col in ["category", "subcategory", "season", "promo_event"] if col in X_train.columns]
        
        model = HistGradientBoostingRegressor(categorical_features=cat_features_indices, random_state=42)
        model.fit(X_train, y_train)
        
        model_forecast = model.predict(X_test)
        # Bounded at 0.0
        model_forecast = np.clip(model_forecast, 0.0, None)
        
        # Calculate WAPE and Bias
        baseline_wape = calculate_wape(y_test.values, baseline_forecast)
        baseline_bias = calculate_bias(y_test.values, baseline_forecast)
        
        model_wape = calculate_wape(y_test.values, model_forecast)
        model_bias = calculate_bias(y_test.values, model_forecast)
        
        print(f"  Baseline -> WAPE: {baseline_wape:.4f}, Bias: {baseline_bias:.4f}")
        print(f"  Model    -> WAPE: {model_wape:.4f}, Bias: {model_bias:.4f}")
        
        results.append({
            "fold": fold_idx + 1,
            "baseline_wape": baseline_wape,
            "baseline_bias": baseline_bias,
            "model_wape": model_wape,
            "model_bias": model_bias,
            "residuals": y_test.values - model_forecast
        })
        
    avg_baseline_wape = np.mean([r["baseline_wape"] for r in results])
    avg_model_wape = np.mean([r["model_wape"] for r in results])
    
    print("\n--- Cross-Validation Summary ---")
    print(f"Average Baseline WAPE: {avg_baseline_wape:.4f}")
    print(f"Average Model WAPE:    {avg_model_wape:.4f}")
    
    # Determine the winning model
    # If the ML model does not beat the baseline, we will set a flag to use baseline instead.
    use_baseline_flag = False
    if avg_model_wape >= avg_baseline_wape:
        print("WARNING: ML model did not outperform seasonal-naive baseline. Baseline will be chosen for predictions.")
        use_baseline_flag = True
    else:
        print("SUCCESS: ML model outperformed seasonal-naive baseline. ML model will be chosen for predictions.")
        
    # Combine residuals to estimate prediction interval standard deviation
    all_residuals = np.concatenate([r["residuals"] for r in results])
    std_residuals = np.std(all_residuals)
    
    return use_baseline_flag, std_residuals

def recursive_forecast(df, model, use_baseline, std_residuals, forecast_horizon=42):
    """
    Generates daily forecasts for the next 42 days (6 weeks) starting from the end of history.
    Uses recursive forecast: updates lags and rolling stats dynamically for future predictions.
    """
    df = df.copy()
    df["date_dt"] = pd.to_datetime([datetime.strptime(x, "%Y-%m-%d") for x in df["date"].astype(str)])
    
    # End of history is 2026-06-30
    history_end = datetime(2026, 6, 30)
    
    # Find active SKUs (all unique SKUs)
    active_skus = df["sku_id"].unique()
    
    # We will build a forecast DataFrame containing predictions for future dates
    future_dates = [history_end + timedelta(days=x) for x in range(1, forecast_horizon + 1)]
    future_dates_str = [d.strftime("%Y-%m-%d") for d in future_dates]
    
    # Let's read calendar file for future features
    df_calendar = pd.read_csv("data/raw/calendar.csv")
    df_calendar["date"] = df_calendar["date"].astype(str)
    
    df_sku = pd.read_csv("data/raw/sku_master.csv")
    
    # We need to construct a rolling history to feed the recursive forecast
    # We'll start with the tail of the historical dataset (last 30 days of history for each SKU)
    history_tail = df[df["date_dt"] > (history_end - timedelta(days=60))].copy()
    
    # Keep track of simulated rows
    sim_df = history_tail.copy()
    
    for d_str in future_dates_str:
        # Precompute date components to bypass pd.to_datetime Cython bug
        d_val = datetime.strptime(d_str, "%Y-%m-%d")
        dow = d_val.weekday()
        dom = d_val.day
        q = (d_val.month - 1) // 3 + 1
        m = d_val.month
        
        # Generate prediction day-by-day
        # For date d_str, for all SKUs:
        # 1. Extract the features for d_str.
        # Wait, the lag features and rolling stats must be calculated dynamically.
        # Let's append placeholder rows for this date to sim_df, calculate their lag/rolling features,
        # then run the model, write the prediction back, and loop.
        
        day_rows = []
        for sku_id in active_skus:
            # Get calendar info for this date
            cal_row = df_calendar[df_calendar["date"] == d_str].iloc[0]
            sku_row = df_sku[df_sku["sku_id"] == sku_id].iloc[0]
            
            # Combine static catalog + calendar properties
            day_rows.append({
                "date": d_str,
                "sku_id": sku_id,
                "category": sku_row["category"],
                "subcategory": sku_row["subcategory"],
                "unit_cost": sku_row["unit_cost"],
                "list_price": sku_row["list_price"],
                "promo_flag": 1 if cal_row["promo_event"] != "None" else 0, # rough estimate
                "is_holiday": cal_row["is_holiday"],
                "promo_event": cal_row["promo_event"],
                "season": cal_row["season"],
                "units_sold": 0.0, # placeholder
                "day_of_week": dow,
                "day_of_month": dom,
                "quarter": q,
                "month": m
            })
            
        df_today = pd.DataFrame(day_rows)
        sim_df = pd.concat([sim_df, df_today], ignore_index=True)
        sim_df.sort_values(by=["sku_id", "date"], inplace=True)
        sim_df.reset_index(drop=True, inplace=True)
        
        # Recalculate features ONLY for the current date (d_str) rows
        # We group by sku_id to compute shifted values
        grouped = sim_df.groupby("sku_id")
        
        # We can update the entire DataFrame's lags/rolling for speed, or just the current date
        sim_df["units_sold_lag_1"] = grouped["units_sold"].shift(1)
        sim_df["units_sold_lag_7"] = grouped["units_sold"].shift(7)
        sim_df["units_sold_lag_28"] = grouped["units_sold"].shift(28)
        
        sim_df["units_sold_roll_mean_7"] = grouped["units_sold"].transform(lambda x: x.shift(1).rolling(7).mean())
        sim_df["units_sold_roll_std_7"] = grouped["units_sold"].transform(lambda x: x.shift(1).rolling(7).std())
        sim_df["units_sold_roll_mean_28"] = grouped["units_sold"].transform(lambda x: x.shift(1).rolling(28).mean())
        sim_df["units_sold_roll_std_28"] = grouped["units_sold"].transform(lambda x: x.shift(1).rolling(28).std())
        
        # Fill NaNs
        for col in ["units_sold_lag_1", "units_sold_lag_7", "units_sold_lag_28",
                    "units_sold_roll_mean_7", "units_sold_roll_std_7",
                    "units_sold_roll_mean_28", "units_sold_roll_std_28"]:
            sim_df[col] = sim_df[col].fillna(0.0)
            
        # Extract calendar components (already assigned in day_rows)
        pass
        
        # Now predict for the current date
        today_mask = sim_df["date"] == d_str
        df_predict_today = sim_df[today_mask].copy()
        
        if use_baseline:
            # Baseline forecast logic:
            # Grab average of units_sold on the same day_of_week for this SKU over the last 28 days
            forecasts = []
            for idx, row in df_predict_today.iterrows():
                sku = row["sku_id"]
                dow = row["day_of_week"]
                # Filter sim_df for historical rows of this SKU and same day of week
                hist_rows = sim_df[(sim_df["sku_id"] == sku) & 
                                   (sim_df["date"] < d_str) & 
                                   (sim_df["day_of_week"] == dow)]
                # Take last 4 occurrences
                val = hist_rows.tail(4)["units_sold"].mean() if len(hist_rows) > 0 else 0.0
                forecasts.append(max(0.0, val))
            sim_df.loc[today_mask, "units_sold"] = forecasts
        else:
            # ML model prediction
            X_today = prepare_ml_features(df_predict_today, is_train=False)
            predictions = model.predict(X_today)
            predictions = np.clip(predictions, 0.0, None)
            sim_df.loc[today_mask, "units_sold"] = predictions
            
    # Extract only the forecast range from sim_df
    forecast_df = sim_df[sim_df["date"].isin(future_dates_str)].copy()
    
    # Add confidence intervals
    # 80% confidence interval uses z = 1.28
    # lower_bound = max(0, point_estimate - 1.28 * std_residuals)
    # upper_bound = point_estimate + 1.28 * std_residuals
    forecast_df["forecast_units"] = forecast_df["units_sold"]
    forecast_df["forecast_lower_80"] = np.clip(forecast_df["forecast_units"] - 1.28 * std_residuals, 0.0, None)
    forecast_df["forecast_upper_80"] = forecast_df["forecast_units"] + 1.28 * std_residuals
    
    return forecast_df[["date", "sku_id", "forecast_units", "forecast_lower_80", "forecast_upper_80"]]

def train_and_save_model():
    # Ingest processed dataset
    processed_path = "data/processed/clean_sales.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Clean sales file not found at {processed_path}. Run pipeline first!")
        
    df = pd.read_csv(processed_path)
    
    # 1. Backtest
    use_baseline, std_residuals = run_backtest(df)
    
    # 2. Train final model on full history
    print("\nTraining final model on full historical dataset...")
    X_train, y_train = prepare_ml_features(df)
    
    cat_features_indices = [X_train.columns.get_loc(col) for col in ["category", "subcategory", "season", "promo_event"] if col in X_train.columns]
    
    final_model = HistGradientBoostingRegressor(categorical_features=cat_features_indices, random_state=42)
    final_model.fit(X_train, y_train)
    
    # Create models directory
    os.makedirs("src/models", exist_ok=True)
    
    # Save model and artifacts
    artifacts = {
        "model": final_model,
        "use_baseline": use_baseline,
        "std_residuals": std_residuals,
        "features": list(X_train.columns)
    }
    
    with open("src/models/forecast_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print("Saved forecasting artifacts to src/models/forecast_artifacts.pkl")
    
    # 3. Generate future forecast for the next 6 weeks (2026-07-01 to 2026-08-11)
    print("Generating 6-week future forecast...")
    df_forecast = recursive_forecast(df, final_model, use_baseline, std_residuals, forecast_horizon=42)
    
    # Save forecast results to data/processed
    df_forecast.to_csv("data/processed/future_forecast.csv", index=False)
    print(f"Saved 6-week future forecast to data/processed/future_forecast.csv: {len(df_forecast)} rows")
    
    # Print metrics file for Makefile/README consumption
    with open("data/processed/backtest_results.txt", "w") as f:
        f.write(f"use_baseline: {use_baseline}\n")
        f.write(f"std_residuals: {std_residuals:.4f}\n")
        
    print("Forecasting step completed successfully!")

if __name__ == "__main__":
    train_and_save_model()
