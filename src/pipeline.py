import os
import pandas as pd
import numpy as np

def run_pipeline():
    print("Starting data ingestion and preprocessing pipeline...")
    
    # Paths
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Ingest datasets
    print("Ingesting raw data...")
    df_sales = pd.read_csv(os.path.join(raw_dir, "sales_daily.csv"))
    df_sku = pd.read_csv(os.path.join(raw_dir, "sku_master.csv"))
    df_calendar = pd.read_csv(os.path.join(raw_dir, "calendar.csv"))
    df_inventory = pd.read_csv(os.path.join(raw_dir, "inventory_snapshots.csv"))
    
    # 2. Cleaning sales_daily
    print(f"Initial sales records count: {len(df_sales)}")
    
    # A. Deduplication
    # Decision: Remove duplicate records for the same (date, sku_id) key.
    # To handle cases where duplicates have inconsistent values (e.g. one copy corrupted to NaN),
    # we first sort so that non-null rows are preferred, and then drop duplicates on [date, sku_id].
    initial_count = len(df_sales)
    df_sales = df_sales.sort_values(by=["date", "sku_id", "units_sold"], na_position="last")
    df_sales.drop_duplicates(subset=["date", "sku_id"], keep="first", inplace=True)
    duplicates_count = initial_count - len(df_sales)
    if duplicates_count > 0:
        print(f"Removed {duplicates_count} duplicate sales records based on date-SKU keys.")
        
    # B. Handle missing values
    # Decision: For missing promo_flag, assume no promotion (value 0) unless it matches a promo period.
    missing_promo = df_sales["promo_flag"].isnull().sum()
    if missing_promo > 0:
        df_sales["promo_flag"] = df_sales["promo_flag"].fillna(0).astype(int)
        print(f"Imputed {missing_promo} missing promo_flags with 0.")
        
    # Decision: For missing units_sold, impute to 0 because missing rows in D2C standard transaction databases
    # typically represent days with zero sales activity for that SKU.
    missing_units = df_sales["units_sold"].isnull().sum()
    if missing_units > 0:
        df_sales["units_sold"] = df_sales["units_sold"].fillna(0.0)
        print(f"Imputed {missing_units} missing units_sold with 0.0.")
        
    # Decision: For missing unit_price or revenue, look up list_price from sku_master and fill.
    # First, let's merge with sku_master to get list_price and unit_cost.
    df_sales = pd.merge(df_sales, df_sku[["sku_id", "list_price", "unit_cost", "category", "subcategory"]], on="sku_id", how="left")
    
    # Fill unit_price with list_price if unit_price is null or <= 0
    # Note: If promo_flag was 1, we could apply a 15% discount. Let's see.
    price_null_mask = df_sales["unit_price"].isnull() | (df_sales["unit_price"] <= 0)
    price_null_count = price_null_mask.sum()
    if price_null_count > 0:
        # If promo_flag is 1, apply a 15% discount, otherwise list_price
        discounted_price = df_sales["list_price"] * 0.85
        df_sales.loc[price_null_mask, "unit_price"] = np.where(
            df_sales.loc[price_null_mask, "promo_flag"] == 1,
            np.round(discounted_price.loc[price_null_mask], -1),
            df_sales.loc[price_null_mask, "list_price"]
        )
        print(f"Imputed {price_null_count} missing unit_prices using sku_master pricing rules.")
        
    # Re-calculate revenue where revenue is null or mismatch
    df_sales["revenue"] = df_sales["units_sold"] * df_sales["unit_price"]
    
    # 3. Clean calendar & inventory tables
    df_calendar["date"] = df_calendar["date"].astype(str)
    df_inventory["date"] = df_inventory["date"].astype(str)
    df_sales["date"] = df_sales["date"].astype(str)
    
    # Deduplicate inventory just in case
    df_inventory.drop_duplicates(subset=["date", "sku_id"], inplace=True)
    
    # 4. Join datasets
    print("Joining datasets...")
    # Join sales with calendar
    df_merged = pd.merge(df_sales, df_calendar, on="date", how="left")
    
    # Join with inventory snapshots
    df_merged = pd.merge(df_merged, df_inventory, on=["date", "sku_id"], how="left")
    
    # Fill missing inventory values if any (e.g. for brand new SKUs before launch)
    # Decisions:
    # - If on_hand is null, fill with 0
    # - If on_order is null, fill with 0
    # - If lead_time_days is null, fill with the SKU's median or default 10 days
    # - If reorder_point is null, fill with 0
    df_merged["on_hand_units"] = df_merged["on_hand_units"].fillna(0.0)
    df_merged["on_order_units"] = df_merged["on_order_units"].fillna(0.0)
    df_merged["lead_time_days"] = df_merged["lead_time_days"].fillna(10.0)
    df_merged["reorder_point"] = df_merged["reorder_point"].fillna(0.0)
    
    # Sort dataset by sku_id and date for correct feature engineering
    df_merged.sort_values(by=["sku_id", "date"], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    
    # 5. Feature Engineering
    print("Engineering features...")
    
    # Ensure no data leakage:
    # All historical sales features (lags, rolling averages) must be shifted by 1 day minimum.
    # For example, units_sold on day t must never be used to calculate features for day t.
    
    # We group by sku_id to compute shifted values
    grouped = df_merged.groupby("sku_id")
    
    # A. Lag Features
    df_merged["units_sold_lag_1"] = grouped["units_sold"].shift(1)
    df_merged["units_sold_lag_7"] = grouped["units_sold"].shift(7)
    df_merged["units_sold_lag_28"] = grouped["units_sold"].shift(28)
    
    # B. Rolling Statistics (on shifted units_sold to avoid data leakage)
    # The shift(1) shifts the entire series, so rolling windows on it do not contain today's units_sold.
    df_merged["units_sold_roll_mean_7"] = grouped["units_sold"].transform(lambda x: x.shift(1).rolling(7).mean())
    df_merged["units_sold_roll_std_7"] = grouped["units_sold"].transform(lambda x: x.shift(1).rolling(7).std())
    df_merged["units_sold_roll_mean_28"] = grouped["units_sold"].transform(lambda x: x.shift(1).rolling(28).mean())
    df_merged["units_sold_roll_std_28"] = grouped["units_sold"].transform(lambda x: x.shift(1).rolling(28).std())
    
    # Fill NaN values created by shift operations with 0.0 or appropriate defaults
    df_merged["units_sold_lag_1"] = df_merged["units_sold_lag_1"].fillna(0.0)
    df_merged["units_sold_lag_7"] = df_merged["units_sold_lag_7"].fillna(0.0)
    df_merged["units_sold_lag_28"] = df_merged["units_sold_lag_28"].fillna(0.0)
    
    df_merged["units_sold_roll_mean_7"] = df_merged["units_sold_roll_mean_7"].fillna(0.0)
    df_merged["units_sold_roll_std_7"] = df_merged["units_sold_roll_std_7"].fillna(0.0)
    df_merged["units_sold_roll_mean_28"] = df_merged["units_sold_roll_mean_28"].fillna(0.0)
    df_merged["units_sold_roll_std_28"] = df_merged["units_sold_roll_std_28"].fillna(0.0)
    
    # C. Date and season categorical encoding helper
    from datetime import datetime
    parsed_dates = [datetime.strptime(x, "%Y-%m-%d") for x in df_merged["date"].astype(str)]
    df_merged["day_of_week"] = [d.weekday() for d in parsed_dates]
    df_merged["day_of_month"] = [d.day for d in parsed_dates]
    df_merged["quarter"] = [(d.month - 1) // 3 + 1 for d in parsed_dates]
    
    # 6. Save cleaned data
    output_path = os.path.join(processed_dir, "clean_sales.csv")
    df_merged.to_csv(output_path, index=False)
    print(f"Data pipeline complete! Processed dataset shape: {df_merged.shape}")
    print(f"Saved cleaned dataset to: {output_path}")

if __name__ == "__main__":
    run_pipeline()
