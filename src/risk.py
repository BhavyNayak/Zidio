import os
import json
import pandas as pd
import numpy as np

def run_risk_scoring():
    print("Starting inventory risk scoring and decisioning grid calculation...")
    
    # Paths
    processed_dir = "data/processed"
    raw_dir = "data/raw"
    
    clean_sales_path = os.path.join(processed_dir, "clean_sales.csv")
    future_forecast_path = os.path.join(processed_dir, "future_forecast.csv")
    sku_master_path = os.path.join(raw_dir, "sku_master.csv")
    
    if not (os.path.exists(clean_sales_path) and os.path.exists(future_forecast_path) and os.path.exists(sku_master_path)):
        raise FileNotFoundError("Required processed sales, forecast, or raw SKU master files are missing!")
        
    df_sales = pd.read_csv(clean_sales_path)
    df_forecast = pd.read_csv(future_forecast_path)
    df_sku = pd.read_csv(sku_master_path)
    
    # 1. Extract the current state of each SKU at the end of history (2026-06-30)
    # We find the last day of historical sales records to get the snapshot of inventory
    history_end_date = "2026-06-30"
    df_current = df_sales[df_sales["date"] == history_end_date].copy()
    
    if len(df_current) == 0:
        # Fallback: take the absolute last day available in clean_sales
        last_date = df_sales["date"].max()
        df_current = df_sales[df_sales["date"] == last_date].copy()
        print(f"Warning: history_end_date {history_end_date} not found. Using last date {last_date}")
        
    print(f"Extracted current inventory state for {len(df_current)} SKUs.")
    
    # 2. Compute forecast summaries for each SKU
    # Lead time demand (D_LT): Sum of forecast over lead_time_days
    # Forward demand (D_FW): Sum of forecast over entire horizon (42 days)
    
    # Ensure forecast dates are sorted
    df_forecast = df_forecast.sort_values(by=["sku_id", "date"]).reset_index(drop=True)
    
    risk_records = []
    
    for idx, row in df_current.iterrows():
        sku_id = row["sku_id"]
        on_hand = row["on_hand_units"]
        on_order = row["on_order_units"]
        lead_time = int(row["lead_time_days"])
        reorder_point = row["reorder_point"]
        list_price = row["list_price"]
        unit_cost = row["unit_cost"]
        category = row["category"]
        subcategory = row["subcategory"]
        
        # Get forecast for this SKU
        sku_fc = df_forecast[df_forecast["sku_id"] == sku_id].reset_index(drop=True)
        
        if len(sku_fc) == 0:
            # Fallback if no forecast (e.g. brand new SKU that wasn't in forecast)
            lead_time_demand = 0.0
            forward_demand = 0.0
            max_upper_bound = 0.0
        else:
            # Lead time demand (sum of first 'lead_time' days of forecast)
            # Clip lead_time to available forecast rows (max 42)
            lt_days = min(lead_time, len(sku_fc))
            lead_time_demand = sku_fc.loc[:lt_days-1, "forecast_units"].sum()
            
            # Forward demand (sum of all 42 days of forecast)
            forward_demand = sku_fc["forecast_units"].sum()
            
            # Forecast uncertainty (width of 80% interval)
            # Width = upper_80 - lower_80. We take the mean width
            fc_widths = sku_fc["forecast_upper_80"] - sku_fc["forecast_lower_80"]
            mean_width = fc_widths.mean()
            mean_forecast = sku_fc["forecast_units"].mean()
            uncertainty_ratio = mean_width / mean_forecast if mean_forecast > 0 else 0.0
            
        # Inventory Position
        inventory_position = on_hand + on_order
        
        # A. Stockout Risk
        # If inventory position is less than lead time demand, we will run out before new stock arrives.
        stockout_units = max(0.0, lead_time_demand - inventory_position)
        stockout_risk_val = stockout_units * list_price # revenue at risk
        
        # B. Overstock Risk
        # If on_hand inventory alone is greater than our 6-week demand forecast.
        # We use a threshold of 1.5x D_FW to denote overstock, and calculate excess units over 1.0x D_FW.
        is_overstocked = on_hand > (forward_demand * 1.5)
        overstock_units = max(0.0, on_hand - forward_demand) if is_overstocked else 0.0
        overstock_locked_capital = overstock_units * unit_cost # cost value of capital locked
        
        # C. Classification (Decisioning Grid)
        # 1. Reorder Now: IP < Reorder Point OR stockout units > 0
        # 2. Markdown-Clear: On Hand > 2.0 * Forward Demand (clear overstock)
        # 3. Watch-Volatile: Demand CV is high or uncertainty ratio is high, but not in critical stockout/overstock
        # 4. Healthy: Satisfies demand, no critical warning
        
        # Calculate historical coefficient of variation (CV) for demand
        sku_sales_hist = df_sales[df_sales["sku_id"] == sku_id]["units_sold"]
        mean_sales = sku_sales_hist.mean()
        std_sales = sku_sales_hist.std()
        sales_cv = std_sales / mean_sales if mean_sales > 0 else 0.0
        
        if inventory_position < reorder_point or stockout_units > 0:
            quadrant = "Reorder Now"
            action = f"Reorder standard batch immediately. Stockout risk is {stockout_units:.0f} units (INR {stockout_risk_val:,.0f} sales at risk)."
        elif on_hand > 2.0 * forward_demand and on_hand > 10: # threshold of 10 units to ignore tiny quantities
            quadrant = "Markdown-Clear"
            action = f"Promote and Markdown by 20-30%. Overstock of {overstock_units:.0f} units (INR {overstock_locked_capital:,.0f} capital locked)."
        elif sales_cv > 0.8 or (len(sku_fc) > 0 and uncertainty_ratio > 1.2):
            quadrant = "Watch-Volatile"
            action = f"High demand volatility (CV={sales_cv:.2f}). Monitor inventory weekly. Maintain safety stock."
        else:
            quadrant = "Healthy"
            action = "No action required. Stock levels are aligned with forecasted demand."
            
        risk_records.append({
            "sku_id": sku_id,
            "category": category,
            "subcategory": subcategory,
            "on_hand_units": on_hand,
            "on_order_units": on_order,
            "inventory_position": inventory_position,
            "lead_time_days": lead_time,
            "reorder_point": reorder_point,
            "lead_time_demand": lead_time_demand,
            "forward_demand": forward_demand,
            "stockout_units": stockout_units,
            "stockout_risk_val": stockout_risk_val,
            "overstock_units": overstock_units,
            "overstock_locked_capital": overstock_locked_capital,
            "quadrant": quadrant,
            "recommended_action": action,
            "unit_price": list_price,
            "unit_cost": unit_cost
        })
        
    df_risk = pd.DataFrame(risk_records)
    
    # Save the risk decisioning dataset
    risk_output_path = os.path.join(processed_dir, "risk_decisions.csv")
    df_risk.to_csv(risk_output_path, index=False)
    print(f"Saved risk decisions to: {risk_output_path}")
    
    # 3. Compute Summary Statistics for Executive Output
    total_stockout_val = df_risk["stockout_risk_val"].sum()
    total_overstock_val = df_risk["overstock_locked_capital"].sum()
    
    quadrant_counts = df_risk["quadrant"].value_counts().to_dict()
    for q in ["Reorder Now", "Markdown-Clear", "Watch-Volatile", "Healthy"]:
        if q not in quadrant_counts:
            quadrant_counts[q] = 0
            
    summary_stats = {
        "analysis_date": history_end_date,
        "total_skus": len(df_risk),
        "total_sales_at_risk_inr": float(total_stockout_val),
        "total_capital_locked_inr": float(total_overstock_val),
        "quadrant_counts": quadrant_counts,
        "category_summary": df_risk.groupby("category")[["stockout_risk_val", "overstock_locked_capital"]].sum().to_dict(orient="index")
    }
    
    summary_path = os.path.join(processed_dir, "risk_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"Saved risk summary stats to: {summary_path}")
    print("Risk scoring complete!")

if __name__ == "__main__":
    run_risk_scoring()
