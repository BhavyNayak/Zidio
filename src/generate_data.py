import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(seed=42):
    np.random.seed(seed)
    
    # 1. Define active period
    # 18 months of history: 2025-01-01 to 2026-06-30 (546 days)
    # Plus 6 weeks of future calendar for forecasting: up to 2026-08-15
    start_date = datetime(2025, 1, 1)
    end_history_date = datetime(2026, 6, 30)
    end_forecast_date = datetime(2026, 8, 15)
    
    history_days = (end_history_date - start_date).days + 1
    total_days = (end_forecast_date - start_date).days + 1
    
    date_list = [start_date + timedelta(days=x) for x in range(total_days)]
    
    # Create raw/ data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    print(f"Generating data from {start_date.date()} to {end_forecast_date.date()}")
    print(f"History period ends at {end_history_date.date()}")
    
    # 2. Generate Calendar Master
    calendar_records = []
    seasons = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    
    # Indian D2C Holiday List (approximate dates for 2025/2026)
    holidays = {
        (1, 26): "Republic Day",
        (3, 14): "Holi",          # 2025 Holi approx
        (8, 15): "Independence Day",
        (10, 20): "Diwali Period", # Diwali is late Oct/Nov
        (10, 21): "Diwali Period",
        (10, 22): "Diwali Period",
        (12, 25): "Christmas"
    }
    
    for d in date_list:
        is_holiday = 1 if (d.month, d.day) in holidays else 0
        promo_event = "None"
        if (d.month == 10 and 15 <= d.day <= 22):
            promo_event = "Diwali Festival Sale"
        elif (d.month == 6 and 10 <= d.day <= 15):
            promo_event = "Summer Clearance"
        elif (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 2):
            promo_event = "New Year Bash"
            
        calendar_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "week": d.isocalendar()[1],
            "month": d.month,
            "season": seasons[d.month],
            "is_holiday": is_holiday,
            "promo_event": promo_event
        })
        
    df_calendar = pd.DataFrame(calendar_records)
    df_calendar.to_csv("data/raw/calendar.csv", index=False)
    print(f"Saved calendar.csv: {len(df_calendar)} rows")
    
    # 3. Generate SKU Master
    # Total ~150 SKUs
    num_skus = 150
    categories = {
        "Bedroom": ["Beds", "Nightstands", "Dressers", "Mattresses"],
        "Living Room": ["Sofas", "Accent Chairs", "Coffee Tables", "TV Units"],
        "Kitchen": ["Dining Tables", "Dining Chairs", "Cookware", "Kitchen Cabinets"],
        "Decor": ["Lighting", "Rugs", "Wall Art", "Vases"]
    }
    
    cat_probs = [0.25, 0.35, 0.20, 0.20] # Living Room is largest
    category_list = np.random.choice(list(categories.keys()), size=num_skus, p=cat_probs)
    
    sku_records = []
    # 15 SKUs are brand new (launched in the last 60 days of history, i.e., after 2026-05-01)
    new_sku_count = 15
    new_sku_start_idx = num_skus - new_sku_count
    
    for i in range(num_skus):
        sku_id = f"SKU{i+1:03d}"
        category = category_list[i]
        subcategory = np.random.choice(categories[category])
        
        # Launch date
        if i >= new_sku_start_idx:
            # Launched between 2026-05-01 and 2026-06-15
            days_offset = np.random.randint(0, 45)
            launch_d = datetime(2026, 5, 1) + timedelta(days=days_offset)
        else:
            # Launched before 2025-01-01
            days_offset = np.random.randint(30, 365)
            launch_d = start_date - timedelta(days=days_offset)
            
        # Pricing and Costing
        if category == "Bedroom":
            unit_cost = np.random.uniform(5000, 25000)
        elif category == "Living Room":
            unit_cost = np.random.uniform(4000, 20000)
        elif category == "Kitchen":
            unit_cost = np.random.uniform(1500, 12000)
        else: # Decor
            unit_cost = np.random.uniform(300, 3000)
            
        # Markup of 1.5x to 2.5x
        markup = np.random.uniform(1.5, 2.5)
        list_price = np.round(unit_cost * markup, -1) # round to nearest 10
        unit_cost = np.round(unit_cost, -1)
        
        sku_records.append({
            "sku_id": sku_id,
            "category": category,
            "subcategory": subcategory,
            "launch_date": launch_d.strftime("%Y-%m-%d"),
            "unit_cost": unit_cost,
            "list_price": list_price
        })
        
    df_sku = pd.DataFrame(sku_records)
    df_sku.to_csv("data/raw/sku_master.csv", index=False)
    print(f"Saved sku_master.csv: {len(df_sku)} rows")
    
    # 4. Generate Sales Daily & Inventory Snapshots via Simulation
    # We will simulate day-by-day inventory and demand to make them consistent
    # For day t, we compute demand. But actual sales = min(demand, on_hand).
    # Then we decrement on_hand by sales.
    # If on_hand + on_order < reorder_point, we place an order (on_order increases).
    # The order arrives after lead_time_days.
    
    history_dates = [start_date + timedelta(days=x) for x in range(history_days)]
    
    # SKU properties for simulation
    sku_sim_props = {}
    for idx, row in df_sku.iterrows():
        sku_id = row["sku_id"]
        # Base daily demand lambda (0.1 to 4.0 units per day)
        base_lambda = np.random.exponential(scale=0.8) + 0.1
        # Lead time days: 5 to 20 days
        lead_time = np.random.randint(5, 21)
        # Reorder point: lead_time * base_lambda * 1.5 safety factor
        reorder_point = int(np.ceil(lead_time * base_lambda * 1.5))
        # Reorder quantity: 30 days of demand, min 10
        reorder_qty = max(10, int(np.ceil(base_lambda * 30)))
        
        # Initial inventory state on 2025-01-01 (or launch date)
        # For pre-existing SKUs, let's start with a random inventory between reorder_point and 3 * reorder_point
        on_hand = np.random.randint(reorder_point, reorder_point * 3 + 1)
        
        sku_sim_props[sku_id] = {
            "base_lambda": base_lambda,
            "lead_time": lead_time,
            "reorder_point": reorder_point,
            "reorder_qty": reorder_qty,
            "on_hand": on_hand,
            "on_order": 0,
            "pending_deliveries": [] # list of dict: {"arrival_date": dt, "qty": q}
        }
        
    # Convert DataFrames to dictionaries for fast O(1) lookups in the loop
    calendar_dict = df_calendar.set_index("date").to_dict(orient="index")
    sku_dict = df_sku.set_index("sku_id").to_dict(orient="index")
    for s_id, info in sku_dict.items():
        info["launch_date_dt"] = datetime.strptime(info["launch_date"], "%Y-%m-%d")
        
    sales_records = []
    inventory_records = []
    
    for d in history_dates:
        d_str = d.strftime("%Y-%m-%d")
        day_of_week = d.weekday() # 0 is Monday, 6 is Sunday
        
        # Get promo event status for the day from dict lookup
        day_cal = calendar_dict[d_str]
        promo_event = day_cal["promo_event"]
        is_holiday = day_cal["is_holiday"]
        season = day_cal["season"]
        
        for sku_id, props in sku_sim_props.items():
            sku_info = sku_dict[sku_id]
            launch_d = sku_info["launch_date_dt"]
            
            if d < launch_d:
                # SKU not launched yet
                continue
                
            # If SKU is newly launched, initialize its inventory on launch date
            if d == launch_d:
                props["on_hand"] = props["reorder_qty"] * 2
                
            # 1. Process incoming deliveries first thing in the morning
            arrived_qty = 0
            remaining_deliveries = []
            for deliv in props["pending_deliveries"]:
                if deliv["arrival_date"].date() <= d.date():
                    arrived_qty += deliv["qty"]
                else:
                    remaining_deliveries.append(deliv)
            props["pending_deliveries"] = remaining_deliveries
            props["on_hand"] += arrived_qty
            props["on_order"] = max(0, props["on_order"] - arrived_qty)
            
            # 2. Determine demand lambda
            # Baseline
            lam = props["base_lambda"]
            
            # Day of week factor: 30% higher on Friday (4), Saturday (5), Sunday (6)
            if day_of_week in [4, 5, 6]:
                lam *= 1.3
            else:
                lam *= 0.85
                
            # Seasonality factor
            sku_cat = sku_info["category"]
            if season == "Winter" and sku_cat in ["Bedroom", "Decor"]:
                lam *= 1.2
            elif season == "Summer" and sku_cat in ["Kitchen"]:
                lam *= 1.15
                
            # Promo factors
            promo_flag = 0
            if promo_event != "None":
                promo_flag = 1
                # Stronger effect on Decor/Living Room during Diwali
                if promo_event == "Diwali Festival Sale":
                    lam *= 2.5 if sku_cat in ["Decor", "Living Room"] else 1.8
                elif promo_event == "Summer Clearance":
                    lam *= 2.0
                else:
                    lam *= 1.5
            else:
                # Random flash promo for specific SKUs (5% chance)
                if np.random.rand() < 0.05:
                    promo_flag = 1
                    lam *= 1.7
                    
            # Holiday factor
            if is_holiday:
                lam *= 1.25
                
            # 3. Generate demand
            demand = np.random.poisson(lam)
            
            # 4. Constrain sales by on-hand inventory
            on_hand_before = props["on_hand"]
            units_sold = min(demand, on_hand_before)
            props["on_hand"] -= units_sold
            
            # 5. Price & Revenue
            list_price = sku_info["list_price"]
            # Apply promo discount if applicable
            unit_price = list_price
            if promo_flag == 1:
                discount = np.random.choice([0.1, 0.15, 0.2]) # 10%, 15%, 20% off
                unit_price = np.round(list_price * (1.0 - discount), -1)
                
            revenue = units_sold * unit_price
            
            # Save daily sales record (only if sku was launched)
            sales_records.append({
                "date": d_str,
                "sku_id": sku_id,
                "units_sold": units_sold,
                "revenue": revenue,
                "unit_price": unit_price,
                "promo_flag": promo_flag
            })
            
            # 6. Reorder decision at the end of the day
            inventory_position = props["on_hand"] + props["on_order"]
            if inventory_position < props["reorder_point"] and props["on_order"] == 0:
                # Order placed!
                order_qty = props["reorder_qty"]
                arrival_date = d + timedelta(days=props["lead_time"])
                props["pending_deliveries"].append({
                    "arrival_date": arrival_date,
                    "qty": order_qty
                })
                props["on_order"] = order_qty
                
            # Save inventory snapshot (end of day)
            # Inject some overstock on the last day to simulate gut-feel over-ordering of slow movers
            on_hand_to_record = props["on_hand"]
            if d == history_dates[-1] and props["base_lambda"] < 0.45 and np.random.rand() < 0.4:
                on_hand_to_record = int(props["reorder_qty"] * 6)
                
            inventory_records.append({
                "date": d_str,
                "sku_id": sku_id,
                "on_hand_units": on_hand_to_record,
                "on_order_units": props["on_order"],
                "lead_time_days": props["lead_time"],
                "reorder_point": props["reorder_point"]
            })
            
    df_sales = pd.DataFrame(sales_records)
    df_inventory = pd.DataFrame(inventory_records)
    
    # --- Inject deliberately imperfect data to mimic raw extracts ---
    print("Injecting data imperfections (missing values, duplicates, and labels)...")
    
    # 1. Duplicates: Duplicate a random 1.5% of rows in sales
    dup_indices = np.random.choice(df_sales.index, size=int(len(df_sales) * 0.015), replace=False)
    df_dups = df_sales.loc[dup_indices].copy()
    # Modify date strings slightly or keep identical. Let's keep them identical to represent standard duplication.
    df_sales = pd.concat([df_sales, df_dups], ignore_index=True)
    
    # 2. Missing values:
    # Set 'units_sold' to NaN for 0.8% of rows in sales
    missing_sales_idx = np.random.choice(df_sales.index, size=int(len(df_sales) * 0.008), replace=False)
    df_sales.loc[missing_sales_idx, "units_sold"] = np.nan
    # When units_sold is missing, let's also make revenue NaN
    df_sales.loc[missing_sales_idx, "revenue"] = np.nan
    
    # Set 'promo_flag' to NaN for 1.2% of rows in sales
    missing_promo_idx = np.random.choice(df_sales.index, size=int(len(df_sales) * 0.012), replace=False)
    df_sales.loc[missing_promo_idx, "promo_flag"] = np.nan
    
    # 3. Save raw tables
    df_sales.to_csv("data/raw/sales_daily.csv", index=False)
    df_inventory.to_csv("data/raw/inventory_snapshots.csv", index=False)
    
    print(f"Saved sales_daily.csv: {len(df_sales)} rows (with injected anomalies)")
    print(f"Saved inventory_snapshots.csv: {len(df_inventory)} rows")
    print("Synthetic data generation complete!")

if __name__ == "__main__":
    generate_synthetic_data()
