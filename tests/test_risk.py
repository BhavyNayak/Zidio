import os
import json
import pandas as pd
import numpy as np

def test_risk_quadrant_decisions():
    # Load the risk decisions output
    decisions_path = "data/processed/risk_decisions.csv"
    summary_path = "data/processed/risk_summary.json"
    
    assert os.path.exists(decisions_path), "risk_decisions.csv is missing!"
    assert os.path.exists(summary_path), "risk_summary.json is missing!"
    
    df = pd.read_csv(decisions_path)
    
    # 1. Check schema
    expected_cols = [
        "sku_id", "category", "on_hand_units", "on_order_units", "inventory_position",
        "lead_time_days", "reorder_point", "lead_time_demand", "forward_demand",
        "stockout_units", "stockout_risk_val", "overstock_units", "overstock_locked_capital",
        "quadrant", "recommended_action", "unit_price", "unit_cost"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col} is missing from risk decisions!"
        
    # 2. Verify quadrant assignment logic
    for idx, row in df.iterrows():
        quad = row["quadrant"]
        ip = row["inventory_position"]
        rp = row["reorder_point"]
        oh = row["on_hand_units"]
        fd = row["forward_demand"]
        su = row["stockout_units"]
        ou = row["overstock_units"]
        
        # Reorder Now verification
        if quad == "Reorder Now":
            assert ip < rp or su > 0, f"SKU {row['sku_id']} is classified as Reorder Now, but has IP={ip} >= RP={rp} and Stockout={su}."
            
        # Markdown Clear verification
        elif quad == "Markdown-Clear":
            assert oh > 2.0 * fd and oh > 10, f"SKU {row['sku_id']} is classified as Markdown-Clear, but has OH={oh} <= 2*FD={2*fd} or OH <= 10."
            
    # 3. Check JSON summary consistency
    with open(summary_path, "r") as f:
        summary = json.load(f)
        
    assert "total_skus" in summary
    assert "total_sales_at_risk_inr" in summary
    assert "total_capital_locked_inr" in summary
    assert "quadrant_counts" in summary
    
    # Sum of stockout risks in csv should match JSON
    total_sales_risk_csv = df["stockout_risk_val"].sum()
    assert np.isclose(total_sales_risk_csv, summary["total_sales_at_risk_inr"]), "Sales at risk sum mismatch between CSV and JSON!"
    
    # Sum of overstocks in csv should match JSON
    total_capital_locked_csv = df["overstock_locked_capital"].sum()
    assert np.isclose(total_capital_locked_csv, summary["total_capital_locked_inr"]), "Capital locked sum mismatch between CSV and JSON!"
    
    # Quadrant counts consistency
    counts = df["quadrant"].value_counts().to_dict()
    for q in ["Reorder Now", "Markdown-Clear", "Watch-Volatile", "Healthy"]:
        csv_count = counts.get(q, 0)
        json_count = summary["quadrant_counts"].get(q, -1)
        assert csv_count == json_count, f"Quadrant count mismatch for {q}: CSV has {csv_count}, JSON has {json_count}"
