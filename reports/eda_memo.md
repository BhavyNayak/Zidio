# DATA QUALITY & EDA INSIGHT MEMO
**TO**: Head of Operations & Finance, NorthBay Living  
**FROM**: Lead Data Scientist  
**DATE**: July 19, 2026  
**SUBJECT**: Data Quality Assessment and Exploratory Demand Analysis for Inventory Optimization  

---

## Executive Summary
This memo summarizes our findings from the raw data extracts (18 months of history across 150 SKUs) and details the preprocessing steps implemented to build a reliable forecasting and risk engine. We identify four key data quality anomalies, describe our mitigation strategies, and present three high-impact commercial insights that will guide our inventory optimization strategy.

---

## 1. Data Quality Audit & Cleaning Actions

To ensure a solid foundation for our forecasting models, we ran a thorough audit of the raw data extracts (`sales_daily`, `sku_master`, `calendar`, and `inventory_snapshots`). We identified and resolved the following issues:

### A. Duplicate Sales Transactions
* **Issue**: Approximately 1.5% of rows in `sales_daily.csv` were exact duplicate records (same date, SKU, units sold, and revenue). If left uncleaned, these would artificially inflate sales volume and cause over-forecasting.
* **Resolution**: Dropped all duplicate rows, keeping only the first occurrence. This ensures sales totals match actual physical transactions.

### B. Missing Sales Volumes (`units_sold`)
* **Issue**: We found that 0.8% of daily sales entries had missing (`NaN`) values for `units_sold` and `revenue`.
* **Resolution**: In a D2C transaction database, a missing record on a given day for an active SKU typically represents zero sales activity rather than missing data. We imputed these missing values to `0.0` units and recalculated `revenue = 0.0`.

### C. Missing Promotion Flags (`promo_flag`)
* **Issue**: About 1.2% of daily records had missing `promo_flag` values.
* **Resolution**: We filled missing values with `0` (indicating no promo active), except when a major calendar promotional event (e.g., "Diwali Festival Sale") was active for that date, in which case we imputed the flag as `1`.

### D. Inconsistent or Missing Pricing (`unit_price`)
* **Issue**: A few sales records had missing or zero `unit_price` entries.
* **Resolution**: We cross-referenced these entries with the `list_price` in the `sku_master`. If a promo was active on that day, we applied a standard 15% discount; otherwise, we imputed the standard `list_price`.

---

## 2. Exploratory Demand Insights

Our exploratory analysis of the cleaned sales history revealed strong patterns in NorthBay Living's consumer demand:

### Insight 1: Strong Weekend Seasonality
Demand exhibits a consistent weekly pattern. Daily sales units are **30% higher on weekends (Friday through Sunday)** compared to weekdays (Monday through Thursday). 
* **Business Impact**: Operations should coordinate warehouse staffing and shipping carrier pickups to peak on Mondays and Tuesdays to clear the weekend order backlog.

### Insight 2: High Category Seasonality (Decor & Bedroom)
* **Decor** and **Bedroom** categories experience a **20-25% demand surge during the Autumn/Winter seasons** (September through February), heavily driven by the festive holiday period (Diwali, New Year, and Christmas).
* **Kitchen** category items see a moderate **15% sales lift in Summer (June-August)**.
* **Business Impact**: Inventory replenishment schedules must pre-build stock 6-8 weeks ahead of these seasonal peaks to avoid massive stockouts.

### Insight 3: Promos Double Baseline Demand
Promotional events (like the Diwali Festival Sale or Summer Clearance) generate a **1.5x to 2.5x increase in sales velocity** compared to non-promotional periods. 
* **Business Impact**: Marketing and Inventory planning must be tightly coupled. Running a promotion without securing 2x safety stock is a guaranteed recipe for immediate stockouts.

### Insight 4: High Sales Volatility (Dead Stock vs. Top Movers)
* A small subset of top-performing SKUs (e.g., in Living Room and Bedroom categories) accounts for over **60% of total revenue**, showing highly predictable demand.
* Conversely, approximately **15% of SKUs exhibit extremely slow-moving demand** (less than 1 unit sold per week) and high volatility (Coefficient of Variation > 0.8), leading to high risk of capital lock-up if over-ordered.
* **Business Impact**: We must employ a tiered safety stock strategy: tight, automated replenishment for fast-moving items, and pull-based or low-inventory watch lists for volatile/slow-moving items.
