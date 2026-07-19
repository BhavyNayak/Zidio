# PROJECT FORESIGHT — EXECUTIVE READOUT
**Client**: NorthBay Living  
**Objective**: Inventory Cost Optimization & Demand Forecasting  
**Audience**: Head of Operations & Finance  

---

## Slide 1: Executive Summary (The Rupee Impact)

NorthBay Living currently plans inventory on gut-feel, resulting in parallel losses: stockouts (lost sales) and overstock (locked capital). 

### The Financial Scorecard (Current State)
* **₹4.49 Lakhs / ₹4.49 Million (INR 4,494,215) in Projected Sales at Risk**
  * Revenue that will be lost over SKU lead times because current inventory position (On Hand + On Order) cannot cover forecasted demand.
* **₹3.48 Lakhs / ₹3.48 Million (INR 3,482,435) in Locked Capital**
  * Capital tied up in excess inventory (On Hand stock exceeding the entire next 6 weeks of forecasted demand).

> [!IMPORTANT]
> **Total Financial Opportunity: ₹7.97 Million (₹79.7 Lakhs)**  
> By aligning inventory with our predictive demand pipeline, we can unlock ₹3.48M of working capital and save up to ₹4.49M in lost revenue.

---

## Slide 2: The Decisioning Grid (SKU Segmentation)

We analyzed 150 SKUs and classified them into four distinct inventory action quadrants:

| Quadrant | SKU Count | Financial Metric | Action Required |
| :--- | :---: | :---: | :--- |
| **Reorder Now** | **41** | ₹4.49M Sales at Risk | Place replenishment orders immediately |
| **Markdown-Clear** | **8** | ₹3.48M Locked Capital | Run 20-30% promotional discount campaigns |
| **Watch-Volatile** | **82** | High Volatility (CV > 0.8) | Monitor weekly; maintain buffer safety stock |
| **Healthy** | **19** | Optimal Levels | No immediate action; monitor |

* **Watch-Volatile** represents the largest group (82 SKUs). These are slow-moving or highly erratic items where demand fluctuates widely, necessitating conservative replenishment.

---

## Slide 3: Action Plan — Replenishment (Reorder Now)

We identified 41 SKUs that require immediate reordering because their stock levels will drop to zero before the replenishment order arrives.

### Category Breakdown of Sales at Risk:
* **Living Room**: ₹1.93 Million at risk (43% of total risk)
* **Kitchen**: ₹1.24 Million at risk (28% of total risk)
* **Bedroom**: ₹1.18 Million at risk (26% of total risk)
* **Decor**: ₹0.15 Million at risk (3% of total risk)

### Key Recommendations:
1. **Prioritize Living Room and Kitchen**: Replenish these categories first as they represent **71% of total lost sales risk**.
2. **Automate Lead-Time Ordering**: Integrate the FastAPI `POST /score` endpoint into the procurement system to automatically trigger purchase orders when inventory position falls below the Reorder Point.

---

## Slide 4: Action Plan — Capital Release (Markdown-Clear)

We identified 8 SKUs with massive overstock relative to forecasted demand. 

### Category Breakdown of Locked Capital:
* **Bedroom**: ₹3.20 Million locked (**91.8% of total locked capital**)
* **Decor**: ₹0.28 Million locked (8.2% of total locked capital)
* **Kitchen & Living Room**: ₹0.00 (Healthy / optimal levels)

### Key Recommendations:
1. **Target Bedroom Overstock**: Bedroom furniture accounts for ₹3.20M of the trapped capital.
2. **Execute Tiered Markdowns**:
   * Implement a **25% discount campaign** specifically targeting these 8 overstocked SKUs.
   * Cross-promote these items on the website header and D2C email newsletters to accelerate velocity.
3. **Reinvest Released Capital**: Funnel the freed ₹3.20M into replenishing the high-demand Living Room and Kitchen SKUs.

---

## Slide 5: The Forecasting Engine (Accuracy & Performance)

We built and backtested a Gradient Boosted Tree model against a standard Seasonal-Naive baseline.

### Backtest Results (Rolling-Origin Cross-Validation):
* **Seasonal-Naive Baseline WAPE**: **80.80%**
* **Machine Learning Model WAPE**: **74.68%**
* **Improvement**: **+6.12% reduction in daily SKU-level error**

### Why This Matters:
* Daily SKU-level forecasting is notoriously difficult in D2C due to high sparsity (many days with zero sales). 
* Our model successfully leverages weekly seasonality, calendar events, and promotional flags to beat the baseline on every single historical fold, leading to significantly tighter safety stock buffers.

---

## Slide 6: Model Limitations & Safety Guardrails

No forecasting model is perfect. To protect NorthBay Living's operations, we have built-in safety guardrails:

### Key Limitations:
1. **New SKU Warm-start**: SKUs launched within the last 60 days have short histories, limiting the model's ability to learn their true seasonality. The system uses a category-average fallback for these.
2. **Unplanned Promotions**: If marketing runs a promo without setting the `promo_flag` in the pipeline, the model will under-forecast.

### Safety Guardrails:
* **80% Uncertainty Band**: We do not just output a single "point" forecast. The planning dashboard displays an **80% uncertainty interval** (e.g., Min and Max expected sales). 
* Procurement should use the **80% Upper Bound** for fast-moving critical SKUs to ensure absolute protection against stockouts.

---

## Slide 7: Next Steps & Ingestion Routine

To transition from spreadsheets to an automated workflow, we recommend the following deployment checklist:

1. **Daily Data Export**: Set up an automated cron job at 12:00 AM to dump daily tables (`sales_daily`, `sku_master`, `calendar`, `inventory_snapshots`) into `data/raw/`.
2. **Automated Pipeline Execution**: Run `make all` every morning to ingest raw data, re-run predictions, and recalculate risk levels.
3. **Integrate Streamlit Dashboard**: Operations planners should open the Streamlit dashboard every Monday morning to review the prioritized Reorder and Markdown tables.
4. **API Integration**: Integrate the FastAPI microservice into the order management system to fetch real-time forecasts.
