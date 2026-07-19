# Project FORESIGHT — Inventory Optimization Suite

## 1. Problem Statement & Client Asks
**NorthBay Living** is a D2C home & lifestyle brand managing around 150 SKUs. Currently, they plan inventory based on manual spreadsheet inputs and intuition, leading to financial losses from two sources: **stockouts** (which cause lost sales) and **overstocking** (which locks up working capital and leads to forced clearance markdowns). 

To solve this, Project FORESIGHT delivers an automated, reproducible forecasting and decisioning suite. The client requested four core outcomes:
1. A reproducible data pipeline that ingests raw tables and resolves data-quality issues.
2. A transparent demand forecasting model that outperforms a seasonal baseline.
3. A financial risk scoring engine that computes rupee impact (sales at risk and locked capital) and groups SKUs into a 4-quadrant decisioning grid.
4. User-friendly entry points (an interactive Streamlit dashboard for operations planners and a FastAPI microservice for systems integration).

---

## 2. Data Description & Ingestion Sources
All raw data tables are synthetic, generated via `src/generate_data.py` to simulate a real D2C retail setup:
* **`sales_daily.csv`**: Contains `date`, `sku_id`, `units_sold`, `revenue`, `unit_price`, and `promo_flag` over 18 months. Includes weekly patterns, holidays, missing values, duplicates, and new launches.
* **`sku_master.csv`**: Catalogs all 150 SKUs, including their category (`Bedroom`, `Living Room`, `Kitchen`, `Decor`), subcategory, launch date, unit cost, and list price.
* **`calendar.csv`**: Calendar dimensions mapping dates to seasonal names, holidays, and marketing campaigns.
* **`inventory_snapshots.csv`**: Daily snapshots of warehouse stock (`on_hand_units` and `on_order_units`) alongside procurement constraints (`lead_time_days` and `reorder_point`).

---

## 3. Setup and Run Instructions
To run the pipeline and services from a clean clone, execute the following commands in your shell:

### Prerequisites
Make sure Python (>= 3.9) and `pip` are installed.

### Setup Steps
1. Clone the repository and navigate into the folder:
   ```bash
   cd foresight
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the complete pipeline end-to-end (generates raw data, runs data cleaning, trains the ML model, scores the risk, and runs unit tests):
   ```bash
   make all
   ```

### Running Services Locally
* **Launch the Streamlit Planning Dashboard**:
  ```bash
  make dashboard
  ```
  *(Opens local web server at http://localhost:8501)*
* **Launch the FastAPI Scoring Service**:
  ```bash
  make api
  ```
  *(Launches FastAPI at http://127.0.0.1:8000. Interactive Swagger documentation is available at http://127.0.0.1:8000/docs)*

---

## 4. Backtest Results
We evaluated both models using rolling-origin cross-validation (3 time-split folds, each with a 6-week forecasting horizon) to prevent future leakage. The primary metric chosen is **WAPE** (Weighted Absolute Percentage Error) at the daily SKU level:

* **Seasonal-Naive Baseline WAPE**: **80.80%**
* **HistGradientBoostingRegressor Model WAPE**: **74.68%**
* **WAPE Reduction**: **+6.12% relative improvement**

Our Machine Learning model successfully outperformed the seasonal baseline across every test fold and was selected as the active model for future forecasts.

---

## 5. Key Assumptions and Limitations
* **Imputation Assumption**: Missing values for `units_sold` are assumed to represent zero-sale days rather than missing data, which is typical for transactional databases.
* **Promotion Planning**: The forecasting model assumes promotional campaigns are pre-logged in the calendar. Unplanned marketing promotions will lead to under-forecasting.
* **Static Lead Times**: Lead times are assumed to be constant for each SKU. In reality, supplier delays and transit volatility can affect risk horizons.

---

## 6. Deployment Notes
* **FastAPI Service**: Ready for deployment on **Render**, **Railway**, or **AWS App Runner** via `uvicorn service.main:app --host 0.0.0.0 --port $PORT`.
* **Streamlit Dashboard**: Can be deployed directly on **Streamlit Community Cloud** or **Hugging Face Spaces** by linking the GitHub repository.

---

## 7. Repository Structure
```
foresight/
  data/
    raw/                # Synthetic database extracts (calendar, sales, inventory)
    processed/          # Cleaned, engineered datasets, forecasts and risk summaries
  notebooks/
    01_eda.ipynb        # Exploratory analysis notebook
    02_baseline.ipynb   # Seasonal-naive benchmark validation
    03_model.ipynb      # Gradient Boosting model evaluation
  src/
    generate_data.py    # Simulates relational D2C tables and overstock/stockout loops
    pipeline.py         # Handles deduplication, imputation, joins, and lag features
    forecast.py         # Performs cross-validation and recursive forecasts
    risk.py             # Computes financial risk and categorizes into the 4 quadrants
  app/
    dashboard.py        # Streamlit interactive application
  service/
    main.py             # FastAPI REST endpoints and schemas
  reports/
    eda_memo.md         # Data quality and consumer insights memo
    executive_readout.md# Financial summary slide deck for Operations/Finance head
  tests/
    test_pipeline.py    # Pipeline unit tests and data leakage verification
    test_forecast.py    # Forecaster metric and baseline unit tests
    test_risk.py        # Risk equation and quadrant verification tests
  requirements.txt    # Project dependencies
  Makefile            # Command runner configurations
  .gitignore          # Version control ignore list
```
