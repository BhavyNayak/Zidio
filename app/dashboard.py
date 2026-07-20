import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# Add parent directory to sys.path to resolve 'src' import on Streamlit Cloud
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure page settings
st.set_page_config(
    page_title="FORESIGHT — Inventory Optimization Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for metrics and layout
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 5px solid #6366f1;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        margin-bottom: 20px;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 25px rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 5px solid #6366f1;
    }
    .kpi-card.stockout {
        border-left: 5px solid #ef4444;
    }
    .kpi-card.stockout:hover {
        box-shadow: 0 12px 25px rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 5px solid #ef4444;
    }
    .kpi-card.overstock {
        border-left: 5px solid #f59e0b;
    }
    .kpi-card.overstock:hover {
        box-shadow: 0 12px 25px rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 5px solid #f59e0b;
    }
    .kpi-card.healthy {
        border-left: 5px solid #10b981;
    }
    .kpi-card.healthy:hover {
        box-shadow: 0 12px 25px rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 5px solid #10b981;
    }
    .kpi-title {
        font-size: 13px;
        color: #9ca3af;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        color: #ffffff;
        font-weight: 700;
    }
    .kpi-subtitle {
        font-size: 11px;
        color: #6b7280;
        margin-top: 6px;
    }
</style>


# Helper function to check and auto-run pipeline if files are missing
def ensure_data_exists():
    processed_dir = "data/processed"
    required_files = ["risk_decisions.csv", "future_forecast.csv", "clean_sales.csv"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(processed_dir, f))]
    
    if missing:
        # Import pipeline functions inline to prevent circular imports on startup
        from src.generate_data import generate_synthetic_data
        from src.pipeline import run_pipeline
        from src.forecast import train_and_save_model
        from src.risk import run_risk_scoring
        
        generate_synthetic_data()
        run_pipeline()
        train_and_save_model()
        run_risk_scoring()

# Run the check before loading data
ensure_data_exists()

# Helper function to load data safely
@st.cache_data
def load_data():
    processed_dir = "data/processed"
    
    try:
        df_risk = pd.read_csv(os.path.join(processed_dir, "risk_decisions.csv"))
        df_forecast = pd.read_csv(os.path.join(processed_dir, "future_forecast.csv"))
        df_sales = pd.read_csv(os.path.join(processed_dir, "clean_sales.csv"))
        return df_risk, df_forecast, df_sales, None
    except Exception as e:
        return None, None, None, str(e)

df_risk, df_forecast, df_sales, error_msg = load_data()

# Header banner
st.title("🔮 Project FORESIGHT")
st.subheader("Inventory Optimization & Demand Forecasting Suite — NorthBay Living")

if error_msg:
    st.error(f"Error loading processed datasets: {error_msg}")
    st.info("Please run the pipeline and forecast engine first to generate inputs (`make all`).")
else:
    # Sidebar Filters
    st.sidebar.header("Filter Inventory View")
    categories = ["All Categories"] + sorted(list(df_risk["category"].unique()))
    selected_category = st.sidebar.selectbox("Category", categories)
    
    # Filter dataset based on category selection
    if selected_category != "All Categories":
        filtered_risk = df_risk[df_risk["category"] == selected_category]
    else:
        filtered_risk = df_risk
        
    skus_in_cat = sorted(list(filtered_risk["sku_id"].unique()))
    selected_sku = st.sidebar.selectbox("SKU Deep-Dive Selection", skus_in_cat)
    
    # Calculate executive metrics for the current view
    total_sales_at_risk = filtered_risk["stockout_risk_val"].sum()
    total_capital_locked = filtered_risk["overstock_locked_capital"].sum()
    reorder_count = (filtered_risk["quadrant"] == "Reorder Now").sum()
    markdown_count = (filtered_risk["quadrant"] == "Markdown-Clear").sum()
    
    # Render KPI Cards in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card stockout">
            <div class="kpi-title">Revenue Sales At Risk</div>
            <div class="kpi-value">₹{total_sales_at_risk:,.0f}</div>
            <div class="kpi-subtitle">Due to projected stockouts before delivery</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card overstock">
            <div class="kpi-title">Capital Locked In Overstock</div>
            <div class="kpi-value">₹{total_capital_locked:,.0f}</div>
            <div class="kpi-subtitle">Excess stock beyond 6-week demand</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">SKUs Requiring Reorder</div>
            <div class="kpi-value">{reorder_count}</div>
            <div class="kpi-subtitle">Inventory Position below safety/reorder point</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card healthy">
            <div class="kpi-title">SKUs for Markdown Clearance</div>
            <div class="kpi-value">{markdown_count}</div>
            <div class="kpi-subtitle">SKUs with inventory > 2x forward demand</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Main content navigation tabs
    tab_decisions, tab_deepdive = st.tabs(["📋 Prioritized Inventory Decisions", "📈 SKU Demand Deep-Dive"])
    
    with tab_decisions:
        st.header("Decision Prioritization Grid")
        st.write("These lists are sorted by financial impact to help operations prioritize actions.")
        
        dec_col1, dec_col2 = st.columns(2)
        
        with dec_col1:
            st.subheader("🚨 REORDER NOW (Replenishment Action Required)")
            df_reorder = filtered_risk[filtered_risk["quadrant"] == "Reorder Now"].sort_values(by="stockout_risk_val", ascending=False)
            if len(df_reorder) > 0:
                # Select only critical columns for stakeholder
                st.dataframe(
                    df_reorder[["sku_id", "category", "on_hand_units", "on_order_units", "lead_time_days", "lead_time_demand", "stockout_risk_val", "recommended_action"]].rename(
                        columns={
                            "sku_id": "SKU",
                            "category": "Category",
                            "on_hand_units": "On Hand",
                            "on_order_units": "On Order",
                            "lead_time_days": "Lead Time",
                            "lead_time_demand": "LT Demand",
                            "stockout_risk_val": "Sales at Risk (INR)",
                            "recommended_action": "Action Plan"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No SKUs require immediate reordering.")
                
        with dec_col2:
            st.subheader("🏷️ MARKDOWN CLEARANCE (Capital Optimization)")
            df_markdown = filtered_risk[filtered_risk["quadrant"] == "Markdown-Clear"].sort_values(by="overstock_locked_capital", ascending=False)
            if len(df_markdown) > 0:
                st.dataframe(
                    df_markdown[["sku_id", "category", "on_hand_units", "forward_demand", "overstock_locked_capital", "recommended_action"]].rename(
                        columns={
                            "sku_id": "SKU",
                            "category": "Category",
                            "on_hand_units": "On Hand",
                            "forward_demand": "6-Wk Forecast",
                            "overstock_locked_capital": "Locked Capital (INR)",
                            "recommended_action": "Action Plan"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No SKUs currently flagged as overstocked.")
                
        # Draw a bar chart showing the categorization counts
        st.subheader("Inventory Health Segmentation")
        quadrant_counts = filtered_risk["quadrant"].value_counts().reset_index()
        quadrant_counts.columns = ["Quadrant", "Count"]
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#0f1116')
        ax.set_facecolor('#131722')
        colors = ["#10b981" if q == "Healthy" else "#ef4444" if q == "Reorder Now" else "#f59e0b" if q == "Markdown-Clear" else "#6366f1" for q in quadrant_counts["Quadrant"]]
        ax.barh(quadrant_counts["Quadrant"], quadrant_counts["Count"], color=colors, height=0.5)
        ax.set_xlabel("Number of SKUs", fontsize=10, color="#9ca3af")
        ax.grid(True, color="#1e222d", linestyle=":", alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#1e222d')
        ax.spines['bottom'].set_color('#1e222d')
        ax.tick_params(colors="#9ca3af", labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab_deepdive:
        st.header(f"Demand Profile & 6-Week Forecast: {selected_sku}")
        
        # Extract SKU attributes
        sku_risk_row = df_risk[df_risk["sku_id"] == selected_sku].iloc[0]
        sku_fc = df_forecast[df_forecast["sku_id"] == selected_sku].sort_values("date")
        sku_sales_hist = df_sales[df_sales["sku_id"] == selected_sku].sort_values("date")
        
        # Display SKU Metadata
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        with meta_col1:
            st.metric("Category", f"{sku_risk_row['category']} ({sku_risk_row['subcategory']})")
            st.metric("Unit Cost", f"₹{sku_risk_row['unit_cost']:,.0f}")
        with meta_col2:
            st.metric("Stock On Hand", f"{sku_risk_row['on_hand_units']:.0f} units")
            st.metric("Stock On Order", f"{sku_risk_row['on_order_units']:.0f} units")
        with meta_col3:
            st.metric("Lead Time", f"{sku_risk_row['lead_time_days']:.0f} days")
            st.metric("Reorder Point", f"{sku_risk_row['reorder_point']:.0f} units")
        with meta_col4:
            st.metric("Lead Time Demand", f"{sku_risk_row['lead_time_demand']:.1f} units")
            st.metric("Forward 6-Wk Demand", f"{sku_risk_row['forward_demand']:.1f} units")
            
        # Recommendations Alert
        quad = sku_risk_row["quadrant"]
        if quad == "Reorder Now":
            st.error(f"**Inventory Alert:** {sku_risk_row['recommended_action']}")
        elif quad == "Markdown-Clear":
            st.warning(f"**Inventory Alert:** {sku_risk_row['recommended_action']}")
        elif quad == "Watch-Volatile":
            st.info(f"**Inventory Alert:** {sku_risk_row['recommended_action']}")
        else:
            st.success(f"**Inventory Status:** {sku_risk_row['recommended_action']}")
            
        # Plot sales history (last 60 days of history) + future forecast (42 days)
        st.subheader("Historical Sales & Future Forecast Timeline")
        
        # Prepare historical plot data (last 60 days)
        hist_plot = sku_sales_hist.tail(60).copy()
        hist_dates = [datetime.strptime(x, "%Y-%m-%d").date() for x in hist_plot["date"].astype(str)]
        hist_sales = hist_plot["units_sold"].values
        
        # Prepare forecast plot data
        fc_dates = [datetime.strptime(x, "%Y-%m-%d").date() for x in sku_fc["date"].astype(str)]
        fc_point = sku_fc["forecast_units"].values
        fc_lower = sku_fc["forecast_lower_80"].values
        fc_upper = sku_fc["forecast_upper_80"].values
        
        # Matplotlib visualization (dark-theme styled)
        plt.style.use('dark_background')
        fig2, ax2 = plt.subplots(figsize=(12, 5.5), facecolor='#0f1116')
        ax2.set_facecolor('#131722')
        
        # Plot historical sales
        ax2.plot(hist_dates, hist_sales, label="Historical Sales", color="#9ca3af", marker="o", markersize=3, linewidth=1, alpha=0.8)
        
        # Plot forecast point estimate
        ax2.plot(fc_dates, fc_point, label="Forecast Demand", color="#6366f1", marker="o", markersize=4, linewidth=2)
        
        # Plot confidence interval bounds
        ax2.fill_between(fc_dates, fc_lower, fc_upper, color="#6366f1", alpha=0.15, label="80% Uncertainty Interval")
        
        # Vertical dotted line indicating where forecast begins
        forecast_start_date = fc_dates[0]
        ax2.axvline(x=forecast_start_date, color="#ef4444", linestyle="--", alpha=0.8, linewidth=1.5, label="Forecast Start (2026-07-01)")
        
        ax2.set_title(f"Sales Trend and 6-Week Demand Forecast for SKU: {selected_sku}", fontsize=13, fontweight="bold", pad=15)
        ax2.set_xlabel("Date", fontsize=10, color="#9ca3af")
        ax2.set_ylabel("Daily Units Sold", fontsize=10, color="#9ca3af")
        ax2.legend(loc="upper left", facecolor='#131722', edgecolor='none')
        
        # Format axes
        plt.xticks(rotation=25)
        ax2.grid(True, color="#1e222d", linestyle=":", alpha=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#1e222d')
        ax2.spines['bottom'].set_color('#1e222d')
        ax2.tick_params(colors="#9ca3af", labelsize=9)
        plt.tight_layout()
        
        st.pyplot(fig2)
