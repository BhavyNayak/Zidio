import os
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="FORESIGHT Inventory Risk Scoring Service",
    description="Microservice providing real-time forecasting demand metrics, stockout risk values, and overstock analysis.",
    version="1.0.0"
)

# 1. Define Request/Response Schemas via Pydantic
class ScoreRequest(BaseModel):
    sku_ids: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of SKU IDs to score. If empty or null, returns scoring for all active SKUs."
    )

class SKUScoreResult(BaseModel):
    sku_id: str = Field(..., example="SKU001")
    category: str = Field(..., example="Bedroom")
    subcategory: str = Field(..., example="Beds")
    on_hand_units: float = Field(..., example=45.0)
    on_order_units: float = Field(..., example=30.0)
    inventory_position: float = Field(..., example=75.0)
    lead_time_days: int = Field(..., example=12)
    reorder_point: float = Field(..., example=18.0)
    lead_time_demand: float = Field(..., example=15.2)
    forward_demand: float = Field(..., example=62.4)
    stockout_units: float = Field(..., example=0.0)
    stockout_risk_val: float = Field(..., example=0.0)
    overstock_units: float = Field(..., example=15.0)
    overstock_locked_capital: float = Field(..., example=7500.0)
    quadrant: str = Field(..., example="Markdown-Clear")
    recommended_action: str = Field(..., example="Promote and Markdown by 20-30%...")

class ScoreResponse(BaseModel):
    total_skus_evaluated: int
    skus: List[SKUScoreResult]

# Helper function to read from processed CSV
def load_risk_decisions() -> pd.DataFrame:
    path = "data/processed/risk_decisions.csv"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503, 
            detail="Scoring engine output is unavailable. Please run pipeline and training first."
        )
    return pd.read_csv(path)

# 2. Endpoints
@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to Project FORESIGHT scoring service.",
        "docs_url": "/docs",
        "status": "active"
    }

@app.get("/health", tags=["General"])
def health_check():
    processed_exists = os.path.exists("data/processed/risk_decisions.csv")
    model_exists = os.path.exists("src/models/forecast_artifacts.pkl")
    
    return {
        "status": "healthy" if (processed_exists and model_exists) else "degraded",
        "pipeline_data_available": processed_exists,
        "trained_model_available": model_exists
    }

@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_inventory(payload: ScoreRequest):
    """
    Accepts list of SKU IDs and returns forecasting summaries, stockout risks, overstock values, 
    and recommended operations actions.
    """
    df_risk = load_risk_decisions()
    
    # Filter SKUs if specified in payload
    if payload.sku_ids is not None and len(payload.sku_ids) > 0:
        # Check if any SKU is missing from our system
        missing_skus = [sku for sku in payload.sku_ids if sku not in df_risk["sku_id"].values]
        if len(missing_skus) > 0:
            raise HTTPException(
                status_code=404, 
                detail=f"The following requested SKU IDs were not found: {missing_skus}"
            )
        df_filtered = df_risk[df_risk["sku_id"].isin(payload.sku_ids)]
    else:
        df_filtered = df_risk
        
    skus_list = []
    for idx, row in df_filtered.iterrows():
        skus_list.append(SKUScoreResult(
            sku_id=row["sku_id"],
            category=row["category"],
            subcategory=row["subcategory"],
            on_hand_units=float(row["on_hand_units"]),
            on_order_units=float(row["on_order_units"]),
            inventory_position=float(row["inventory_position"]),
            lead_time_days=int(row["lead_time_days"]),
            reorder_point=float(row["reorder_point"]),
            lead_time_demand=float(row["lead_time_demand"]),
            forward_demand=float(row["forward_demand"]),
            stockout_units=float(row["stockout_units"]),
            stockout_risk_val=float(row["stockout_risk_val"]),
            overstock_units=float(row["overstock_units"]),
            overstock_locked_capital=float(row["overstock_locked_capital"]),
            quadrant=row["quadrant"],
            recommended_action=row["recommended_action"]
        ))
        
    return ScoreResponse(
        total_skus_evaluated=len(skus_list),
        skus=skus_list
    )
