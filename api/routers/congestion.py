import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from api.schemas import CongestionRequest, CongestionResponse
from api.dependencies import get_artifact_store, ArtifactStore
from ml_pipeline.features import compute_live_congestion_features

router = APIRouter()


@router.post("/predict/congestion", response_model=CongestionResponse)
def predict_congestion(request: CongestionRequest,
                        horizon: str = Query("1h", pattern="^(1h|6h|24h)$"),
                        store: ArtifactStore = Depends(get_artifact_store)):
    if horizon not in store.congestion_models:
        raise HTTPException(status_code=400, detail=f"Unknown horizon '{horizon}'")

    window_df = pd.DataFrame([r.model_dump() for r in request.telemetry_window])
    feature_dict = compute_live_congestion_features(window_df, request.static.model_dump())
    X = store.encode_row(feature_dict, store.congestion_schema)

    model = store.congestion_models[horizon]
    forecast = float(model.predict(X)[0])
    latest_ts = window_df["timestamp"].max()

    return CongestionResponse(
        cell_id=request.cell_id, timestamp=latest_ts, horizon=horizon,
        forecast_prb_utilization_pct=forecast, history_hours_used=len(window_df),
    )
