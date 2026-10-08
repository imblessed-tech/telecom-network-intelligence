import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from api.schemas import AnomalyRequest, AnomalyResponse
from api.dependencies import get_artifact_store, ArtifactStore
from ml_pipeline.config import ANOMALY_Z_THRESHOLD, ANOMALY_PERSISTENCE_HOURS

router = APIRouter()


@router.post("/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(request: AnomalyRequest, store: ArtifactStore = Depends(get_artifact_store)):
    values = sorted(request.recent_values, key=lambda v: v.timestamp)
    latest = values[-1]
    latest_ts = pd.Timestamp(latest.timestamp)

    baseline_row = store.anomaly_baselines[
        (store.anomaly_baselines.cell_id == request.cell_id)
        & (store.anomaly_baselines.hour_of_day == latest_ts.hour)
        & (store.anomaly_baselines.day_of_week == latest_ts.dayofweek)
    ]
    if baseline_row.empty:
        return AnomalyResponse(cell_id=request.cell_id, metric_name=request.metric_name,
                                timestamp=latest_ts, actual_value=latest.value,
                                expected_value=None, z_score=None, is_anomaly_flagged=False)

    expected = baseline_row[f"{request.metric_name}_expected"].iloc[0]
    std = baseline_row[f"{request.metric_name}_std"].iloc[0]
    if pd.isna(expected) or pd.isna(std) or std == 0:
        return AnomalyResponse(cell_id=request.cell_id, metric_name=request.metric_name,
                                timestamp=latest_ts, actual_value=latest.value,
                                expected_value=float(expected) if pd.notna(expected) else None,
                                z_score=None, is_anomaly_flagged=False)

    z_scores = [(v.value - expected) / std for v in values[-ANOMALY_PERSISTENCE_HOURS:]]
    breach_streak = all(abs(z) > ANOMALY_Z_THRESHOLD for z in z_scores) and len(z_scores) >= ANOMALY_PERSISTENCE_HOURS

    return AnomalyResponse(
        cell_id=request.cell_id, metric_name=request.metric_name, timestamp=latest_ts,
        actual_value=latest.value, expected_value=float(expected), z_score=float(z_scores[-1]),
        is_anomaly_flagged=breach_streak,
    )
