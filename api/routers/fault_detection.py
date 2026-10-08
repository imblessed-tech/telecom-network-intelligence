import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from api.schemas import FaultDetectionRequest, FaultDetectionResponse, FaultExplanationResponse, FeatureContribution
from api.dependencies import get_artifact_store, ArtifactStore
from ml_pipeline.features import compute_live_cell_fault_features

router = APIRouter()


def _prepare(request: FaultDetectionRequest, store: ArtifactStore):
    window_df = pd.DataFrame([r.model_dump() for r in request.telemetry_window])
    try:
        feature_dict = compute_live_cell_fault_features(window_df, request.static.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    X = store.encode_row(feature_dict, store.cell_fault_schema)
    latest_ts = window_df["timestamp"].max()
    return X, latest_ts


@router.post("/predict/fault/cell", response_model=FaultDetectionResponse)
def predict_cell_fault(request: FaultDetectionRequest, store: ArtifactStore = Depends(get_artifact_store)):
    X, latest_ts = _prepare(request, store)
    prob = float(store.cell_fault_model.predict_proba(X)[:, 1][0])
    threshold = store.cell_fault_schema["decision_threshold"]
    return FaultDetectionResponse(
        cell_id=request.cell_id, timestamp=latest_ts,
        fault_probability=prob, is_fault_flagged=prob >= threshold, threshold_used=threshold,
    )


@router.post("/predict/fault/cell/explain", response_model=FaultExplanationResponse)
def explain_cell_fault(request: FaultDetectionRequest, store: ArtifactStore = Depends(get_artifact_store),
                        top_n: int = 8):
    X, latest_ts = _prepare(request, store)
    prob = float(store.cell_fault_model.predict_proba(X)[:, 1][0])
    threshold = store.cell_fault_schema["decision_threshold"]

    shap_values = store.cell_fault_explainer.shap_values(X)
    row_shap = shap_values[0] if shap_values.ndim == 2 else shap_values
    contributions = sorted(
        [FeatureContribution(feature=col, value=float(X[col].iloc[0]), shap_value=float(val))
         for col, val in zip(X.columns, row_shap)],
        key=lambda c: abs(c.shap_value), reverse=True,
    )[:top_n]

    return FaultExplanationResponse(
        cell_id=request.cell_id, timestamp=latest_ts,
        fault_probability=prob, is_fault_flagged=prob >= threshold, threshold_used=threshold,
        top_contributions=contributions,
    )
