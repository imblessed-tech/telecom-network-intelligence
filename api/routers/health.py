from fastapi import APIRouter, Depends
from api.dependencies import get_artifact_store, ArtifactStore

router = APIRouter()


@router.get("/health")
def health(store: ArtifactStore = Depends(get_artifact_store)):
    return {
        "status": "ok",
        "cell_fault_model_loaded": store.cell_fault_model is not None,
        "congestion_models_loaded": list(store.congestion_models.keys()),
        "anomaly_baseline_rows": len(store.anomaly_baselines),
    }
