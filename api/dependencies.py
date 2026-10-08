import json
import joblib
import pandas as pd
from functools import lru_cache
from ml_pipeline.config import ARTIFACTS_DIR


class ArtifactStore:
    def __init__(self):
        self.cell_fault_model = joblib.load(ARTIFACTS_DIR / "cell_fault_model.joblib")
        self.cell_fault_explainer = joblib.load(ARTIFACTS_DIR / "shap_explainer_cell_fault.joblib")
        with open(ARTIFACTS_DIR / "cell_fault_schema.json") as f:
            self.cell_fault_schema = json.load(f)
        with open(ARTIFACTS_DIR / "cell_fault_metadata.json") as f:
            self.cell_fault_metadata = json.load(f)

        self.congestion_models = {
            horizon: joblib.load(ARTIFACTS_DIR / f"congestion_model_{horizon}.joblib")
            for horizon in ["1h", "6h", "24h"]
        }
        with open(ARTIFACTS_DIR / "congestion_schema.json") as f:
            self.congestion_schema = json.load(f)

        self.anomaly_baselines = pd.read_parquet(ARTIFACTS_DIR / "anomaly_baselines.parquet")

    def encode_row(self, feature_dict: dict, schema: dict) -> pd.DataFrame:
        """Applies the exact one-hot encoding saved at training time, so an unseen
        category at serve time is safely zero-filled rather than crashing or
        misaligning columns."""
        row = {col: feature_dict.get(col, None) for col in schema["numeric_features"]}
        for col, cats in schema["categories"].items():
            value = feature_dict.get(col)
            for cat in cats[1:]:
                row[f"{col}_{cat}"] = int(value == cat)
        ordered = {c: row.get(c, 0) for c in schema["encoded_feature_order"]}
        return pd.DataFrame([ordered])


@lru_cache(maxsize=1)
def get_artifact_store() -> ArtifactStore:
    return ArtifactStore()
