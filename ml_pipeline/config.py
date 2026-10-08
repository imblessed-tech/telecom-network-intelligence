from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Feature engineering constants — single source of truth, shared by training (batch)
# and serving (live) feature computation so the two can never silently drift apart.
LAG_HOURS = [1, 4]
ROLLING_WINDOW_HOURS = 4
ALARM_TRAILING_WINDOW_HOURS = 24
NO_ALARM_SENTINEL_HOURS = 9999  # "never alarmed" must NOT equal 0 ("alarm just fired") — see prior session bug

CONGESTION_LAG_HOURS = [1, 2, 24, 168]
CONGESTION_ROLLING_24H = 24
CONGESTION_ROLLING_7D = 24 * 7
CONGESTION_SLOPE_14D = 24 * 14
CONGESTION_HORIZONS = {"1h": 1, "6h": 6, "24h": 24}
MIN_CONGESTION_HISTORY_HOURS = 24 * 14  # needed for the 14-day slope feature

ANOMALY_Z_THRESHOLD = 2.5
ANOMALY_PERSISTENCE_HOURS = 2

# Deployed cell-fault model feature set — SANITIZED (no availability_mean, no region
# dummies): both were found to be near-tautological / low-sample-size shortcuts during
# validation, not genuine behavioral precursors. See training notes in train.py.
CELL_FAULT_FEATURES_NUMERIC = [
    "sinr_mean", "bler_mean", "drop_rate_mean", "prb_util_mean", "handover_success_mean",
    "sinr_lag_1h", "bler_lag_1h", "drop_rate_lag_1h", "prb_lag_1h", "prb_lag_4h",
    "prb_mean_4h", "prb_std_4h", "sinr_mean_4h", "sinr_slope_4h",
    "bler_mean_4h", "drop_rate_mean_4h",
    "prb_change_1h", "sinr_change_4h", "bler_change_4h",
    "alarm_count_trailing_24h", "hours_since_last_alarm",
    "hour_of_day", "day_of_week", "is_weekend", "cell_capacity_mbps",
]
CELL_FAULT_FEATURES_CATEGORICAL = ["technology", "band", "site_type"]

CONGESTION_FEATURES_NUMERIC = [
    "prb_util_t", "traffic_t", "capacity_headroom_current",
    "prb_lag_1h", "prb_lag_2h", "prb_lag_24h", "prb_lag_168h",
    "traffic_lag_1h", "traffic_lag_24h",
    "prb_mean_24h", "prb_std_24h", "prb_mean_7d", "prb_slope_7d",
    "traffic_mean_24h", "traffic_slope_7d",
    "hour_of_day", "day_of_week", "is_weekend", "cell_capacity_mbps",
]
CONGESTION_FEATURES_CATEGORICAL = ["technology", "region_name", "site_type"]

# ---- Mesh / core concurrent fault detection ----
MESH_FEATURES_NUMERIC_ALL = [
    "utilization_mean", "utilization_max", "util_lag_1h", "util_mean_4h", "util_max_4h", "util_change_1h",
    "latency_mean_ms", "jitter_mean_ms", "packet_loss_mean_pct", "availability_mean",
    "latency_mean_4h", "jitter_mean_4h", "packet_loss_mean_4h",
]
# Near-deterministic functions of fault `intensity` in the generator (no/negligible noise):
# a detector using them re-derives the fault's definition rather than learning a pattern.
MESH_PROXY_FEATURES = [
    "latency_mean_ms", "jitter_mean_ms", "packet_loss_mean_pct", "availability_mean",
    "latency_mean_4h", "jitter_mean_4h", "packet_loss_mean_4h",
]
MESH_FEATURES_NUMERIC_SANITIZED = [c for c in MESH_FEATURES_NUMERIC_ALL if c not in MESH_PROXY_FEATURES]

# alarm_count omitted: core alarms never fire at current fault severities (constant 0)
CORE_FEATURES_NUMERIC_ALL = [
    "cpu_mean_pct", "cpu_max_pct", "memory_mean_pct", "memory_max_pct",
    "session_failure_rate_mean", "availability_mean",
    "cpu_lag_1h", "cpu_mean_4h", "cpu_std_4h", "cpu_change_1h", "memory_mean_4h", "session_failure_mean_4h",
]
CORE_PROXY_FEATURES = ["session_failure_rate_mean", "availability_mean", "session_failure_mean_4h"]
CORE_FEATURES_NUMERIC_SANITIZED = [c for c in CORE_FEATURES_NUMERIC_ALL if c not in CORE_PROXY_FEATURES]

MIN_FAULT_EVENTS_FOR_MODEL = 15   # distinct fault episodes needed before any held-out evaluation is meaningful
MIN_DEPLOY_LIFT = 3.0             # sanitized blocked-CV PR-AUC must beat base rate by this factor to be saved
DETECTION_CV_FOLDS = 4
DETECTION_CV_GAP_HOURS = 24