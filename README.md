# Telecom Network Intelligence

A synthetic telecom network covering topology, 90 days of operational telemetry, and a served machine learning layer. Built to demonstrate end-to-end network analytics engineering from causal data generation through feature engineering, model validation, and production API deployment.

> **Note on data:** The network topology, subscriber demand, and fault events are synthetic, generated to be structurally and statistically plausible (modeled loosely on the scale and geography of a large Nigerian mobile operator) without claiming to represent any real operator's infrastructure or performance.

---

## Project Motivation

Telecom operators run Network Operations Centers (NOCs) that need to answer three questions continuously: *is something about to break, is the network under pressure, and is this behavior actually unusual?* Real telemetry for building and validating this kind of tooling is proprietary and rarely available for public portfolio work. This project builds a network from first principles with enough structural realism that the resulting analytics and ML problems are genuine, not manufactured, to demonstrate the full pipeline a telecom data/ML engineer would own in practice.

The build follows one rule throughout: **the simulator generates evidence, never answers.** No table says `fault = True` directly; instead, a fault changes upstream conditions (utilization, radio quality, link state), and those changes propagate through the network's real topology into observable telemetry. Every downstream model infers what happened from the same evidence a NOC engineer would have, keeping this from being a toy dataset with pre-baked labels.

---

## Network Architecture & Scale

| Layer | Scale |
|---|---|
| Regions | 5 (geographic, with distinct demand profiles) |
| Sites | 50 |
| Cells | 130+ (4G/5G, multi-sector, multi-band) |
| Aggregation nodes | 10 (geographically clustered, capacity-constrained) |
| Transport mesh links | ~20 (redundant, sized from routed failover load) |
| Core nodes | 5 (pooled primary/secondary dependency) |
| Telemetry window | 90 days at 15-minute resolution |

The topology is not arbitrary: aggregation clustering runs k-means-constrained on real geographic coordinates, transport redundancy is built via a minimum spanning tree (Kruskal's algorithm) augmented with shortest-path failover routing, and every layer's capacity is sized bottom-up from the actual traffic it carries rather than assigned independently. Utilization, congestion, and cascading failure are structural properties of the network, not scripted outcomes.

---

## Use Cases

| Use case | Status | Notes |
|---|---|---|
| **Concurrent fault detection** (cell-level) | Deployed | XGBoost + SHAP; 32x lift over base rate on a 0.5%-positive dataset, served via API |
| **Congestion forecasting** (1h / 6h / 24h) | Deployed | Beats seasonal-naive baseline by 74 to 96% MAE |
| **Anomaly detection** | Deployed | Seasonal baseline + z-score, evaluated against three distinct phenomena (fault / congestion / anomaly) |

---

## System Structure

```
simulator/            topology + capacity + 90-day telemetry generation
ml_pipeline/          feature engineering + model training (shared by
                      training and live inference to prevent skew)
artifacts/            trained models, SHAP explainers, feature schemas
api/                  FastAPI serving layer (stateless, loads artifacts/
                      at startup, never touches the simulator)
scripts/              single-command pipeline runner and results audits
notebooks/            exploratory + analytics-layer notebook (BigQuery / Looker)
```

The data-generation pipeline and the serving API are two independent lifecycles that meet at one point: the `artifacts/` directory. The API never regenerates data or retrains anything; it loads what is already there. This mirrors how a real ML system separates offline training from online serving.

---

## Technical Scope

- **Network engineering and graph algorithms:** Kruskal MST for transport redundancy, shortest-path failover routing (NetworkX), haversine-based geographic distance, capacity-constrained clustering
- **Causal synthetic data design:** multi-layer capacity chains (cell to backhaul to aggregation to core), topology-driven fault propagation, deliberate separation of fault, congestion, and anomaly as distinct, non-overlapping phenomena
- **Time-series feature engineering:** trailing rolling statistics, lag features, linear-trend slopes, seasonal baselines, strict temporal leakage discipline (train/val/test split with window-aware gaps)
- **ML validation rigor:** precision-recall analysis under severe class imbalance, blocked time-series cross-validation, SHAP-based feature attribution, and declining to deploy a model when the validation evidence did not support it
- **Production ML serving:** FastAPI, Pydantic schema validation, stateless feature computation shared between training and inference, versioned model artifacts
- **Analytics engineering:** BigQuery data marts, Looker Studio dashboards, SQL view design

---

## Design Decisions & Validation Findings

- **Concurrent detection over future prediction:** 4-hour-ahead fault forecasting was tested and found infeasible because fault onset is exogenous (no learnable precursor signal exists in prior telemetry). PR-AUC landed at ~0.002, statistically indistinguishable from random. The project pivoted to concurrent detection (identifying active faults right now) where the signal is real: faulted cells show Cohen's d > 1.6 on drop rate and alarm features, and the deployed model achieves 32x lift over base rate on a severely class-imbalanced dataset (0.5% positive hours).
- **Exclusion of mesh and core fault detectors:** Mesh link telemetry carries zero baseline traffic outside failover events, so sanitized features have no variance to learn from. Core faults number too few to support any held-out evaluation. The investigation is documented in the reports rather than forced into a misleading endpoint.
- **Stateless serving design:** The live cell fault detector uses RAN-layer telemetry only to keep the API stateless. The cross-layer-enriched version (including backhaul and core-uplink context) measurably outperforms it in offline validation. Extending the live feature set to accept current backhaul context is the direct next iteration for operational use.

---

## Running Locally

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the network and train models (optional checkpointed run)

```bash
python scripts/run_full_pipeline.py
```

This generates the static topology, 90 days of telemetry, builds ML feature tables, and trains every model, writing results to `artifacts/`. Each stage checkpoints to `data/` and is skipped on rerun unless `--force` is passed.

```bash
python scripts/run_full_pipeline.py --stop-after topology   # inspect just the network
python scripts/run_full_pipeline.py --force                 # full regeneration
```

### 3. Run the API locally

```bash
uvicorn api.main:app --reload
```

Interactive OpenAPI docs will be available at `http://127.0.0.1:8000/docs`.

---

## API Reference

### `GET /health`

Confirms all serialized artifacts loaded successfully into memory at startup.

### `POST /predict/fault/cell`

Detects whether a cell is currently experiencing an active fault from its recent telemetry window.

```json
{
  "cell_id": "cell_00001",
  "static": {
    "technology": "5G",
    "band": "3500MHz",
    "site_type": "urban",
    "region_name": "Lagos",
    "cell_capacity_mbps": 800
  },
  "telemetry_window": [
    {"timestamp": "2025-03-20T14:00:00", "prb_util_mean": 62.1, "traffic_mbps_mean": 496.8, "sinr_mean": 17.9, "bler_mean": 2.4, "drop_rate_mean": 0.4, "availability_mean": 100.0, "handover_success_mean": 96.8, "alarm_count": 0},
    {"timestamp": "2025-03-20T15:00:00", "prb_util_mean": 64.3, "traffic_mbps_mean": 514.4, "sinr_mean": 17.5, "bler_mean": 2.6, "drop_rate_mean": 0.4, "availability_mean": 100.0, "handover_success_mean": 96.5, "alarm_count": 0},
    {"timestamp": "2025-03-20T16:00:00", "prb_util_mean": 66.0, "traffic_mbps_mean": 528.0, "sinr_mean": 16.8, "bler_mean": 3.0, "drop_rate_mean": 0.5, "availability_mean": 100.0, "handover_success_mean": 96.1, "alarm_count": 0},
    {"timestamp": "2025-03-20T17:00:00", "prb_util_mean": 68.9, "traffic_mbps_mean": 551.2, "sinr_mean": 15.2, "bler_mean": 3.8, "drop_rate_mean": 0.7, "availability_mean": 100.0, "handover_success_mean": 95.0, "alarm_count": 0},
    {"timestamp": "2025-03-20T18:00:00", "prb_util_mean": 71.4, "traffic_mbps_mean": 571.2, "sinr_mean": 13.1, "bler_mean": 4.9, "drop_rate_mean": 1.1, "availability_mean": 100.0, "handover_success_mean": 93.2, "alarm_count": 1},
    {"timestamp": "2025-03-20T19:00:00", "prb_util_mean": 73.0, "traffic_mbps_mean": 584.0, "sinr_mean": 11.0, "bler_mean": 6.3, "drop_rate_mean": 1.6, "availability_mean": 100.0, "handover_success_mean": 91.0, "alarm_count": 1}
  ]
}
```

### `POST /predict/fault/cell/explain`

Accepts the same request body and returns the SHAP feature contributions explaining the prediction.

### `POST /predict/congestion?horizon=1h` (also `6h`, `24h`)

Forecasts future resource utilization and capacity headroom across designated time horizons.

```json
{
  "cell_id": "cell_00001",
  "static": {
    "technology": "5G",
    "band": "3500MHz",
    "site_type": "urban",
    "region_name": "Lagos",
    "cell_capacity_mbps": 800
  },
  "telemetry_window": [
    {"timestamp": "2025-03-20T10:00:00", "prb_util_mean": 55.0, "traffic_mbps_mean": 440.0, "sinr_mean": 18.0, "bler_mean": 2.0, "drop_rate_mean": 0.3, "availability_mean": 100.0, "handover_success_mean": 97.0, "alarm_count": 0},
    {"timestamp": "2025-03-20T11:00:00", "prb_util_mean": 58.0, "traffic_mbps_mean": 464.0, "sinr_mean": 17.8, "bler_mean": 2.1, "drop_rate_mean": 0.3, "availability_mean": 100.0, "handover_success_mean": 97.0, "alarm_count": 0},
    {"timestamp": "2025-03-20T12:00:00", "prb_util_mean": 60.5, "traffic_mbps_mean": 484.0, "sinr_mean": 17.9, "bler_mean": 2.0, "drop_rate_mean": 0.3, "availability_mean": 100.0, "handover_success_mean": 97.1, "alarm_count": 0},
    {"timestamp": "2025-03-20T13:00:00", "prb_util_mean": 61.8, "traffic_mbps_mean": 494.4, "sinr_mean": 17.7, "bler_mean": 2.2, "drop_rate_mean": 0.3, "availability_mean": 100.0, "handover_success_mean": 97.0, "alarm_count": 0},
    {"timestamp": "2025-03-20T14:00:00", "prb_util_mean": 62.1, "traffic_mbps_mean": 496.8, "sinr_mean": 17.9, "bler_mean": 2.4, "drop_rate_mean": 0.4, "availability_mean": 100.0, "handover_success_mean": 96.8, "alarm_count": 0}
  ]
}
```

### `POST /predict/anomaly`

Scores recent values against the precomputed seasonal baseline lookup table.

```json
{
  "cell_id": "cell_00001",
  "metric_name": "prb_util_mean",
  "recent_values": [
    {"timestamp": "2025-03-20T13:00:00", "value": 61.8},
    {"timestamp": "2025-03-20T14:00:00", "value": 91.5}
  ]
}
```

---

## Analytics Dashboard

BigQuery data marts and Looker Studio report covering network health, RAN metrics, capacity headroom, and regional distribution:

[Looker Studio Dashboard](https://datastudio.google.com/reporting/6e9a6d18-6c9d-494f-adba-21f3db47ef4a)

---

## Deployment Configuration

Configured for deployment on [Render](https://render.com) using standard web service specifications in `render.yaml`. The service loads serialized artifacts from `artifacts/` at boot and serves inference requests via FastAPI.

---

## Technology Stack

Python, pandas, NumPy, SciPy, NetworkX, scikit-learn, XGBoost, SHAP, FastAPI, Pydantic, Google BigQuery, Looker Studio, Render

