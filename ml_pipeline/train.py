import json
import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve, mean_absolute_error

from ml_pipeline.config import (
    ARTIFACTS_DIR, CELL_FAULT_FEATURES_NUMERIC, CELL_FAULT_FEATURES_CATEGORICAL,
    CONGESTION_FEATURES_NUMERIC, CONGESTION_FEATURES_CATEGORICAL, CONGESTION_HORIZONS,
)
from ml_pipeline.features import build_batch_cell_fault_features


from ml_pipeline.config import (
    ARTIFACTS_DIR, CELL_FAULT_FEATURES_NUMERIC, CELL_FAULT_FEATURES_CATEGORICAL,
    CONGESTION_FEATURES_NUMERIC, CONGESTION_FEATURES_CATEGORICAL, CONGESTION_HORIZONS,
    MESH_FEATURES_NUMERIC_ALL, MESH_FEATURES_NUMERIC_SANITIZED,
    CORE_FEATURES_NUMERIC_ALL, CORE_FEATURES_NUMERIC_SANITIZED, CORE_PROXY_FEATURES,
    MIN_FAULT_EVENTS_FOR_MODEL, MIN_DEPLOY_LIFT, DETECTION_CV_FOLDS, DETECTION_CV_GAP_HOURS,
)
from ml_pipeline.features import (
    build_batch_cell_fault_features, build_batch_mesh_fault_features, build_batch_core_fault_features,
)
from ml_pipeline.labels import build_concurrent_mesh_labels, build_concurrent_core_labels, count_fault_events

def encode_categoricals(df, categorical_cols, known_categories=None):
    """One-hot encode with an explicit, saved category list so train/serve encoding
    can never silently diverge (a new category at serve time is safely dropped, not
    a crash or a shifted column)."""
    df = df.copy()
    encoded_cols = []
    categories_used = {}
    for col in categorical_cols:
        cats = known_categories[col] if known_categories else sorted(df[col].dropna().unique().tolist())
        categories_used[col] = cats
        for cat in cats[1:]:  # drop_first equivalent, deterministic ordering
            new_col = f"{col}_{cat}"
            df[new_col] = (df[col] == cat).astype(int)
            encoded_cols.append(new_col)
    return df, encoded_cols, categories_used


def train_cell_fault_model(hourly_ran_labeled, label_col="is_currently_faulted",
                            train_end="2025-03-04", val_end="2025-03-18"):
    """hourly_ran_labeled: ml_hourly_ran with `label_col` ALREADY merged in (pipeline does this)."""
    df = build_batch_cell_fault_features(hourly_ran_labeled)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["prb_lag_4h", "sinr_change_4h", "bler_change_4h", "sinr_slope_4h", "prb_std_4h"])

    df, encoded_cat_cols, categories = encode_categoricals(df, CELL_FAULT_FEATURES_CATEGORICAL)
    feature_cols = CELL_FAULT_FEATURES_NUMERIC + encoded_cat_cols

    train = df[df.timestamp <= train_end]
    val = df[(df.timestamp > train_end) & (df.timestamp <= val_end)]
    test = df[df.timestamp > val_end]
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part[label_col].sum() == 0:
            raise ValueError(f"No positive labels in the {name} split — adjust split dates.")

    X_train, y_train = train[feature_cols], train[label_col]
    X_val, y_val = val[feature_cols], val[label_col]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(scale_pos_weight=pos_weight, n_estimators=200, max_depth=4,
                           learning_rate=0.05, min_child_weight=5, colsample_bytree=0.8,
                           subsample=0.8, eval_metric="aucpr", random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_prauc = average_precision_score(y_val, val_probs)
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    valid_idx = np.where(recalls[:-1] >= 0.7)[0]
    chosen_threshold = float(thresholds[valid_idx[-1]]) if len(valid_idx) else 0.5

    test_prauc = average_precision_score(test[label_col], model.predict_proba(test[feature_cols])[:, 1])
    explainer = shap.TreeExplainer(model)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "cell_fault_model.joblib")
    joblib.dump(explainer, ARTIFACTS_DIR / "shap_explainer_cell_fault.joblib")
    schema = {"numeric_features": CELL_FAULT_FEATURES_NUMERIC,
              "categorical_features": CELL_FAULT_FEATURES_CATEGORICAL, "categories": categories,
              "encoded_feature_order": feature_cols, "decision_threshold": chosen_threshold}
    with open(ARTIFACTS_DIR / "cell_fault_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    metadata = {"model": "cell_fault_detector", "task": "concurrent silent-fault detection",
                "val_prauc": val_prauc, "test_prauc": test_prauc, "val_base_rate": float(y_val.mean()),
                "threshold": chosen_threshold, "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
                "note": "Sanitized: availability_mean and region features excluded (tautological / low-sample shortcuts)."}
    with open(ARTIFACTS_DIR / "cell_fault_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Cell fault model — val PR-AUC {val_prauc:.4f}, test PR-AUC {test_prauc:.4f}, threshold {chosen_threshold:.4f}")
    return model, schema, metadata

def _fit_xgb(X, y):
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    model = XGBClassifier(scale_pos_weight=pos_weight, n_estimators=150, max_depth=3, learning_rate=0.05,
                           min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                           eval_metric="aucpr", random_state=42)
    model.fit(X, y)
    return model


def blocked_time_cv(df, feature_cols, label_col, n_folds=DETECTION_CV_FOLDS, gap_hours=DETECTION_CV_GAP_HOURS):
    """Contiguous time blocks as test folds; training rows within `gap_hours` of the test
    block are purged so rolling windows can't straddle the boundary. Valid for DETECTION
    (concurrent labels); not valid for forecasting tasks."""
    stamps = np.sort(df["timestamp"].unique())
    gap = pd.Timedelta(hours=gap_hours)
    folds = []
    for i, block in enumerate(np.array_split(stamps, n_folds)):
        t0, t1 = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        test = df[df.timestamp.between(t0, t1)]
        train = df[~df.timestamp.between(t0 - gap, t1 + gap)]
        if test[label_col].sum() == 0 or train[label_col].sum() == 0:
            folds.append({"fold": i, "test_positive_hours": int(test[label_col].sum()),
                          "prauc": None, "base_rate": float(test[label_col].mean()), "lift": None})
            continue
        model = _fit_xgb(train[feature_cols], train[label_col])
        prauc = average_precision_score(test[label_col], model.predict_proba(test[feature_cols])[:, 1])
        base = float(test[label_col].mean())
        folds.append({"fold": i, "test_positive_hours": int(test[label_col].sum()), "prauc": float(prauc),
                      "base_rate": base, "lift": float(prauc / base)})
    return folds


def cohens_d_table(df, feature_cols, label_col):
    healthy, faulty = df[df[label_col] == 0], df[df[label_col] == 1]
    rows = []
    for col in feature_cols:
        h, f = healthy[col].dropna(), faulty[col].dropna()
        if len(f) < 2 or len(h) < 2:
            continue
        pooled = np.sqrt(((len(h) - 1) * h.std() ** 2 + (len(f) - 1) * f.std() ** 2) / (len(h) + len(f) - 2))
        rows.append({"feature": col, "healthy_mean": float(h.mean()), "faulty_mean": float(f.mean()),
                     "cohens_d": float((f.mean() - h.mean()) / (pooled + 1e-9))})
    return sorted(rows, key=lambda r: abs(r["cohens_d"]), reverse=True)


def _run_detector_study(df, entity_col, numeric_all, numeric_sanitized, categorical, label_col, name):
    df = df.copy()
    df, enc_cols, categories = encode_categoricals(df, categorical) if categorical else (df, [], {})
    report = {"name": name, "rows": len(df), "positive_hours": int(df[label_col].sum()),
              "fault_events": count_fault_events(df, entity_col, label_col), "variants": {}, "warnings": []}

    for variant, numeric in (("full", numeric_all), ("sanitized", numeric_sanitized)):
        feats = numeric + enc_cols
        constant = [c for c in feats if df[c].nunique(dropna=True) <= 1]
        folds = blocked_time_cv(df, feats, label_col)
        valid = [f for f in folds if f["prauc"] is not None]
        report["variants"][variant] = {
            "features": feats, "constant_features": constant, "folds": folds, "valid_folds": len(valid),
            "mean_prauc": float(np.mean([f["prauc"] for f in valid])) if valid else None,
            "mean_lift": float(np.mean([f["lift"] for f in valid])) if valid else None,
        }
        if constant and len(constant) == len(feats):
            report["warnings"].append(f"{variant}: every feature is constant — no behavioral signal exists.")

    if report["variants"]["full"]["mean_lift"] and report["variants"]["full"]["mean_lift"] >= MIN_DEPLOY_LIFT:
        report["warnings"].append("full-feature score reflects deterministic fault-effect formulas in the "
                                  "simulator (upper bound / sanity check), not generalizable learning.")

    san = report["variants"]["sanitized"]
    report["deployable"] = bool(san["mean_lift"] is not None and san["mean_lift"] >= MIN_DEPLOY_LIFT
                                and san["valid_folds"] >= 2)
    if report["deployable"]:
        feats = numeric_sanitized + enc_cols
        model = _fit_xgb(df[feats], df[label_col])
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ARTIFACTS_DIR / f"{name}_model.joblib")
        joblib.dump(shap.TreeExplainer(model), ARTIFACTS_DIR / f"shap_explainer_{name}.joblib")
        with open(ARTIFACTS_DIR / f"{name}_schema.json", "w") as f:
            json.dump({"numeric_features": numeric_sanitized, "categorical_features": categorical,
                       "categories": categories, "encoded_feature_order": feats}, f, indent=2)

    with open(ARTIFACTS_DIR / f"{name}_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"[{name}] events={report['fault_events']} pos_hours={report['positive_hours']} "
          f"full_lift={report['variants']['full']['mean_lift']} sanitized_lift={san['mean_lift']} "
          f"deployable={report['deployable']}")
    for w in report["warnings"]:
        print(f"[{name}] WARNING: {w}")
    return report


def train_mesh_fault_detector(ml_hourly_mesh, network_faults):
    df = build_batch_mesh_fault_features(ml_hourly_mesh)
    labels = build_concurrent_mesh_labels(df["link_id"].unique(), df["timestamp"], network_faults)
    df = df.merge(labels, on=["link_id", "timestamp"], how="left")
    return _run_detector_study(df, "link_id", MESH_FEATURES_NUMERIC_ALL, MESH_FEATURES_NUMERIC_SANITIZED,
                                [], "is_currently_faulted", "mesh_fault")


def train_core_fault_detector(ml_hourly_core, network_faults):
    df = build_batch_core_fault_features(ml_hourly_core)
    labels = build_concurrent_core_labels(df["core_id"].unique(), df["timestamp"], network_faults)
    df = df.merge(labels, on=["core_id", "timestamp"], how="left")

    n_events = count_fault_events(df, "core_id", "is_currently_faulted")
    if n_events < MIN_FAULT_EVENTS_FOR_MODEL:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        report = {"name": "core_fault", "modeled": False, "fault_events": n_events,
                  "positive_hours": int(df["is_currently_faulted"].sum()),
                  "reason": f"{n_events} fault episodes < {MIN_FAULT_EVENTS_FOR_MODEL}: no held-out evaluation "
                            "is statistically meaningful. Descriptive separation only.",
                  "separation_behavioral": cohens_d_table(df, CORE_FEATURES_NUMERIC_SANITIZED, "is_currently_faulted"),
                  "separation_proxy_features": cohens_d_table(df, CORE_PROXY_FEATURES, "is_currently_faulted")}
        with open(ARTIFACTS_DIR / "core_fault_report.json", "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"[core_fault] {report['reason']}")
        return report
    return _run_detector_study(df, "core_id", CORE_FEATURES_NUMERIC_ALL, CORE_FEATURES_NUMERIC_SANITIZED,
                                [], "is_currently_faulted", "core_fault")


def train_congestion_models(ml_congestion_forecast, train_end="2025-03-04", val_end="2025-03-18"):
    df = ml_congestion_forecast.copy()
    df, encoded_cat_cols, categories = encode_categoricals(df, CONGESTION_FEATURES_CATEGORICAL)
    feature_cols = CONGESTION_FEATURES_NUMERIC + encoded_cat_cols

    results = {}
    for horizon_label, horizon_hours in CONGESTION_HORIZONS.items():
        target_col = f"prb_t_plus_{horizon_label}"
        other_targets = [f"prb_t_plus_{h}" for h in CONGESTION_HORIZONS if h != horizon_label]
        clean = df.dropna(subset=[target_col, "prb_lag_168h", "prb_slope_7d"]).drop(columns=other_targets)

        train = clean[clean.timestamp <= train_end]
        train = train[train.timestamp <= train.timestamp.max() - pd.Timedelta(hours=horizon_hours)]
        val = clean[(clean.timestamp > train_end) & (clean.timestamp <= val_end)]
        val = val[val.timestamp <= val.timestamp.max() - pd.Timedelta(hours=horizon_hours)]

        model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(train[feature_cols], train[target_col])

        val_preds = model.predict(val[feature_cols])
        mae = mean_absolute_error(val[target_col], val_preds)
        naive_mae = mean_absolute_error(val[target_col], val["prb_lag_168h"])

        joblib.dump(model, ARTIFACTS_DIR / f"congestion_model_{horizon_label}.joblib")
        results[horizon_label] = {"mae": mae, "naive_mae": naive_mae,
                                   "improvement_pct": (naive_mae - mae) / naive_mae * 100}
        print(f"Congestion {horizon_label} — MAE: {mae:.3f} vs naive MAE: {naive_mae:.3f} "
              f"({results[horizon_label]['improvement_pct']:.1f}% improvement)")

    schema = {"numeric_features": CONGESTION_FEATURES_NUMERIC,
              "categorical_features": CONGESTION_FEATURES_CATEGORICAL,
              "categories": categories, "encoded_feature_order": feature_cols}
    with open(ARTIFACTS_DIR / "congestion_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    with open(ARTIFACTS_DIR / "congestion_metadata.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def build_anomaly_baselines(ml_hourly_ran, metrics=("prb_util_mean", "traffic_mbps_mean", "sinr_mean", "bler_mean"),
                             seasonal_window_weeks=4):
    """Per (cell_id, hour_of_day, day_of_week): trailing seasonal mean/std for each
    metric. Saved as a lookup table the API loads once and queries — no live rolling
    computation needed at serve time for anomaly scoring."""
    df = ml_hourly_ran.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    rows = []
    for (cell_id, hour, dow), group in df.groupby(["cell_id", "hour_of_day", "day_of_week"]):
        group = group.sort_values("timestamp")
        row = {"cell_id": cell_id, "hour_of_day": hour, "day_of_week": dow}
        for m in metrics:
            tail = group[m].tail(seasonal_window_weeks)
            row[f"{m}_expected"] = tail.mean()
            row[f"{m}_std"] = tail.std() if len(tail) >= 2 else np.nan
        rows.append(row)

    baselines = pd.DataFrame(rows)
    baselines.to_parquet(ARTIFACTS_DIR / "anomaly_baselines.parquet", index=False)
    print(f"Saved anomaly baselines: {baselines.shape}")
    return baselines
