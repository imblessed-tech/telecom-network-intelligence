import numpy as np
import pandas as pd
from ml_pipeline.config import (
    LAG_HOURS, ROLLING_WINDOW_HOURS, ALARM_TRAILING_WINDOW_HOURS, NO_ALARM_SENTINEL_HOURS,
    CONGESTION_LAG_HOURS, CONGESTION_ROLLING_24H, CONGESTION_ROLLING_7D, CONGESTION_SLOPE_14D,
)

def rolling_slope(values):
    """Linear trend of a trailing window. Returns NaN if fewer than 2 finite points."""
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(len(y))[valid]
    return np.polyfit(x, y[valid], 1)[0]


def _hours_since_last_alarm(alarm_counts):
    """Sentinel-based: 'never alarmed' -> NO_ALARM_SENTINEL_HOURS, not 0. Conflating the
    two (as an earlier version of this pipeline did) makes 'just alarmed' and 'never
    alarmed' numerically identical, corrupting any recency-based feature."""
    result = np.full(len(alarm_counts), float(NO_ALARM_SENTINEL_HOURS))
    last_idx = None
    for i, c in enumerate(alarm_counts):
        if c > 0:
            last_idx = i
        elif last_idx is not None:
            result[i] = i - last_idx
    return result


def build_batch_cell_fault_features(hourly_ran_df):
    """Vectorized, per-cell-group feature builder for the historical training table.
    hourly_ran_df must have one row per (cell_id, timestamp), sorted or not (sorted here).
    Expects columns: cell_id, timestamp, prb_util_mean, sinr_mean, bler_mean, drop_rate_mean,
    availability_mean, handover_success_mean, traffic_mbps_mean, alarm_count,
    technology, band, site_type, region_name, cell_capacity_mbps.
    """
    df = hourly_ran_df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("cell_id")

    for lag in LAG_HOURS:
        df[f"prb_lag_{lag}h"] = g["prb_util_mean"].shift(lag)
    df["sinr_lag_1h"] = g["sinr_mean"].shift(1)
    df["bler_lag_1h"] = g["bler_mean"].shift(1)
    df["drop_rate_lag_1h"] = g["drop_rate_mean"].shift(1)

    w = ROLLING_WINDOW_HOURS
    df["prb_mean_4h"] = g["prb_util_mean"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["prb_std_4h"] = g["prb_util_mean"].transform(lambda s: s.rolling(w, min_periods=2).std())
    df["sinr_mean_4h"] = g["sinr_mean"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["sinr_slope_4h"] = g["sinr_mean"].transform(lambda s: s.rolling(w, min_periods=2).apply(rolling_slope, raw=True))
    df["bler_mean_4h"] = g["bler_mean"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["drop_rate_mean_4h"] = g["drop_rate_mean"].transform(lambda s: s.rolling(w, min_periods=1).mean())

    df["prb_change_1h"] = df["prb_util_mean"] - df["prb_lag_1h"]
    df["sinr_change_4h"] = df["sinr_mean"] - g["sinr_mean"].shift(4)
    df["bler_change_4h"] = df["bler_mean"] - g["bler_mean"].shift(4)

    df["alarm_count_trailing_24h"] = g["alarm_count"].transform(
        lambda s: s.rolling(ALARM_TRAILING_WINDOW_HOURS, min_periods=1).sum())
    df["hours_since_last_alarm"] = g["alarm_count"].transform(
        lambda s: pd.Series(_hours_since_last_alarm(s.to_numpy()), index=s.index))

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def compute_live_cell_fault_features(window_df: pd.DataFrame, static_attrs: dict) -> dict:
    """Stateless single-entity feature computation for the API. window_df: hourly rows for
    ONE cell, sorted ascending by timestamp, most recent row last. Returns a flat dict of
    features for the LATEST timestamp only — this is what gets scored.
    Requires at least 5 rows (4h of history + current) for full feature coverage; degrades
    gracefully (NaN -> later imputed) with fewer, but a fault decision on <2 rows is unreliable.
    """
    df = window_df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least 2 hourly rows to compute lag/delta features.")

    latest = df.iloc[-1]
    w = ROLLING_WINDOW_HOURS

    def lag(col, n):
        idx = len(df) - 1 - n
        return df[col].iloc[idx] if idx >= 0 else np.nan

    def roll_mean(col, window):
        return df[col].tail(window).mean()

    def roll_std(col, window):
        tail = df[col].tail(window)
        return tail.std() if len(tail) >= 2 else np.nan

    def slope(col, window):
        return rolling_slope(df[col].tail(window).to_numpy())

    features = {
        "sinr_mean": latest["sinr_mean"], "bler_mean": latest["bler_mean"],
        "drop_rate_mean": latest["drop_rate_mean"], "prb_util_mean": latest["prb_util_mean"],
        "handover_success_mean": latest["handover_success_mean"],
        "sinr_lag_1h": lag("sinr_mean", 1), "bler_lag_1h": lag("bler_mean", 1),
        "drop_rate_lag_1h": lag("drop_rate_mean", 1),
        "prb_lag_1h": lag("prb_util_mean", 1), "prb_lag_4h": lag("prb_util_mean", 4),
        "prb_mean_4h": roll_mean("prb_util_mean", w), "prb_std_4h": roll_std("prb_util_mean", w),
        "sinr_mean_4h": roll_mean("sinr_mean", w), "sinr_slope_4h": slope("sinr_mean", w),
        "bler_mean_4h": roll_mean("bler_mean", w), "drop_rate_mean_4h": roll_mean("drop_rate_mean", w),
        "prb_change_1h": latest["prb_util_mean"] - lag("prb_util_mean", 1),
        "sinr_change_4h": latest["sinr_mean"] - lag("sinr_mean", 4),
        "bler_change_4h": latest["bler_mean"] - lag("bler_mean", 4),
    }

    alarm_counts = df["alarm_count"].to_numpy()
    features["alarm_count_trailing_24h"] = df["alarm_count"].tail(ALARM_TRAILING_WINDOW_HOURS).sum()
    hsla = _hours_since_last_alarm(alarm_counts)
    features["hours_since_last_alarm"] = float(hsla[-1])

    ts = pd.Timestamp(latest["timestamp"])
    features["hour_of_day"] = ts.hour
    features["day_of_week"] = ts.dayofweek
    features["is_weekend"] = int(ts.dayofweek >= 5)

    features.update({
        "technology": static_attrs["technology"], "band": static_attrs["band"],
        "site_type": static_attrs["site_type"], "cell_capacity_mbps": static_attrs["cell_capacity_mbps"],
    })
    return features


def compute_live_congestion_features(window_df: pd.DataFrame, static_attrs: dict) -> dict:
    """window_df: hourly rows for ONE cell, columns prb_util_mean, traffic_mbps_mean,
    timestamp. Requires up to MIN_CONGESTION_HISTORY_HOURS for the full feature set;
    slope/168h-lag features return NaN gracefully if history is shorter."""
    df = window_df.sort_values("timestamp").reset_index(drop=True)
    latest = df.iloc[-1]

    def lag(col, n):
        idx = len(df) - 1 - n
        return df[col].iloc[idx] if idx >= 0 else np.nan

    def roll_mean(col, window):
        return df[col].tail(window).mean()

    def roll_std(col, window):
        tail = df[col].tail(window)
        return tail.std() if len(tail) >= 2 else np.nan

    def slope(col, window):
        tail = df[col].tail(window).to_numpy()
        return rolling_slope(tail) if len(tail) >= 2 else np.nan

    features = {
        "prb_util_t": latest["prb_util_mean"], "traffic_t": latest["traffic_mbps_mean"],
        "capacity_headroom_current": static_attrs["cell_capacity_mbps"] - latest["traffic_mbps_mean"],
        "prb_lag_1h": lag("prb_util_mean", 1), "prb_lag_2h": lag("prb_util_mean", 2),
        "prb_lag_24h": lag("prb_util_mean", 24), "prb_lag_168h": lag("prb_util_mean", 168),
        "traffic_lag_1h": lag("traffic_mbps_mean", 1), "traffic_lag_24h": lag("traffic_mbps_mean", 24),
        "prb_mean_24h": roll_mean("prb_util_mean", CONGESTION_ROLLING_24H),
        "prb_std_24h": roll_std("prb_util_mean", CONGESTION_ROLLING_24H),
        "prb_mean_7d": roll_mean("prb_util_mean", CONGESTION_ROLLING_7D),
        "prb_slope_7d": slope("prb_util_mean", CONGESTION_SLOPE_14D),
        "traffic_mean_24h": roll_mean("traffic_mbps_mean", CONGESTION_ROLLING_24H),
        "traffic_slope_7d": slope("traffic_mbps_mean", CONGESTION_ROLLING_7D),
    }
    ts = pd.Timestamp(latest["timestamp"])
    features["hour_of_day"] = ts.hour
    features["day_of_week"] = ts.dayofweek
    features["is_weekend"] = int(ts.dayofweek >= 5)
    features.update({
        "technology": static_attrs["technology"], "region_name": static_attrs["region_name"],
        "site_type": static_attrs["site_type"], "cell_capacity_mbps": static_attrs["cell_capacity_mbps"],
    })
    return features

def _mesh_link_id(df):
    a, b = df["source_agg_id"], df["target_agg_id"]
    lo, hi = a.where(a <= b, b), b.where(a <= b, a)
    return lo + "|" + hi


def build_batch_mesh_fault_features(ml_hourly_mesh):
    df = ml_hourly_mesh.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["link_id"] = _mesh_link_id(df)
    df = df.sort_values(["link_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("link_id")
    w = ROLLING_WINDOW_HOURS

    df["util_lag_1h"] = g["utilization_mean"].shift(1)
    df["util_mean_4h"] = g["utilization_mean"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["util_max_4h"] = g["utilization_max"].transform(lambda s: s.rolling(w, min_periods=1).max())
    df["util_change_1h"] = df["utilization_mean"] - df["util_lag_1h"]
    for src, dst in [("latency_mean_ms", "latency_mean_4h"), ("jitter_mean_ms", "jitter_mean_4h"),
                     ("packet_loss_mean_pct", "packet_loss_mean_4h")]:
        df[dst] = g[src].transform(lambda s: s.rolling(w, min_periods=1).mean())
    return df


def build_batch_core_fault_features(ml_hourly_core):
    df = ml_hourly_core.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["core_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("core_id")
    w = ROLLING_WINDOW_HOURS

    df["cpu_lag_1h"] = g["cpu_mean_pct"].shift(1)
    df["cpu_mean_4h"] = g["cpu_mean_pct"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["cpu_std_4h"] = g["cpu_mean_pct"].transform(lambda s: s.rolling(w, min_periods=2).std())
    df["cpu_change_1h"] = df["cpu_mean_pct"] - df["cpu_lag_1h"]
    df["memory_mean_4h"] = g["memory_mean_pct"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["session_failure_mean_4h"] = g["session_failure_rate_mean"].transform(
        lambda s: s.rolling(w, min_periods=1).mean())
    return df

def create_ml_congestion_forecast(ml_hourly_ran, watch_band_cell_ids=None):
    df = ml_hourly_ran[["cell_id", "timestamp", "prb_util_mean", "traffic_mbps_mean",
                         "technology", "region_name", "site_type", "cell_capacity_mbps"]].copy()
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("cell_id")

    # ---- Lags ----
    df["prb_lag_1h"] = g["prb_util_mean"].shift(1)
    df["prb_lag_2h"] = g["prb_util_mean"].shift(2)
    df["prb_lag_24h"] = g["prb_util_mean"].shift(24)
    df["prb_lag_168h"] = g["prb_util_mean"].shift(168)
    df["traffic_lag_1h"] = g["traffic_mbps_mean"].shift(1)
    df["traffic_lag_24h"] = g["traffic_mbps_mean"].shift(24)

    # ---- Rolling ----
    df["prb_mean_24h"] = g["prb_util_mean"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    df["prb_std_24h"] = g["prb_util_mean"].transform(lambda s: s.rolling(24, min_periods=2).std())
    df["prb_mean_7d"] = g["prb_util_mean"].transform(lambda s: s.rolling(24 * 7, min_periods=24).mean())
    df["prb_slope_7d"] = g["prb_util_mean"].transform(
        lambda s: s.rolling(24 * 14, min_periods=48).apply(rolling_slope, raw=True))  # 14-day window per spec
    df["traffic_mean_24h"] = g["traffic_mbps_mean"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    df["traffic_slope_7d"] = g["traffic_mbps_mean"].transform(
        lambda s: s.rolling(24 * 7, min_periods=24).apply(rolling_slope, raw=True))

    # ---- Current state ----
    df["prb_util_t"] = df["prb_util_mean"]
    df["traffic_t"] = df["traffic_mbps_mean"]
    df["capacity_headroom_current"] = df["cell_capacity_mbps"] - df["traffic_t"]

    # ---- Calendar ----
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ---- Static flag ----
    if watch_band_cell_ids is not None:
        df["is_watch_band_cell"] = df["cell_id"].isin(watch_band_cell_ids).astype(int)

    # ---- Labels: the cell's own FUTURE value — no hidden table involved ----
    df["prb_t_plus_1h"] = g["prb_util_mean"].shift(-1)
    df["prb_t_plus_6h"] = g["prb_util_mean"].shift(-6)
    df["prb_t_plus_24h"] = g["prb_util_mean"].shift(-24)

    return df

