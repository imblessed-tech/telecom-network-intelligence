import numpy as np
import pandas as pd
from simulator.operational import resolve_fault_scope

HOUR = pd.Timedelta(hours=1)


def make_link_id(a, b):
    """Order-independent id for an undirected mesh link — also parquet-safe (a plain string)."""
    lo, hi = sorted((a, b))
    return f"{lo}|{hi}"


def _label_frame(entity_col, entity_ids, timestamps, windows_by_entity, label_name):
    """An hourly row is faulted if its hour bucket [t, t+1h) overlaps any fault window [start, end]."""
    ts = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(timestamps))))
    frames = []
    for eid in entity_ids:
        flag = np.zeros(len(ts), dtype=int)
        for start, end in windows_by_entity.get(eid, []):
            flag |= ((ts <= end) & (ts + HOUR > start)).astype(int)
        frames.append(pd.DataFrame({entity_col: eid, "timestamp": ts, label_name: flag}))
    return pd.concat(frames, ignore_index=True)


def build_concurrent_cell_labels(cell_ids, timestamps, network_faults, sites, cells, aggregations,
                                  core_deps_df, mesh_links, label_name="is_currently_faulted"):
    windows = {c: [] for c in cell_ids}
    for f in network_faults.itertuples():
        scope = resolve_fault_scope(f, sites, cells, aggregations, core_deps_df, mesh_links)
        for cid in scope.get("cells", []):
            if cid in windows:
                windows[cid].append((f.start_time, f.end_time))
    return _label_frame("cell_id", cell_ids, timestamps, windows, label_name)


def build_concurrent_mesh_labels(link_ids, timestamps, network_faults, label_name="is_currently_faulted"):
    """Only transport_link_failure faults act on mesh-link telemetry in the generator."""
    windows = {l: [] for l in link_ids}
    link_faults = network_faults[network_faults.fault_type == "transport_link_failure"]
    for f in link_faults.itertuples():
        lid = make_link_id(f.link_source_id, f.link_target_id)
        if lid in windows:
            windows[lid].append((f.start_time, f.end_time))
    return _label_frame("link_id", link_ids, timestamps, windows, label_name)


def build_concurrent_core_labels(core_ids, timestamps, network_faults, label_name="is_currently_faulted"):
    windows = {c: [] for c in core_ids}
    core_faults = network_faults[network_faults.fault_type == "core_degradation"]
    for f in core_faults.itertuples():
        if f.entity_id in windows:
            windows[f.entity_id].append((f.start_time, f.end_time))
    return _label_frame("core_id", core_ids, timestamps, windows, label_name)


def count_fault_events(df, entity_col, label_col):
    """Distinct fault episodes = contiguous runs of 1s per entity (not positive HOURS — the
    number that actually limits how trustworthy any evaluation can be)."""
    total = 0
    for _, g in df.sort_values("timestamp").groupby(entity_col):
        flags = g[label_col].to_numpy()
        if len(flags) == 0:
            continue
        total += int(flags[0] == 1) + int(((flags[1:] == 1) & (flags[:-1] == 0)).sum())
    return total


def build_fault_labels_for_cells(cell_ids, timestamps, network_faults, sites, cells, aggregations,
                                  core_deps_df, mesh_links, horizon_hours=4):
    """Returns a DataFrame keyed by (cell_id, timestamp) with will_direct_fault_next_4h,
    will_cascaded_fault_next_4h, will_any_fault_next_4h, fault_type_if_any."""
    direct_types = {"ran_degradation", "cell_outage"}
    cascade_types = {"aggregation_failure", "core_degradation", "backhaul_degradation"}

    direct_starts = {c: [] for c in cell_ids}
    cascade_starts = {c: [] for c in cell_ids}
    fault_type_by_cell_time = {}

    for f in network_faults.itertuples():
        scope = resolve_fault_scope(f, sites, cells, aggregations, core_deps_df, mesh_links)
        affected_cells = scope.get("cells", [])
        for cid in affected_cells:
            if cid not in direct_starts:
                continue
            if f.fault_type in direct_types:
                direct_starts[cid].append((f.start_time, f.fault_type))
            elif f.fault_type in cascade_types:
                cascade_starts[cid].append((f.start_time, f.fault_type))

    unique_ts = sorted(pd.unique(timestamps))
    rows = []
    for cid in cell_ids:
        d_events = direct_starts[cid]
        c_events = cascade_starts[cid]
        for t in unique_ts:
            window_end = t + pd.Timedelta(hours=horizon_hours)
            d_hit = [ft for s, ft in d_events if t < s <= window_end]
            c_hit = [ft for s, ft in c_events if t < s <= window_end]
            will_direct = int(len(d_hit) > 0)
            will_cascaded = int(len(c_hit) > 0)
            rows.append({
                "cell_id": cid, "timestamp": t,
                "will_direct_fault_next_4h": will_direct,
                "will_cascaded_fault_next_4h": will_cascaded,
                "will_any_fault_next_4h": int(will_direct or will_cascaded),
                "fault_type_if_any": (d_hit + c_hit)[0] if (d_hit or c_hit) else None,
            })
    return pd.DataFrame(rows)