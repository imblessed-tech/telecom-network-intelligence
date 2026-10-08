import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from simulator.topology import compute_design_load

TIME_CONFIG = {"n_days": 90, "interval_minutes": 15}
N_STEPS = TIME_CONFIG["n_days"] * 24 * 60 // TIME_CONFIG["interval_minutes"]
SEVERITY_LEVELS = {"low": 0.30, "medium": 0.60, "high": 1.00}
SEVERITY_WEIGHTS = [0.50, 0.35, 0.15]
FAULT_TYPE_CONFIG = {
    "ran_degradation":       {"weight": 0.30, "entity_type": "cell",        "duration_hr": (1, 6)},
    "cell_outage":           {"weight": 0.15, "entity_type": "site",        "duration_hr": (0.5, 4)},
    "backhaul_degradation":  {"weight": 0.20, "entity_type": "site",        "duration_hr": (1, 8)},
    "transport_link_failure":{"weight": 0.15, "entity_type": "mesh_link",   "duration_hr": (1, 12)},
    "aggregation_failure":   {"weight": 0.10, "entity_type": "aggregation", "duration_hr": (2, 10)},
    "core_degradation":      {"weight": 0.10, "entity_type": "core",        "duration_hr": (1, 6)},
}
LINK_PROFILES = {
    "backhaul": (3, 0.5),
    "fiber_metro": (1, 0.2),
    "fiber_regional": (2, 0.4),
    "microwave": (4, 1.2),
    "core_uplink": (2, 0.3),
}
RADIO_BASELINE_SINR = {
    ("4G", "urban"): 12, ("4G", "suburban"): 15, ("4G", "rural"): 18,
    ("5G", "urban"): 18, ("5G", "suburban"): 22, ("5G", "rural"): 25,
}
MBPS_PER_ACTIVE_USER = {"4G": 3.0, "5G": 8.0}
ALARM_RULES = [
    ("ran", "prb_utilization_pct", ">", 90, "<", 75, 4, "high_prb_utilization"),
    ("ran", "availability_pct", "<", 50, ">", 90, 1, "cell_unavailable"),
    ("transport", "packet_loss_pct", ">", 5, "<", 2, 4, "packet_loss_threshold"),
    ("transport", "latency_ms", ">", 15, "<", 8, 4, "latency_threshold"),
    ("transport", "link_status", "==", "DOWN", "!=", "DOWN", 1, "link_down"),
    ("core", "cpu_utilization_pct", ">", 85, "<", 70, 4, "core_resource_exhaustion"),
]
FALSE_ALARM_RATE = 0.03
SITE_TYPE_MULTIPLIER = {"urban": 1.15, "suburban": 1.00, "rural": 0.70}
TECH_MULTIPLIER = {"4G": 1.00, "5G": 1.15}
REGION_FACTOR_VALUES = {"very_low": 0.70, "low": 0.80, "low_medium": 0.90, "medium": 1.00,
                         "medium_high": 1.10, "high": 1.20, "very_high": 1.35}
TRANSPORT_MEDIUM_PROFILES = [(15, "fiber_metro", 20.0), (60, "fiber_regional", 10.0), (np.inf, "microwave", 2.5)]
REGION_PEAK_SHIFT_HR = {"Lagos": 0.5, "West": 0.0, "North": -0.5, "South": 0.2, "East": -0.3}
SITE_TYPE_PEAK_WIDTH = {"urban": 1.6, "suburban": 2.2, "rural": 3.0}
WEEKDAY_MULT, WEEKEND_MULT = 1.0, 0.86
TREND_GROWTH_OVER_90D = 0.05
DEMAND_NOISE_STD = 0.03
time_index = pd.date_range(start="2025-01-01", periods=N_STEPS, freq=f"{TIME_CONFIG['interval_minutes']}min")
hours = time_index.hour + time_index.minute / 60
day_index = (time_index - time_index[0]).days
is_weekend = time_index.dayofweek.isin([5, 6])

def build_time_index(start="2025-01-01", n_steps=N_STEPS, interval_minutes=15):
    return pd.date_range(start=start, periods=n_steps, freq=f"{interval_minutes}min")

def diurnal_curve(hours, region_name, site_type):
    """Two Gaussian bumps (morning ~9h, evening ~19h) + a night floor. Region shifts
    peak timing slightly; site_type controls how sharp/broad the peak is (urban commute
    peaks are sharp, rural demand is flatter across the day). Normalized so max == 1.0."""
    shift = REGION_PEAK_SHIFT_HR[region_name]
    width = SITE_TYPE_PEAK_WIDTH[site_type]
    morning = np.exp(-0.5 * ((hours - (9 + shift)) / width) ** 2)
    evening = np.exp(-0.5 * ((hours - (19 + shift)) / (width * 1.1)) ** 2) * 1.15  # evening slightly taller
    night_floor = 0.12
    curve = night_floor + (1 - night_floor) * np.maximum(morning, evening)
    return curve / curve.max()

def generate_traffic_demand(regions, sites, cells, rng):
    region_lookup = regions.set_index("region_id")["region_name"]
    site_lookup = sites.set_index("site_id")[["region_id", "site_type"]]
    trend = 1 + TREND_GROWTH_OVER_90D * (day_index / (TIME_CONFIG["n_days"] - 1))
    weekday_mult = np.where(is_weekend, WEEKEND_MULT, WEEKDAY_MULT)

    rows = []
    for cell in cells.itertuples():
        region_name = region_lookup[site_lookup.loc[cell.site_id, "region_id"]]
        site_type = site_lookup.loc[cell.site_id, "site_type"]

        curve = diurnal_curve(hours, region_name, site_type)
        noise = rng.normal(1.0, DEMAND_NOISE_STD, N_STEPS).clip(0.85, 1.15)

        offered_load_mbps = (cell.cell_capacity_mbps * cell.target_utilization
                              * curve * weekday_mult * trend * noise)
        active_users = np.round(offered_load_mbps / MBPS_PER_ACTIVE_USER[cell.technology]).astype(int)

        rows.append(pd.DataFrame({
            "timestamp": time_index, "cell_id": cell.cell_id,
            "offered_load_mbps": offered_load_mbps.round(2), "active_users": active_users,
        }))
    return pd.concat(rows, ignore_index=True)

def inject_anomalies(traffic_demand, rng, n_events=25):
    """Short, legitimate-looking spikes with NO underlying fault — the anomaly-detection
    ground truth. Kept separate from network_faults so 'anomaly' and 'fault' stay distinct."""
    cells_list = traffic_demand["cell_id"].unique()
    events = []
    td = traffic_demand.set_index(["cell_id", "timestamp"])["offered_load_mbps"].copy()
    for i in range(n_events):
        cid = rng.choice(cells_list)
        start_idx = rng.integers(0, N_STEPS - 8)
        span = slice(start_idx, start_idx + rng.integers(2, 8))
        multiplier = rng.uniform(1.4, 2.2)
        ts_slice = time_index[span]
        for ts in ts_slice:
            td.loc[(cid, ts)] *= multiplier
        events.append({"anomaly_id": f"anom_{i+1:04d}", "cell_id": cid,
                        "start_time": ts_slice[0], "end_time": ts_slice[-1], "multiplier": round(multiplier, 2)})
    traffic_demand = td.reset_index()
    return traffic_demand, pd.DataFrame(events)

def generate_congestion_events(regions, sites, cells, backhaul_links, rng, n_events=8,
                                duration_hours=(2, 4), peak_multiplier=(1.3, 1.7),
                                target_util_band=(0.55, 0.80)):
    """Targets sites whose NORMAL peak utilization already sits in target_util_band —
    close enough to their backhaul ceiling that a moderate surge genuinely pushes them
    over, rather than picking sites at random and hoping the multiplier is enough.
    Also anchors event start times to the site's actual diurnal peak hour, so the
    surge multiplies real peak demand instead of an arbitrary time-of-day baseline."""
    loads = compute_design_load(regions, sites, cells)
    site_demand = loads["site_load"].set_index("site_id")["site_demand_gbps"]
    bh_capacity = backhaul_links.set_index("source_id")["capacity"]
    design_util = (site_demand / bh_capacity).dropna()

    candidates = design_util[(design_util >= target_util_band[0]) & (design_util <= target_util_band[1])]
    if len(candidates) < n_events:
        print(f"Only {len(candidates)} sites fall in the target utilization band "
              f"{target_util_band} — widening band or reducing n_events may be needed.")
    site_ids = rng.choice(candidates.index, size=min(n_events, len(candidates)), replace=False)

    events = []
    for i, site_id in enumerate(site_ids):
        # anchor to a plausible peak window (late morning or evening) rather than any random step
        peak_hour = rng.choice([8, 9, 18, 19])
        day = rng.integers(0, TIME_CONFIG["n_days"] - 1)
        start = time_index[0] + pd.Timedelta(days=int(day), hours=int(peak_hour))
        duration_steps = int(rng.uniform(*duration_hours) * 4)
        end = start + pd.Timedelta(minutes=15 * duration_steps)
        multiplier = round(rng.uniform(*peak_multiplier), 2)
        events.append({"congestion_id": f"cong_{i+1:04d}", "site_id": site_id,
                       "start_time": start, "end_time": end, "peak_multiplier": multiplier,
                       "design_util_baseline": round(candidates[site_id], 2)})
    return pd.DataFrame(events)

def apply_congestion_to_demand(traffic_demand, congestion_events, cells, ramp_frac=0.2):
    """Applies each event's multiplier to ALL cells at the targeted site, shaped with the
    same ramp-plateau-ramp trapezoid used for faults — a real surge builds and fades,
    it doesn't switch on/off instantly."""
    traffic_demand = traffic_demand.copy()
    site_cells = cells.groupby("site_id")["cell_id"].apply(list)

    for ev in congestion_events.itertuples():
        affected_cells = site_cells.get(ev.site_id, [])
        intensity = fault_intensity(ev.start_time, ev.end_time, ramp_frac=ramp_frac)  # reuse existing trapezoid shape
        multiplier_curve = 1 + (ev.peak_multiplier - 1) * intensity  # 1.0 baseline -> peak -> back to 1.0

        mask = traffic_demand["cell_id"].isin(affected_cells)
        # align multiplier_curve (indexed by time_index position) to each affected row's timestamp
        ts_to_mult = pd.Series(multiplier_curve, index=time_index)
        traffic_demand.loc[mask, "offered_load_mbps"] = (
            traffic_demand.loc[mask, "offered_load_mbps"].to_numpy()
            * traffic_demand.loc[mask, "timestamp"].map(ts_to_mult).to_numpy()
        )
    return traffic_demand

def build_neighbor_map(cells, sites, k=3):
    """Cells inherit their site's coordinates. Nearest OTHER cells by straight-line
    distance, excluding cells at the same site (a cell doesn't hand over to its own
    sibling sector in this simplified model)."""
    merged = cells.merge(sites[["site_id", "latitude", "longitude"]], on="site_id")
    coords = merged[["latitude", "longitude"]].to_numpy()
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 10)  # over-fetch, then filter same-site

    neighbor_map = {}
    for i, cell_id in enumerate(merged["cell_id"]):
        own_site = merged.iloc[i]["site_id"]
        candidates = [merged.iloc[j]["cell_id"] for j in idx[i] if merged.iloc[j]["site_id"] != own_site]
        neighbor_map[cell_id] = candidates[:k]
    return neighbor_map

def generate_network_faults(cells, sites, aggregations, core_nodes, mesh_links, rng, n_incidents=50):
    types = list(FAULT_TYPE_CONFIG.keys())
    weights = [FAULT_TYPE_CONFIG[t]["weight"] for t in types]
    entity_pools = {
        "cell": cells["cell_id"].tolist(), "site": sites["site_id"].tolist(),
        "aggregation": aggregations["aggregation_id"].tolist(), "core": core_nodes["core_id"].tolist(),
        "mesh_link": list(zip(mesh_links["source_id"], mesh_links["target_id"])),
    }
    rows = []
    for i in range(n_incidents):
        ftype = rng.choice(types, p=weights)
        cfg = FAULT_TYPE_CONFIG[ftype]
        entity = entity_pools[cfg["entity_type"]][rng.integers(len(entity_pools[cfg["entity_type"]]))]
        severity = rng.choice(list(SEVERITY_LEVELS), p=SEVERITY_WEIGHTS)

        start = time_index[rng.integers(0, N_STEPS - 1)]
        duration_hr = rng.uniform(*cfg["duration_hr"])
        end = start + pd.Timedelta(hours=duration_hr)
        detect_delay_min = rng.uniform(5, 45) / SEVERITY_LEVELS[severity]
        detection = min(start + pd.Timedelta(minutes=detect_delay_min), end)

        # flat, parquet-safe representation: single-entity faults use entity_id,
        # link faults use entity_id_source/entity_id_target instead of a tuple.
        if cfg["entity_type"] == "mesh_link":
            entity_id, link_source, link_target = None, entity[0], entity[1]
        else:
            entity_id, link_source, link_target = entity, None, None

        rows.append({
            "fault_id": f"fault_{i+1:04d}", "fault_type": ftype, "entity_type": cfg["entity_type"],
            "entity_id": entity_id, "link_source_id": link_source, "link_target_id": link_target,
            "severity": severity, "start_time": start, "detection_time": detection, "end_time": end,
        })
    return pd.DataFrame(rows)

def fault_intensity(start, end, ramp_frac=0.15):
    """Trapezoid: linear ramp up, plateau, linear ramp down, all within [0,1]."""
    duration = (end - start) / pd.Timedelta(minutes=TIME_CONFIG["interval_minutes"])
    ramp_steps = max(1, int(duration * ramp_frac))
    mask = (time_index >= start) & (time_index <= end)
    intensity = np.zeros(N_STEPS)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return intensity
    n = len(idx)
    ramp_steps = min(ramp_steps, n // 2 if n > 1 else 1)
    shape = np.ones(n)
    shape[:ramp_steps] = np.linspace(0, 1, ramp_steps)
    shape[-ramp_steps:] = np.linspace(1, 0, ramp_steps)
    intensity[idx] = shape
    return intensity

def resolve_fault_scope(fault, sites, cells, aggregations, core_deps_df, mesh_links):
    """Returns which cells/sites/aggregations/cores/links are affected — driven purely
    by topology, never by random selection."""
    
    scope = {"cells": [], "sites": [], "aggregations": [], "cores": [], "mesh_links": []}
    t, e = fault.fault_type, fault.entity_id

    if t == "ran_degradation":
        scope["cells"] = [e]
    elif t == "cell_outage":
        scope["sites"] = [e]
        scope["cells"] = cells.loc[cells.site_id == e, "cell_id"].tolist()
    elif t == "backhaul_degradation":
        scope["sites"] = [e]
        scope["cells"] = cells.loc[cells.site_id == e, "cell_id"].tolist()
    elif t == "transport_link_failure":
        scope["mesh_links"] = [(fault.link_source_id, fault.link_target_id)]  # (source_agg, target_agg) tuple
    elif t == "aggregation_failure":
        scope["aggregations"] = [e]
        dep_sites = sites.loc[sites.aggregation_id == e, "site_id"].tolist()
        scope["sites"] = dep_sites
        scope["cells"] = cells.loc[cells.site_id.isin(dep_sites), "cell_id"].tolist()
    elif t == "core_degradation":
        scope["cores"] = [e]
        dep_aggs = core_deps_df.loc[(core_deps_df.target_id == e) & (core_deps_df.dependency_role == "primary"),
                                     "source_id"].tolist()
        scope["aggregations"] = dep_aggs
        dep_sites = sites.loc[sites.aggregation_id.isin(dep_aggs), "site_id"].tolist()
        scope["sites"] = dep_sites
        scope["cells"] = cells.loc[cells.site_id.isin(dep_sites), "cell_id"].tolist()
    return scope

def generate_ran_telemetry(cells, sites, traffic_demand, faults, neighbor_map, sites_meta, agg_meta, core_meta, rng):
    demand_by_cell = {cid: g.set_index("timestamp")["offered_load_mbps"].reindex(time_index).to_numpy()
                       for cid, g in traffic_demand.groupby("cell_id")}

    ran_intensity, outage_intensity, backhaul_derate = {}, {}, {}
    for c in cells.cell_id:
        ran_intensity[c] = np.zeros(N_STEPS); outage_intensity[c] = np.zeros(N_STEPS); backhaul_derate[c] = np.ones(N_STEPS)

    for f in faults.itertuples():
        scope = resolve_fault_scope(f, sites, cells, agg_meta, core_meta, None)
        intensity = fault_intensity(f.start_time, f.end_time) * SEVERITY_LEVELS[f.severity]
        for cid in scope["cells"]:
            if f.fault_type == "ran_degradation":
                ran_intensity[cid] = np.maximum(ran_intensity[cid], intensity)
            elif f.fault_type in ("cell_outage", "aggregation_failure"):
                outage_intensity[cid] = np.maximum(outage_intensity[cid], intensity)
            elif f.fault_type in ("backhaul_degradation", "core_degradation"):
                backhaul_derate[cid] = np.minimum(backhaul_derate[cid], 1 - 0.6 * intensity)

    # ---- PASS 1: intrinsic per-cell quality (no handover yet) ----
    intrinsic = {}
    for cell in cells.itertuples():
        site_type = sites.set_index("site_id").loc[cell.site_id, "site_type"]
        demand = demand_by_cell[cell.cell_id]
        capacity = cell.cell_capacity_mbps
        avail = 1 - outage_intensity[cell.cell_id]

        load_ratio = demand / capacity
        prb_util = np.clip(load_ratio, 0, 1) * 100
        congestion_penalty = 1 / (1 + np.clip(load_ratio - 1, 0, None) * 1.5)
        throughput = np.minimum(demand, capacity) * congestion_penalty * backhaul_derate[cell.cell_id] * avail

        # base_sinr = RADIO_BASELINE_SINR[(cell.technology, site_type)]
        # sinr = base_sinr - 12 * ran_intensity[cell.cell_id] + rng.normal(0, 0.8, N_STEPS)
        # rsrp = -80 - 15 * ran_intensity[cell.cell_id] + rng.normal(0, 2, N_STEPS)
        # rsrq = -8 - 5 * ran_intensity[cell.cell_id] + rng.normal(0, 0.5, N_STEPS)

        base_sinr = RADIO_BASELINE_SINR[(cell.technology, site_type)]
        sinr = base_sinr + cell.quality_offset_db - 12 * ran_intensity[cell.cell_id] + rng.normal(0, 0.8, N_STEPS)
        rsrp = -80 + cell.quality_offset_db - 15 * ran_intensity[cell.cell_id] + rng.normal(0, 2, N_STEPS)
        rsrq = -8 + cell.quality_offset_db * 0.5 - 5 * ran_intensity[cell.cell_id] + rng.normal(0, 0.5, N_STEPS)

        bler = np.clip(2 + 6 * ran_intensity[cell.cell_id] + rng.normal(0, 0.5, N_STEPS), 0, 100)
        harq_retx = np.clip(bler * 0.8 + rng.normal(0, 0.5, N_STEPS), 0, 100)

        congestion_pressure = np.clip(load_ratio - 0.85, 0, None)
        rrc_success = np.clip(99.5 - 20 * congestion_pressure - 15 * ran_intensity[cell.cell_id], 50, 100)
        erab_success = np.clip(99.0 - 22 * congestion_pressure - 18 * ran_intensity[cell.cell_id], 45, 100)
        drop_rate = np.clip(0.3 + 8 * ran_intensity[cell.cell_id] + 4 * congestion_pressure, 0, 100)
        availability_pct = avail * 100

        intrinsic[cell.cell_id] = dict(
            technology=cell.technology, demand=demand, throughput=throughput, sinr=sinr, rsrp=rsrp, rsrq=rsrq,
            bler=bler, harq_retx=harq_retx, prb_util=prb_util, rrc_success=rrc_success, erab_success=erab_success,
            drop_rate=drop_rate, availability_pct=availability_pct, congestion_pressure=congestion_pressure,
            ran_intensity=ran_intensity[cell.cell_id],
        )

    # ---- PASS 2: handover, now genuinely using neighbor SINR ----
    rows = []
    for cell_id, m in intrinsic.items():
        neighbors = neighbor_map.get(cell_id, [])
        if neighbors:
            neighbor_sinr = np.mean([intrinsic[n]["sinr"] for n in neighbors if n in intrinsic], axis=0)
        else:
            neighbor_sinr = m["sinr"]  # no neighbors found -> fall back to own quality, not a magic constant

        sinr_gap = np.clip(m["sinr"] - neighbor_sinr, -10, 10)  # own better than neighbor -> less reason to fail HO
        ho_success = np.clip(
            97 - 20 * m["ran_intensity"] - 12 * m["congestion_pressure"] - 4 * np.clip(-sinr_gap, 0, None),
            40, 100
        )

        rows.append(pd.DataFrame({
            "timestamp": time_index, "cell_id": cell_id,
            "active_users": (m["demand"] / MBPS_PER_ACTIVE_USER[m["technology"]]).round().astype(int),
            "dl_throughput_mbps": (m["throughput"] * 0.7).round(2), "ul_throughput_mbps": (m["throughput"] * 0.3).round(2),
            "prb_utilization_pct": m["prb_util"].round(1), "rsrp_dbm": m["rsrp"].round(1), "rsrq_db": m["rsrq"].round(1),
            "sinr_db": m["sinr"].round(1), "bler_pct": m["bler"].round(2), "harq_retx_pct": m["harq_retx"].round(2),
            "rrc_setup_success_pct": m["rrc_success"].round(2), "erab_setup_success_pct": m["erab_success"].round(2),
            "drop_rate_pct": m["drop_rate"].round(2), "handover_success_pct": ho_success.round(2),
            "availability_pct": m["availability_pct"].round(1),
        }))
    return pd.concat(rows, ignore_index=True)

def compute_hierarchy_loads(cells, ran_telemetry, sites, aggregations, core_deps_df, faults):
    served = ran_telemetry.assign(served_mbps=lambda d: d.dl_throughput_mbps + d.ul_throughput_mbps)
    served = served.merge(sites[["site_id", "aggregation_id"]], left_on="cell_id", right_on="cell_id", how="left") \
        if "site_id" in served.columns else served.merge(
            sites.merge(cells[["cell_id", "site_id"]], on="site_id")[["cell_id", "site_id", "aggregation_id"]],
            on="cell_id")

    site_series = served.pivot_table(index="timestamp", columns="site_id", values="served_mbps", aggfunc="sum") \
                          .reindex(time_index).fillna(0) / 1000  # Gbps

    site_to_agg = sites.set_index("site_id")["aggregation_id"]
    agg_series = site_series.T.groupby(site_to_agg).sum().T

    primary = core_deps_df[core_deps_df.dependency_role == "primary"].set_index("source_id")["target_id"]
    core_series = agg_series.T.groupby(primary.reindex(agg_series.columns)).sum().T
    return site_series, agg_series, core_series

def build_link_load_lookup(link_frame, series_by_source):
    """Maps (source_id, target_id) -> that source's traffic time series (Gbps).
    Falls back to an all-zero series if the source isn't found in series_by_source —
    which should never happen for a correctly-wired link, so the assert below is
    there to catch a silent mismatch rather than let it pass unnoticed."""
    lookup = {}
    for link in link_frame.itertuples():
        key = (link.source_id, link.target_id)
        source = series_by_source.get(link.source_id)
        lookup[key] = source.to_numpy() if source is not None else np.zeros(N_STEPS)
    return lookup

def generate_link_telemetry(link_frame, load_series_lookup, faults, rng, link_type_col="link_type"):
    """One row per (timestamp, link). traffic/capacity -> utilization; utilization and
    fault intensity together drive latency/jitter/packet_loss; availability and
    link_status are derived from fault intensity, not independently randomized."""
    rows = []
    for link in link_frame.itertuples():
        key = (link.source_id, link.target_id)
        traffic = load_series_lookup.get(key, np.zeros(N_STEPS))
        capacity = link.capacity
        util = np.clip(traffic / capacity, 0, 3)  # allow >100% to surface as a real overload, not silently cap

        intensity = np.zeros(N_STEPS)
        for f in faults.itertuples():
            #if f.fault_type == "transport_link_failure" and (f.entity_id == key or f.entity_id == key[::-1]):
            if f.fault_type == "transport_link_failure" and (
                    (f.link_source_id, f.link_target_id) == key or (f.link_target_id, f.link_source_id) == key
                    ):
                intensity = np.maximum(
                    intensity,
                    fault_intensity(f.start_time, f.end_time) * SEVERITY_LEVELS[f.severity]
                )

        base_lat, base_jit = LINK_PROFILES[getattr(link, link_type_col)]
        #latency = base_lat + 15 * np.clip(util - 0.8, 0, None) + 40 * intensity + rng.normal(0, 0.3, N_STEPS)
        jitter = base_jit + 3 * np.clip(util - 0.8, 0, None) + 8 * intensity + rng.normal(0, 0.1, N_STEPS)
        latency = base_lat + 15 * np.clip(util - 0.93, 0, None) + 40 * intensity + rng.normal(0, 0.3, N_STEPS)
        packet_loss = np.clip(2 * np.clip(util - 0.97, 0, None) ** 2 * 100 + 40 * intensity, 0, 100)
        #packet_loss = np.clip(2 * np.clip(util - 0.9, 0, None) ** 2 * 100 + 40 * intensity, 0, 100)
        availability = np.clip(100 * (1 - intensity), 0, 100)

        status = np.where(
            availability < 50, "DOWN",
            np.where((packet_loss > 5) | (latency > base_lat * 4), "DEGRADED", "UP")
        )

        rows.append(pd.DataFrame({
            "timestamp": time_index,
            "source_id": link.source_id,
            "target_id": link.target_id,
            "utilization_pct": np.round(util * 100, 1),
            "throughput_gbps": np.round(traffic, 3),
            "latency_ms": np.round(latency, 2),
            "jitter_ms": np.round(jitter, 2),
            "packet_loss_pct": np.round(packet_loss, 2),
            "availability_pct": np.round(availability, 1),
            "link_status": status,
        }))
    return pd.concat(rows, ignore_index=True)

def generate_transport_telemetry(backhaul_links, mesh_links, core_deps_df,
                                  site_series, agg_series, network_faults, transport_links, rng):
    """Builds all three link-layer telemetry sets and returns one combined,
    link_id-tagged transport_telemetry table."""

    # --- sanity checks before trusting the lookups (silent-mismatch guard) ---
    assert set(backhaul_links.source_id) <= set(site_series.columns), \
        "backhaul source_id mismatch with site_series columns"
    assert set(core_deps_df.source_id) <= set(agg_series.columns), \
        "core_uplink source_id mismatch with agg_series columns"

    # --- 1. Backhaul: site -> aggregation ---
    backhaul_lookup = build_link_load_lookup(backhaul_links, site_series)
    backhaul_telemetry = generate_link_telemetry(backhaul_links, backhaul_lookup, network_faults, rng)

    # --- 2. Transport mesh: aggregation <-> aggregation ---
    # Zero baseline: normal traffic goes aggregation -> core directly (see design note),
    # so mesh links normally carry no load. This means packet_loss/latency on a mesh
    # link during a transport_link_failure fault will show the outage via availability/
    # link_status, but NOT via utilization-driven packet_loss (util stays 0). Wiring in
    # actual failover-routed traffic is the flagged next enhancement, not done here.
    mesh_lookup = {(row.source_id, row.target_id): np.zeros(N_STEPS) for row in mesh_links.itertuples()}
    mesh_telemetry = generate_link_telemetry(mesh_links, mesh_lookup, network_faults, rng)

    # --- 3. Core uplink: aggregation -> core ---
    core_lookup = build_link_load_lookup(core_deps_df, agg_series)
    core_uplink_telemetry = generate_link_telemetry(core_deps_df, core_lookup, network_faults, rng)

    transport_telemetry = pd.concat(
        [backhaul_telemetry, mesh_telemetry, core_uplink_telemetry],
        ignore_index=True
    )

    # Reattach link_id from the static transport_links table (source/target/link_type
    # uniquely identify a link there, so this merge should not create duplicate rows —
    # validated below).
    key_cols = ["source_id", "target_id", "link_type"]
    before = len(transport_telemetry)
    transport_telemetry = transport_telemetry.merge(
        transport_links[["link_id"] + key_cols], on=["source_id", "target_id"], how="left"
    )
    assert len(transport_telemetry) == before, \
        "merge created duplicate rows — check transport_links for (source_id, target_id) duplicates"
    assert transport_telemetry["link_id"].notna().all(), \
        "some transport_telemetry rows failed to match a link_id — check link_type_col alignment"

    return transport_telemetry

def generate_core_telemetry(core_nodes, core_series, faults, rng):
    rows = []
    for core in core_nodes.itertuples():
        traffic = core_series.get(core.core_id, pd.Series(0, index=time_index)).to_numpy()
        util = np.clip(traffic / core.capacity, 0, 2)

        intensity = np.zeros(N_STEPS)
        for f in faults.itertuples():
            if f.fault_type == "core_degradation" and f.entity_id == core.core_id:
                intensity = np.maximum(intensity, fault_intensity(f.start_time, f.end_time) * SEVERITY_LEVELS[f.severity])

        # cpu = np.clip(30 + 60 * util + 30 * intensity + rng.normal(0, 3, N_STEPS), 0, 100)
        cpu = np.clip(20 + 55 * util + 30 * intensity + rng.normal(0, 3, N_STEPS), 0, 100)  # was 30 + 60*util
        mem = np.clip(25 + 50 * util + 25 * intensity + rng.normal(0, 3, N_STEPS), 0, 100)
        latency = 1 + 5 * np.clip(util - 0.8, 0, None) + 15 * intensity + rng.normal(0, 0.2, N_STEPS)
        error_rate = np.clip(0.1 + 3 * np.clip(util - 0.9, 0, None) + 10 * intensity, 0, 100)
        session_failure = np.clip(0.2 + 5 * intensity + 2 * np.clip(util - 0.9, 0, None), 0, 100)
        availability = np.clip(100 - 80 * intensity, 0, 100)

        rows.append(pd.DataFrame({
            "timestamp": time_index, "core_id": core.core_id,
            "cpu_utilization_pct": cpu.round(1), "memory_utilization_pct": mem.round(1),
            "throughput_gbps": traffic.round(3), "latency_ms": latency.round(2),
            "error_rate_pct": error_rate.round(2), "session_failure_rate_pct": session_failure.round(2),
            "availability_pct": availability.round(1),
        }))
    return pd.concat(rows, ignore_index=True)

def _apply_op(series, op, value):
    return {">": series > value, "<": series < value,
            "==": series == value, "!=": series != value}[op]

def generate_alarms(ran, transport, core, rng):
    tables = {"ran": (ran, "cell_id"), "transport": (transport, "link_id"), "core": (core, "core_id")}
    rows, alarm_cnt = [], 0

    for table_name, metric, breach_op, breach_th, clear_op, clear_th, persistence, alarm_type in ALARM_RULES:
        df, key = tables[table_name]
        breach = _apply_op(df[metric], breach_op, breach_th)
        clear = _apply_op(df[metric], clear_op, clear_th)
        tagged = df.assign(breach=breach, clear=clear).sort_values("timestamp")

        for entity, g in tagged.groupby(key):
            g = g.sort_values("timestamp")
            fired = fire_with_hysteresis(g["breach"].to_numpy(), g["clear"].to_numpy(), persistence)
            for i in fired:
                alarm_cnt += 1
                raw_value = g[metric].iloc[i]
                rows.append({
                    "alarm_id": f"alarm_{alarm_cnt:05d}", "timestamp": g["timestamp"].iloc[i],
                    "entity_type": table_name, "entity_id": entity, "alarm_type": alarm_type,
                    "metric_value": raw_value if isinstance(raw_value, (int, float, np.floating)) else np.nan,
                    "metric_value_text": str(raw_value) if not isinstance(raw_value, (int, float, np.floating)) else None,
                    "false_alarm": False,
                })

    n_false = int(len(rows) * FALSE_ALARM_RATE / (1 - FALSE_ALARM_RATE))
    for i in range(n_false):
        rows.append({"alarm_id": f"alarm_{alarm_cnt+i+1:05d}", "timestamp": time_index[rng.integers(N_STEPS)],
                     "entity_type": "ran", "entity_id": rng.choice(ran["cell_id"].unique()),
                     "alarm_type": "sensor_spike", "metric_value": np.nan, "metric_value_text": None,
                     "false_alarm": True})
    return pd.DataFrame(rows)

def fire_with_hysteresis(breach_series, clear_series, persistence):
    """breach_series: bool array — condition that counts toward firing.
    clear_series: bool array — condition that must hold before a NEW alarm
    can fire again. Must be a LOOSER condition than breach (e.g. breach at >5,
    clear at <2) or a value oscillating around the boundary refires constantly."""
    fired_indices = []
    run, armed = 0, True
    for i in range(len(breach_series)):
        if breach_series[i] and armed:
            run += 1
            if run == persistence:
                fired_indices.append(i)
                armed = False
                run = 0
        else:
            run = 0
        if not armed and clear_series[i]:
            armed = True
    return fired_indices

def build_evaluation_labels(network_faults, sites, cells, aggregations, core_deps_df, mesh_links):
    """timestamp x cell_id -> fault_active/fault_type/severity. NEVER merge this into
    ran_telemetry/transport_telemetry/core_telemetry feature tables — evaluation-only."""
    label = pd.DataFrame({"timestamp": time_index})
    per_cell = {cid: {"fault_active": np.zeros(N_STEPS, dtype=bool),
                       "fault_type": np.array([None] * N_STEPS, dtype=object),
                       "fault_severity": np.array([None] * N_STEPS, dtype=object)}
                for cid in cells.cell_id}

    for f in network_faults.itertuples():
        scope = resolve_fault_scope(f, sites, cells, aggregations, core_deps_df, mesh_links)
        mask = (time_index >= f.start_time) & (time_index <= f.end_time)
        for cid in scope["cells"]:
            per_cell[cid]["fault_active"] |= mask
            per_cell[cid]["fault_type"] = np.where(mask, f.fault_type, per_cell[cid]["fault_type"])
            per_cell[cid]["fault_severity"] = np.where(mask, f.severity, per_cell[cid]["fault_severity"])

    rows = []
    for cid, d in per_cell.items():
        rows.append(pd.DataFrame({"timestamp": time_index, "cell_id": cid,
                                   "fault_active": d["fault_active"], "fault_type": d["fault_type"],
                                   "fault_severity": d["fault_severity"]}))
    return pd.concat(rows, ignore_index=True)

def build_eval_fault_events(network_faults, sites, cells, aggregations, core_deps_df, mesh_links):
    rows = []
    for f in network_faults.itertuples():
        scope = resolve_fault_scope(f, sites, cells, aggregations, core_deps_df, mesh_links)
        for entity_type, entities in scope.items():
            for eid in entities:
                rows.append({"entity_id": eid, "entity_type": entity_type.rstrip("s"),
                             "fault_id": f.fault_id, "fault_type": f.fault_type,
                             "severity": f.severity, "start_time": f.start_time, "end_time": f.end_time})
    return pd.DataFrame(rows)

def eval_congestion_events(congestion_events):
    return congestion_events[["congestion_id", "site_id", "start_time", "end_time",
                              "peak_multiplier", "design_util_baseline"]].copy()

def eval_data_anomalies(data_anomalies):
    return data_anomalies[["anomaly_id", "cell_id", "start_time", "end_time", "multiplier"]].copy()


def create_ml_congestion_forecast(ml_hourly_ran, watch_band_cell_ids=None):
    """Build the congestion forecasting feature table from hourly RAN rollup.
    Lags, rolling stats, calendar features, and future PRB labels (t+1h, t+6h, t+24h).
    Labels are the cell's own future value — no hidden table involved."""
    import numpy as np

    def rolling_slope(values):
        y = np.asarray(values, dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 2:
            return np.nan
        x = np.arange(len(y))[valid]
        return np.polyfit(x, y[valid], 1)[0]

    df = ml_hourly_ran[["cell_id", "timestamp", "prb_util_mean", "traffic_mbps_mean",
                          "technology", "region_name", "site_type", "cell_capacity_mbps"]].copy()
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("cell_id")

    # Lags
    df["prb_lag_1h"] = g["prb_util_mean"].shift(1)
    df["prb_lag_2h"] = g["prb_util_mean"].shift(2)
    df["prb_lag_24h"] = g["prb_util_mean"].shift(24)
    df["prb_lag_168h"] = g["prb_util_mean"].shift(168)
    df["traffic_lag_1h"] = g["traffic_mbps_mean"].shift(1)
    df["traffic_lag_24h"] = g["traffic_mbps_mean"].shift(24)

    # Rolling
    df["prb_mean_24h"] = g["prb_util_mean"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    df["prb_std_24h"] = g["prb_util_mean"].transform(lambda s: s.rolling(24, min_periods=2).std())
    df["prb_mean_7d"] = g["prb_util_mean"].transform(lambda s: s.rolling(24 * 7, min_periods=24).mean())
    df["prb_slope_7d"] = g["prb_util_mean"].transform(
        lambda s: s.rolling(24 * 14, min_periods=48).apply(rolling_slope, raw=True))
    df["traffic_mean_24h"] = g["traffic_mbps_mean"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    df["traffic_slope_7d"] = g["traffic_mbps_mean"].transform(
        lambda s: s.rolling(24 * 7, min_periods=24).apply(rolling_slope, raw=True))

    # Current state
    df["prb_util_t"] = df["prb_util_mean"]
    df["traffic_t"] = df["traffic_mbps_mean"]
    df["capacity_headroom_current"] = df["cell_capacity_mbps"] - df["traffic_t"]

    # Calendar
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Watch-band flag
    if watch_band_cell_ids is not None:
        df["is_watch_band_cell"] = df["cell_id"].isin(watch_band_cell_ids).astype(int)

    # Labels: the cell's own FUTURE value — no hidden table involved
    df["prb_t_plus_1h"] = g["prb_util_mean"].shift(-1)
    df["prb_t_plus_6h"] = g["prb_util_mean"].shift(-6)
    df["prb_t_plus_24h"] = g["prb_util_mean"].shift(-24)

    return df


def create_ml_anomaly_features(ml_hourly_ran, seasonal_window_weeks=4, zscore_window_hours=24 * 7):
    """Per-cell, per-metric seasonal baseline + z-score table used by build_anomaly_baselines."""
    import numpy as np

    df = ml_hourly_ran[["cell_id", "timestamp", "prb_util_mean", "traffic_mbps_mean",
                          "sinr_mean", "bler_mean"]].copy()
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    metrics = ["prb_util_mean", "traffic_mbps_mean", "sinr_mean", "bler_mean"]

    for metric in metrics:
        def seasonal_expected(group):
            group = group.sort_values("timestamp")
            return group[metric].shift(1).rolling(seasonal_window_weeks, min_periods=1).mean()

        df[f"{metric}_expected"] = (
            df.groupby(["cell_id", "hour_of_day", "day_of_week"], group_keys=False)
              .apply(seasonal_expected, include_groups=False)
        )
        df[f"{metric}_residual"] = df[metric] - df[f"{metric}_expected"]

        rolling_std = (df.groupby("cell_id")[f"{metric}_residual"]
                         .transform(lambda s: s.rolling(zscore_window_hours, min_periods=24).std()))
        df[f"{metric}_zscore"] = df[f"{metric}_residual"] / rolling_std.replace(0, np.nan)

    return df

