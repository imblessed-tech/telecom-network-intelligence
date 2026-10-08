import pandas as pd

def create_ml_hourly_ran(ran_telemetry, sites, cells, regions, network_alarms):
    ran_telemetry = ran_telemetry.copy()
    ran_telemetry["timestamp"] = pd.to_datetime(ran_telemetry["timestamp"])
    ran_telemetry["traffic_mbps"] = ran_telemetry["dl_throughput_mbps"] + ran_telemetry["ul_throughput_mbps"]

    hourly_ran_agg = ran_telemetry.groupby(["cell_id", pd.Grouper(key="timestamp", freq="h")]).agg(
        traffic_mbps_mean=("traffic_mbps", "mean"), traffic_mbps_max=("traffic_mbps", "max"),
        prb_util_mean=("prb_utilization_pct", "mean"), prb_util_max=("prb_utilization_pct", "max"),
        sinr_mean=("sinr_db", "mean"), sinr_min=("sinr_db", "min"),
        rsrp_mean=("rsrp_dbm", "mean"), rsrq_mean=("rsrq_db", "mean"),
        bler_mean=("bler_pct", "mean"), bler_max=("bler_pct", "max"),
        rrc_success_mean=("rrc_setup_success_pct", "mean"), erab_success_mean=("erab_setup_success_pct", "mean"),
        drop_rate_mean=("drop_rate_pct", "mean"), handover_success_mean=("handover_success_pct", "mean"),
        availability_mean=("availability_pct", "mean"),
    ).reset_index()

    # FIX: alarm count must be computed PER HOUR, not once across the whole dataset,
    # or every row for a cell gets the same constant (and leaks future alarms into past hours).
    alarms = network_alarms[network_alarms["entity_type"] == "ran"].copy()
    alarms["timestamp"] = pd.to_datetime(alarms["timestamp"])
    hourly_alarm_count = (alarms.groupby(["entity_id", pd.Grouper(key="timestamp", freq="h")])
                                 .size().reset_index(name="alarm_count")
                                 .rename(columns={"entity_id": "cell_id"}))

    cell_static_info = (cells[["cell_id", "site_id", "region_id", "technology", "band", "sector", "cell_capacity_mbps"]]
        .merge(regions[["region_id", "region_name"]], on="region_id", how="left", validate="many_to_one")
        .merge(sites[["site_id", "site_type", "latitude", "longitude", "aggregation_id"]], on="site_id", how="left",
               validate="many_to_one")
        .drop(columns="region_id"))

    ml_hourly_ran = (hourly_ran_agg
        .merge(hourly_alarm_count, on=["cell_id", "timestamp"], how="left")
        .merge(cell_static_info, on="cell_id", how="left", validate="many_to_one"))
    ml_hourly_ran["alarm_count"] = ml_hourly_ran["alarm_count"].fillna(0).astype(int)  # FIX: NaN -> 0
    return ml_hourly_ran


def create_ml_hourly_backhaul(transport_telemetry, backhaul_links):
    tt = transport_telemetry[transport_telemetry["link_type"] == "backhaul"].copy()
    tt["timestamp"] = pd.to_datetime(tt["timestamp"])

    hourly = tt.groupby(["source_id", pd.Grouper(key="timestamp", freq="h")]).agg(
        aggregation_id=("target_id", "first"),
        utilization_mean=("utilization_pct", "mean"), utilization_max=("utilization_pct", "max"),
        latency_mean_ms=("latency_ms", "mean"), jitter_mean_ms=("jitter_ms", "mean"),
        packet_loss_mean_pct=("packet_loss_pct", "mean"), availability_mean=("availability_pct", "mean"),
        link_status_mode=("link_status", lambda x: x.value_counts().idxmax()),  # faster than .mode()
    ).reset_index()

    links = backhaul_links[["source_id", "distance_km", "capacity"]].drop_duplicates()
    ml_hourly_backhaul = hourly.merge(links, on="source_id", how="left", validate="many_to_one") \
                                .rename(columns={"source_id": "site_id"})
    return ml_hourly_backhaul


def create_ml_hourly_mesh(transport_telemetry, mesh_links):
    # FIX: mesh edges are undirected, but transport_telemetry rows can have either
    # aggregation as source_id. Build an order-independent pair key on BOTH sides
    # before grouping/joining, so a link isn't split or dropped by source/target ordering.
    tt = transport_telemetry[transport_telemetry["link_type"].isin(
        ["fiber_metro", "fiber_regional", "microwave"])].copy()
    tt["timestamp"] = pd.to_datetime(tt["timestamp"])
    tt["pair_key"] = tt.apply(lambda r: tuple(sorted((r.source_id, r.target_id))), axis=1)

    hourly = tt.groupby(["pair_key", pd.Grouper(key="timestamp", freq="h")]).agg(
        utilization_mean=("utilization_pct", "mean"), utilization_max=("utilization_pct", "max"),
        latency_mean_ms=("latency_ms", "mean"), jitter_mean_ms=("jitter_ms", "mean"),
        packet_loss_mean_pct=("packet_loss_pct", "mean"), availability_mean=("availability_pct", "mean"),
        link_status_mode=("link_status", lambda x: x.value_counts().idxmax()),
    ).reset_index()

    hourly["source_agg_id"] = hourly["pair_key"].apply(lambda p: p[0])
    hourly["target_agg_id"] = hourly["pair_key"].apply(lambda p: p[1])

    links = mesh_links.copy()
    links["pair_key"] = links.apply(lambda r: tuple(sorted((r.source_id, r.target_id))), axis=1)
    links = links[["pair_key", "link_type", "distance_km", "capacity"]].drop_duplicates("pair_key")

    ml_hourly_mesh = hourly.merge(links, on="pair_key", how="left", validate="many_to_one").drop(columns="pair_key")
    return ml_hourly_mesh


def create_ml_hourly_core_uplink(transport_telemetry, core_deps_df):
    tt = transport_telemetry[transport_telemetry["link_type"] == "core_uplink"].copy()
    tt["timestamp"] = pd.to_datetime(tt["timestamp"])

    hourly = tt.groupby(["source_id", "target_id", pd.Grouper(key="timestamp", freq="h")]).agg(
        utilization_mean=("utilization_pct", "mean"), utilization_max=("utilization_pct", "max"),
        latency_mean_ms=("latency_ms", "mean"), jitter_mean_ms=("jitter_ms", "mean"),
        packet_loss_mean_pct=("packet_loss_pct", "mean"), availability_mean=("availability_pct", "mean"),
        link_status_mode=("link_status", lambda x: x.value_counts().idxmax()),
    ).reset_index()
    # FIX: original grouped by source_id alone, first()-ing target_id — but an aggregation
    # has TWO core uplinks (primary + secondary), so that silently collapsed both into one
    # row per hour. Group by (source_id, target_id) so primary and secondary stay separate rows.

    deps = core_deps_df[["source_id", "target_id", "distance_km", "capacity", "dependency_role"]].drop_duplicates()
    ml_hourly_core_uplink = (hourly.merge(deps, on=["source_id", "target_id"], how="left", validate="many_to_one")
                                    .rename(columns={"source_id": "aggregation_id", "target_id": "core_id"}))
    return ml_hourly_core_uplink


def create_ml_hourly_core(core_telemetry, network_alarms):
    ct = core_telemetry.copy()
    ct["timestamp"] = pd.to_datetime(ct["timestamp"])

    hourly = ct.groupby(["core_id", pd.Grouper(key="timestamp", freq="h")]).agg(
        cpu_mean_pct=("cpu_utilization_pct", "mean"), cpu_max_pct=("cpu_utilization_pct", "max"),
        memory_mean_pct=("memory_utilization_pct", "mean"), memory_max_pct=("memory_utilization_pct", "max"),
        session_failure_rate_mean=("session_failure_rate_pct", "mean"),
        availability_mean=("availability_pct", "mean"),
    ).reset_index()

    # FIX: same per-hour grouping fix as ml_hourly_ran's alarm count
    alarms = network_alarms[network_alarms["entity_type"] == "core"].copy()
    alarms["timestamp"] = pd.to_datetime(alarms["timestamp"])
    hourly_alarm_count = (alarms.groupby(["entity_id", pd.Grouper(key="timestamp", freq="h")])
                                 .size().reset_index(name="alarm_count")
                                 .rename(columns={"entity_id": "core_id"}))

    ml_hourly_core = hourly.merge(hourly_alarm_count, on=["core_id", "timestamp"], how="left")
    ml_hourly_core["alarm_count"] = ml_hourly_core["alarm_count"].fillna(0).astype(int)
    return ml_hourly_core