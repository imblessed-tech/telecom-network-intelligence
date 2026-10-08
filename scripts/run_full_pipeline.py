import argparse
import sys
import os
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from simulator.topology import (
    create_regions, create_sites, assign_site_coordinates, assign_site_type,
    create_cells, assign_sector_and_band, assign_baseline_heterogeneity, assign_cell_capacity,
    assign_aggregation_nodes, create_aggregation_nodes, assign_aggregation_capacity,
    build_backhaul_links, build_transport_mesh, create_core_nodes,
    assign_core_dependencies, assign_core_capacity, size_transport_mesh,
    build_transport_links_table, build_topology_table,compute_design_load,
)
from simulator.capacity import audit_capacity_chain
from simulator.operational import (
    generate_traffic_demand, inject_anomalies, generate_congestion_events, apply_congestion_to_demand,
    generate_network_faults, build_neighbor_map, generate_ran_telemetry, compute_hierarchy_loads,
    generate_transport_telemetry, generate_core_telemetry, generate_alarms, build_evaluation_labels,
    create_ml_congestion_forecast, create_ml_anomaly_features,
)
from ml_pipeline.hourly import create_ml_hourly_ran, create_ml_hourly_mesh, create_ml_hourly_core
from ml_pipeline.features import create_ml_congestion_forecast
from ml_pipeline.labels import build_concurrent_cell_labels
from ml_pipeline.train import (
    train_cell_fault_model, train_congestion_models, build_anomaly_baselines,
    train_mesh_fault_detector, train_core_fault_detector,
)
from ml_pipeline.config import ARTIFACTS_DIR

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR, HOURLY_DIR, ML_DIR = DATA_DIR / "raw", DATA_DIR / "hourly", DATA_DIR / "ml"
RNG_SEED = 42


def checkpoint_exists(name, directory):
    return (directory / f"{name}.parquet").exists()


def save(df, name, directory):
    directory.mkdir(parents=True, exist_ok=True)
    df.to_parquet(directory / f"{name}.parquet", index=False)


def load(name, directory):
    return pd.read_parquet(directory / f"{name}.parquet")


def run_static_topology(force):
    tables = ["regions", "sites", "cells", "aggregations", "core_nodes",
              "backhaul_links", "mesh_links", "core_deps_df", "transport_links", "topology"]
    if not force and all(checkpoint_exists(t, RAW_DIR) for t in tables):
        print("Static topology already generated — skipping (use --force to regenerate).")
        return {t: load(t, RAW_DIR) for t in tables}

    print("Generating static topology...")
    rng = np.random.default_rng(RNG_SEED)
    regions = create_regions()
    sites = create_sites(regions)
    sites = assign_site_coordinates(sites, regions, rng=rng)
    sites = assign_site_type(sites, regions)
    cells = create_cells(regions, sites, rng)
    cells = assign_sector_and_band(cells, rng)
    cells = assign_baseline_heterogeneity(cells, rng)
    cells = assign_cell_capacity(cells)

    sites, cluster_centers = assign_aggregation_nodes(sites)
    aggregations = create_aggregation_nodes(cluster_centers)
    aggregations = assign_aggregation_capacity(regions, sites, cells, aggregations)
    backhaul_links = build_backhaul_links(regions, sites, cells, aggregations)

    transport_mesh_G = build_transport_mesh(aggregations)
    core_nodes = create_core_nodes()
    core_deps_df = assign_core_dependencies(aggregations, core_nodes, rng)
    core_nodes = assign_core_capacity(core_nodes, core_deps_df, aggregations)
    mesh_links, _ = size_transport_mesh(transport_mesh_G, regions, sites, cells, core_deps_df)

    transport_links = build_transport_links_table(backhaul_links, mesh_links, core_deps_df)
    topology = build_topology_table(cells, backhaul_links, mesh_links, core_deps_df)

    # if any of the above tables are none type raise error, state the exact table
    if any([df is None for df in [regions, sites, cells, aggregations, core_nodes, backhaul_links, mesh_links, core_deps_df, transport_links, topology]]):
        raise ValueError(f"One or more topology tables are None { [df for df in [regions, sites, cells, aggregations, core_nodes, backhaul_links, mesh_links, core_deps_df, transport_links, topology] if df is None] }")
    audit_capacity_chain(regions, sites, cells, aggregations, backhaul_links, mesh_links, core_nodes, core_deps_df)

    result = {"regions": regions, "sites": sites, "cells": cells, "aggregations": aggregations,
              "core_nodes": core_nodes, "backhaul_links": backhaul_links, "mesh_links": mesh_links,
              "core_uplink_links": core_deps_df, "transport_links": transport_links, "topology": topology}
    for name, df in result.items():
        save(df, name, RAW_DIR)
    return result


def run_operational_layer(topo, force):
    tables = ["traffic_demand", "network_faults", "ran_telemetry", "transport_telemetry",
              "core_telemetry", "network_alarms", "data_anomalies", "congestion_events", "evaluation_labels"]
    if not force and all(checkpoint_exists(t, RAW_DIR) for t in tables):
        print("Operational telemetry already generated — skipping.")
        return {t: load(t, RAW_DIR) for t in tables}

    print("Generating 90 days of operational telemetry (this is the slow stage)...")
    rng = np.random.default_rng(RNG_SEED)
    regions, sites, cells = topo["regions"], topo["sites"], topo["cells"]
    aggregations, core_nodes = topo["aggregations"], topo["core_nodes"]

    backhaul_links, mesh_links, core_uplink_links = topo["backhaul_links"], topo["mesh_links"], topo["core_uplink_links"]# if False else "core_deps_df"]
    transport_links = topo["transport_links"]

    if "target_utilization" in cells.columns:
        cells = cells.drop(columns="target_utilization")
    cells = cells.merge(compute_design_load(regions, sites, cells)["cell_load"][["cell_id", "target_utilization"]], on="cell_id")

    traffic_demand = generate_traffic_demand(regions, sites, cells, rng)
    traffic_demand, data_anomalies = inject_anomalies(traffic_demand, rng)
    congestion_events = generate_congestion_events(regions, sites, cells, backhaul_links, rng)
    traffic_demand = apply_congestion_to_demand(traffic_demand, congestion_events, cells)
    network_faults = generate_network_faults(cells, sites, aggregations, core_nodes, mesh_links, rng)

    neighbor_map = build_neighbor_map(cells, sites)
    ran_telemetry = generate_ran_telemetry(cells, sites, traffic_demand, network_faults, neighbor_map,
                                            sites, aggregations, core_uplink_links, rng)
    site_series, agg_series, core_series = compute_hierarchy_loads(cells, ran_telemetry, sites, aggregations,
                                                                     core_uplink_links, network_faults)
    transport_telemetry = generate_transport_telemetry(backhaul_links, mesh_links, core_uplink_links,
                                                         site_series, agg_series, network_faults, transport_links, rng)
    core_telemetry = generate_core_telemetry(core_nodes, core_series, network_faults, rng)
    network_alarms = generate_alarms(ran_telemetry, transport_telemetry, core_telemetry, rng)
    evaluation_labels = build_evaluation_labels(network_faults, sites, cells, aggregations, core_uplink_links, mesh_links)

    result = {"traffic_demand": traffic_demand, "network_faults": network_faults,
              "ran_telemetry": ran_telemetry, "transport_telemetry": transport_telemetry,
              "core_telemetry": core_telemetry, "network_alarms": network_alarms,
              "data_anomalies": data_anomalies, "congestion_events": congestion_events,
              "evaluation_labels": evaluation_labels}
    for name, df in result.items():
        save(df, name, RAW_DIR)
    return result


def run_ml_tables(topo, ops, force):
    needed = ["ml_hourly_ran_labeled", "ml_hourly_mesh", "ml_hourly_core", "ml_congestion_forecast"]
    if not force and all(checkpoint_exists(t, ML_DIR) for t in needed):
        print("ML tables already built — skipping.")
        return {t: load(t, ML_DIR) for t in needed}

    print("Building hourly rollups, labels and ML tables...")
    cells = topo["cells"]
    hourly_ran = create_ml_hourly_ran(ops["ran_telemetry"], topo["sites"], cells, topo["regions"], ops["network_alarms"])
    hourly_mesh = create_ml_hourly_mesh(ops["transport_telemetry"], topo["mesh_links"])
    hourly_core = create_ml_hourly_core(ops["core_telemetry"], ops["network_alarms"])

    cell_labels = build_concurrent_cell_labels(
        hourly_ran["cell_id"].unique(), hourly_ran["timestamp"], ops["network_faults"],
        topo["sites"], cells, topo["aggregations"], topo["core_uplink_links"], topo["mesh_links"])
    hourly_ran_labeled = hourly_ran.merge(cell_labels, on=["cell_id", "timestamp"], how="left", validate="one_to_one")
    assert len(hourly_ran_labeled) == len(hourly_ran), "label merge changed row count"

    watch_ids = cells.loc[cells["watch_utilization_override"].notna(), "cell_id"].tolist()
    congestion = create_ml_congestion_forecast(hourly_ran, watch_ids)

    result = {"ml_hourly_ran_labeled": hourly_ran_labeled, "ml_hourly_mesh": hourly_mesh,
              "ml_hourly_core": hourly_core, "ml_congestion_forecast": congestion}
    for name, df in result.items():
        save(df, name, ML_DIR)
    return result


def run_training(ml, ops):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Training cell fault detector...")
    train_cell_fault_model(ml["ml_hourly_ran_labeled"])
    print("Training congestion forecasters...")
    train_congestion_models(ml["ml_congestion_forecast"])
    print("Building anomaly baselines...")
    build_anomaly_baselines(ml["ml_hourly_ran_labeled"])
    print("Evaluating mesh fault detection...")
    train_mesh_fault_detector(ml["ml_hourly_mesh"], ops["network_faults"])
    print("Evaluating core fault detection...")
    train_core_fault_detector(ml["ml_hourly_core"], ops["network_faults"])
    print(f"\nDone. Artifacts written to: {ARTIFACTS_DIR}")



# def run_training(ml_tables, ops, force):
#     print("Training models and saving artifacts...")

#     # 1. Fault detection (XGBoost classifier)
#     train_cell_fault_model(ml_tables["ml_cell_fault"], ops["network_faults"])

#     # 2. Congestion forecasting (XGBoost regressor, three horizons: 1h, 6h, 24h)
#     print("Training congestion models...")
#     train_congestion_models(ml_tables["ml_congestion_forecast"])

#     # 3. Anomaly detection (statistical baselines saved as lookup table)
#     print("Building anomaly baselines...")
#     build_anomaly_baselines(ml_tables["ml_hourly_ran"])

#     print(f"\nDone. Artifacts written to: {ARTIFACTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Run the full simulator -> ML pipeline end to end.")
    parser.add_argument("--force", action="store_true", help="Regenerate every stage, ignoring checkpoints.")
    parser.add_argument("--stop-after", choices=["topology", "operational", "ml_tables", "training"], default="training")
    args = parser.parse_args()

    topo = run_static_topology(args.force)
    if args.stop_after == "topology":
        return
    ops = run_operational_layer(topo, args.force)
    if args.stop_after == "operational":
        return
    ml = run_ml_tables(topo, ops, args.force)
    if args.stop_after == "ml_tables":
        return
    run_training(ml, ops)


if __name__ == "__main__":
    main()
