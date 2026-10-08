import pandas as pd
import numpy as np
import networkx as nx
from simulator.topology import compute_design_load, compute_mesh_failover_load, compute_core_load_scenarios

def _flag_outliers(util_series, low, high, label):
    for entity_id, u in util_series.items():
        if u < low:
            print(f"  LOW: {entity_id} at {u:.0%} utilization — capacity likely overprovisioned")
        elif u > high:
            print(f"  HIGH: {entity_id} at {u:.0%} utilization — no headroom for normal peak variation")

_flag = _flag_outliers

def audit_capacity_chain(regions, sites, cells, aggregations, backhaul_links, mesh_links, core_nodes, core_deps_df,
                          low_util=0.30, high_util=0.90):
    loads = compute_design_load(regions, sites, cells)
    agg_demand = loads["aggregation_load"].set_index("aggregation_id")["aggregation_demand_gbps"]

    print("=== CELL ===")
    c = loads["cell_load"]
    print((c["cell_demand_gbps"] / (c["cell_capacity_mbps"] / 1000)).describe())

    print("\n=== AGGREGATION ===")
    agg_util = agg_demand / aggregations.set_index("aggregation_id")["capacity"]
    print(agg_util.describe()); _flag(agg_util, low_util, high_util, "aggregation")

    print("\n=== BACKHAUL ===")
    bh_util = (loads["site_load"].set_index("site_id")["site_demand_gbps"]
               / backhaul_links.set_index("source_id")["capacity"])
    print(bh_util.describe())

    print("\n=== TRANSPORT MESH (worst-case local-outage failover) ===")
    mesh_G_for_audit = nx.from_pandas_edgelist(
        mesh_links.rename(columns={"distance_km": "weight"}), "source_id", "target_id", edge_attr="weight"
    )
    _, routes = compute_mesh_failover_load(mesh_G_for_audit, agg_demand, core_deps_df)
    print(routes[["source_aggregation_id", "gateway_aggregation_id", "source_demand_gbps", "path_km"]])

    print("\n=== CORE: normal vs worst single-core-failure ===")
    normal, worst = compute_core_load_scenarios(agg_demand, core_deps_df)
    cap = core_nodes.set_index("core_id")["capacity"]
    normal_util, worst_util = normal / cap, worst / cap
    print("normal:\n", normal_util.describe())
    print("worst-failure:\n", worst_util.describe())
    _flag(normal_util, low_util, high_util, "core-normal")
    _flag(worst_util, 0.0, 1.0, "core-failover")
