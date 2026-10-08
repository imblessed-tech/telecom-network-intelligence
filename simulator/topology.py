import pandas as pd
import numpy as np
import networkx as nx
from k_means_constrained import KMeansConstrained

NETWORK_CONFIG = {
    "n_regions": 5,
    "n_sites": 50,
    "n_aggregation_nodes": 10,
    "n_transport_links": 20,
    "n_core_nodes": 5,
    "seed": 42,
}
REGION_INFO = {
    "Lagos": {"primary_hub": "Lagos", "population_pressure": "very_high", "urbanization": "very_high", "traffic_demand": "very_high"},
    "West":  {"primary_hub": "Ibadan", "population_pressure": "high", "urbanization": "high", "traffic_demand": "high"},
    "North": {"primary_hub": "Abuja/Kaduna", "population_pressure": "high", "urbanization": "medium_high", "traffic_demand": "high"},
    "South": {"primary_hub": "Port Harcourt", "population_pressure": "medium_high", "urbanization": "high", "traffic_demand": "high"},
    "East":  {"primary_hub": "Enugu", "population_pressure": "medium", "urbanization": "medium_high", "traffic_demand": "medium_high"},
}
REGION_ANCHORS = {
    "Lagos": (6.5244, 3.3792),
    "West":  (7.3775, 3.9470),
    "North": (10.5, 8.5),
    "South": (4.8156, 7.0498),
    "East":  (6.4413, 7.4988),
}
REGION_SPREAD_DEG = {
    "Lagos": 0.12, "West": 0.25,
    "North": 1.8,
    "South": 0.20, "East": 0.30,
}
CORE_ANCHORS = [
    (7.00, 4.50),
    (9.50, 8.20),
    (5.60, 6.20),
    (6.90, 6.00),
    (8.20, 5.30),
]
REGION_FACTOR_VALUES = {"very_low": 0.70, "low": 0.80, "low_medium": 0.90, "medium": 1.00, "medium_high": 1.10, "high": 1.20, "very_high": 1.35}
SITE_TYPE_MULTIPLIER = {"urban": 1.15, "suburban": 1.00, "rural": 0.70}
TECH_MULTIPLIER = {"4G": 1.00, "5G": 1.15}
rng = np.random.default_rng(NETWORK_CONFIG["seed"])
CELL_TARGET_UTILIZATION_BY_SITE_TYPE = {"urban": 0.65, "suburban": 0.50, "rural": 0.35}
CELL_CAPACITY_MBPS = {
    ("4G", "700MHz"): 75, ("4G", "1800MHz"): 100, ("4G", "2100MHz"): 110, ("4G", "2600MHz"): 150,
    ("5G", "2100MHz"): 300, ("5G", "2600MHz"): 500, ("5G", "3500MHz"): 800,
}
TRANSPORT_MEDIUM_PROFILES = [(15, "fiber_metro", 20.0), (60, "fiber_regional", 10.0), (np.inf, "microwave", 2.5)]

def factor_lookup(factor):
    scale = {"very_high": 10, "high": 7.5, "medium_high": 6, "medium": 5,
             "low_medium": 4, "low": 3, "very_low": 1}
    if factor not in scale:
        raise ValueError(f"Unrecognized factor level: {factor!r}")
    return scale[factor]

def create_regions(region_info=REGION_INFO, n_sites=NETWORK_CONFIG["n_sites"], anchors=REGION_ANCHORS):
    weighted = {}
    for region, info in region_info.items():
        weight = (factor_lookup(info["population_pressure"]) * 0.50
                  + factor_lookup(info["urbanization"]) * 0.15
                  + factor_lookup(info["traffic_demand"]) * 0.35)
        weighted[region] = {**info, "site_allocation_weight": weight}

    total_weight = sum(v["site_allocation_weight"] for v in weighted.values())

    rows = []
    for i, (region, info) in enumerate(weighted.items(), start=1):
        lat, lon = anchors[region]
        rows.append({
            "region_id": f"region_{i:03d}",
            "region_name": region,
            "hub_city": info["primary_hub"],
            "population_factor": info["population_pressure"],
            "urbanization_factor": info["urbanization"],
            "traffic_demand_factor": info["traffic_demand"],
            "latitude": lat,
            "longitude": lon,
            "site_target": round(info["site_allocation_weight"] / total_weight * n_sites),
        })
    return pd.DataFrame(rows)

def create_sites(regions):
    rows = []
    cnt = 1
    for row in regions.itertuples():
        for _ in range(row.site_target):
            rows.append({"region_id": row.region_id, "site_id": f"site_{cnt:03d}"})
            cnt += 1
    return pd.DataFrame(rows)

def assign_site_coordinates(sites, regions, spread_lookup=REGION_SPREAD_DEG, rng=rng):
    region_lookup = regions.set_index("region_id")[["region_name", "latitude", "longitude"]]
    sites = sites.copy()
    lats, lons = [], []
    for region_id in sites["region_id"]:
        name, base_lat, base_lon = region_lookup.loc[region_id]
        spread = spread_lookup[name]
        lats.append(np.clip(base_lat + rng.normal(0, spread), base_lat - 3*spread, base_lat + 3*spread))
        lons.append(np.clip(base_lon + rng.normal(0, spread), base_lon - 3*spread, base_lon + 3*spread))
    sites["latitude"], sites["longitude"] = lats, lons
    return sites

def assign_site_type(sites, regions):
    region_lookup = regions.set_index("region_id")[["urbanization_factor", "latitude", "longitude"]]
    sites = sites.copy()

    def classify(row):
        urban_factor, hub_lat, hub_lon = region_lookup.loc[row.region_id]
        dist_from_hub = np.hypot(row.latitude - hub_lat, row.longitude - hub_lon)
        if dist_from_hub < 0.15 and urban_factor in ("very_high", "high"):
            return "urban"
        elif dist_from_hub < 0.35:
            return "suburban"
        return "rural"

    sites["site_type"] = sites.apply(classify, axis=1)
    return sites

def create_cells(regions, sites, rng=rng):
    cell_probs = {
        "Lagos": ([1,2,3,4,5,6], [0.00,0.10,0.45,0.20,0.15,0.10], [0.40,0.60]),
        "West":  ([1,2,3,4,5,6], [0.05,0.15,0.50,0.15,0.10,0.05], [0.50,0.50]),
        "North": ([1,2,3,4,5,6], [0.10,0.15,0.50,0.15,0.08,0.02], [0.60,0.40]),
    }
    default = ([1,2,3,4,5,6], [0.10,0.20,0.60,0.08,0.02,0.00], [0.75,0.25])

    rows = []
    cnt = 1
    for region in regions.itertuples(index=False):
        region_sites = sites[sites.region_id == region.region_id]
        n = len(region_sites)  # <- actual count, not region.site_target
        counts, count_probs, tech_probs = cell_probs.get(region.region_name, default)

        cell_distr = rng.choice(counts, n, p=count_probs)
        tech_distr = rng.choice(["4G", "5G"], n, p=tech_probs)

        for site_row, n_cells, tech in zip(region_sites.itertuples(index=False), cell_distr, tech_distr):
            for _ in range(n_cells):
                rows.append({"site_id": site_row.site_id, "cell_id": f"cell_{cnt:03d}",
                             "region_id": region.region_id, "technology": tech})
                cnt += 1
    return pd.DataFrame(rows)

def assign_sector_and_band(cells, rng=rng):
    cells = cells.copy()
    cells["sector"] = cells.groupby("site_id").cumcount() % 3 + 1

    band_map = {
        "4G": (["700MHz","1800MHz","2100MHz","2600MHz"], [0.15,0.35,0.30,0.20]),
        "5G": (["2100MHz","2600MHz","3500MHz"], [0.15,0.25,0.60]),
    }
    cells["band"] = cells["technology"].apply(
        lambda t: rng.choice(band_map[t][0], p=band_map[t][1])
    )
    return cells

def assign_baseline_heterogeneity(cells, rng, watch_fraction=0.12, watch_range=(0.75, 0.85)):
    """Gives each cell a persistent 'identity' instead of every cell in the same
    (technology, site_type) bucket behaving identically up to per-timestep noise.
    - quality_offset_db: a fixed radio-quality offset per cell, applied on top of the
      existing bucket baseline in generate_ran_telemetry.
    - demand_variation_factor: a fixed demand multiplier per cell, so cells in the
      same bucket don't all run at exactly the same average utilization.
    - watch_utilization_override: for a deliberately chosen subset of cells, pins
      their target utilization into a persistent 70-85% band with NO fault entry —
      this is what creates a Healthy -> Watch -> Degraded -> Fault continuum instead
      of only Healthy/Fault.
    """
    cells = cells.copy()
    n = len(cells)
    cells["quality_offset_db"] = rng.normal(0, 1.5, n)
    cells["demand_variation_factor"] = rng.uniform(0.85, 1.20, n)

    watch_mask = rng.random(n) < watch_fraction
    cells["watch_utilization_override"] = np.where(watch_mask, rng.uniform(*watch_range, n), np.nan)
    return cells

def assign_cell_capacity(cells, capacity_map=CELL_CAPACITY_MBPS):
    cells = cells.copy()
    keys = list(zip(cells["technology"], cells["band"]))
    cells["cell_capacity_mbps"] = pd.Series(keys, index=cells.index).map(capacity_map)
    missing = cells[cells["cell_capacity_mbps"].isna()]
    if not missing.empty:
        raise ValueError(f"No capacity mapping for:\n{missing[['technology','band']].drop_duplicates()}")
    return cells

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km. Works on scalars, arrays, or Series of matching shape."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

def assign_aggregation_nodes(sites, n_clusters=10, size_min=3, size_max=7, seed=NETWORK_CONFIG["seed"]):
    sites = sites.copy()
    X = sites[["latitude", "longitude"]].to_numpy()
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, random_state=seed)
    sites["aggregation_id"] = [f"agg_{i+1:03d}" for i in clf.fit_predict(X)]
    return sites, clf.cluster_centers_

def create_aggregation_nodes(cluster_centers):
    return pd.DataFrame([
        {"aggregation_id": f"agg_{i+1:03d}", "latitude": lat, "longitude": lon}
        for i, (lat, lon) in enumerate(cluster_centers)
    ])

def add_region_demand_multiplier(regions):
    regions = regions.copy()
    for col in ["population_factor", "urbanization_factor", "traffic_demand_factor"]:
        regions[f"{col}_score"] = regions[col].map(REGION_FACTOR_VALUES)
        if regions[f"{col}_score"].isna().any():
            raise ValueError(f"Unmapped value in {col}")
    regions["region_demand_multiplier"] = (
        0.50 * regions["population_factor_score"]
        + 0.15 * regions["urbanization_factor_score"]
        + 0.35 * regions["traffic_demand_factor_score"]
    )
    regions["region_demand_multiplier"] /= regions["region_demand_multiplier"].mean()  # relative, not absolute
    return regions

def compute_design_load(regions, sites, cells, base_utilization=0.50):
    regions = add_region_demand_multiplier(regions)
    df = (cells.merge(sites[["site_id", "site_type", "aggregation_id"]], on="site_id")
                .merge(regions[["region_id", "region_demand_multiplier"]], on="region_id"))

    df["target_utilization"] = (
        base_utilization * df["region_demand_multiplier"]
        * df["site_type"].map(SITE_TYPE_MULTIPLIER)
        * df["technology"].map(TECH_MULTIPLIER)
        * df["demand_variation_factor"]                       # NEW — persistent per-cell demand variation
    ).clip(0.20, 0.85)

    # NEW — Watch-band cells override the formula entirely with a pinned high-utilization band
    df["target_utilization"] = df["watch_utilization_override"].combine_first(df["target_utilization"])

    df["cell_demand_gbps"] = df["cell_capacity_mbps"] * df["target_utilization"] / 1000

    site_load = (df.groupby(["site_id", "aggregation_id"], as_index=False)["cell_demand_gbps"]
                   .sum().rename(columns={"cell_demand_gbps": "site_demand_gbps"}))
    agg_load = (site_load.groupby("aggregation_id", as_index=False)["site_demand_gbps"]
                          .sum().rename(columns={"site_demand_gbps": "aggregation_demand_gbps"}))
    return {"cell_load": df, "site_load": site_load, "aggregation_load": agg_load}

def assign_aggregation_capacity(regions, sites, cells, aggregations, headroom_factor=1.4):
    agg_demand = compute_design_load(regions, sites, cells)["aggregation_load"].set_index("aggregation_id")["aggregation_demand_gbps"]
    aggregations = aggregations.drop(columns="capacity", errors="ignore").copy()
    aggregations["capacity"] = (aggregations["aggregation_id"].map(agg_demand) * headroom_factor).round(1)
    return aggregations

def build_backhaul_links(regions, sites, cells, aggregations, headroom_factor=1.5, min_capacity_gbps=0.3):
    """headroom_factor raised 1.2 -> 1.5: backhaul carries one site's raw, unsmoothed
    demand with no multiplexing gain (unlike aggregation, which pools ~5 sites and
    benefits from bursts partially cancelling out) — so it needs MORE relative margin,
    not less. min_capacity_gbps floors very small single-cell rural sites at a realistic
    minimum link size, rather than letting a tiny demand value round to something that
    creates artificial >100% utilization.
    """
    site_load = compute_design_load(regions, sites, cells)["site_load"]
    merged = (sites.merge(aggregations, on="aggregation_id", suffixes=("_site", "_agg"))
                    .merge(site_load[["site_id", "site_demand_gbps"]], on="site_id"))
    merged["distance_km"] = haversine_km(merged.latitude_site, merged.longitude_site,
                                          merged.latitude_agg, merged.longitude_agg).round(2)

    # full precision for the math; only floor + a wider rounding step for storage
    raw_capacity = merged["site_demand_gbps"] * headroom_factor
    merged["capacity"] = np.maximum(raw_capacity, min_capacity_gbps).round(3)  # 3dp, not 1dp

    out = merged.rename(columns={"site_id": "source_id", "aggregation_id": "target_id"})
    out["source_type"], out["target_type"], out["link_type"] = "site", "aggregation", "backhaul"
    return out[["source_id", "source_type", "target_id", "target_type", "link_type", "distance_km", "capacity"]]

def build_transport_mesh(aggregations, n_links=20, max_degree=5):
    ids = aggregations["aggregation_id"].to_numpy()
    lat, lon = aggregations["latitude"].to_numpy(), aggregations["longitude"].to_numpy()
    i, j = np.triu_indices(len(ids), k=1)  # every unique pair of aggregation nodes, once each
    dist = haversine_km(lat[i], lon[i], lat[j], lon[j])

    candidates = nx.Graph()
    for a, b, d in zip(ids[i], ids[j], dist):
        candidates.add_edge(a, b, weight=d)

    G = nx.minimum_spanning_tree(candidates, weight="weight")  # Kruskal: cheapest edges, no cycles, full connectivity

    leftover = sorted(
        ((d["weight"], u, v) for u, v, d in candidates.edges(data=True) if not G.has_edge(u, v)),
        key=lambda x: x[0]
    )
    for weight, u, v in leftover:
        if G.number_of_edges() >= n_links:
            break
        if G.degree[u] >= max_degree or G.degree[v] >= max_degree:
            continue
        G.add_edge(u, v, weight=weight)

    if G.number_of_edges() < n_links:
        print(f"Note: reached only {G.number_of_edges()}/{n_links} links under max_degree={max_degree}.")
    return G

def classify_medium(distance_km):
    for max_dist, link_type, ceiling in TRANSPORT_MEDIUM_PROFILES:
        if distance_km <= max_dist:
            return link_type, ceiling

def size_transport_mesh(transport_mesh_G, regions, sites, cells, core_deps_df, headroom_factor=1.2, floor_gbps=0.5):
    agg_demand = compute_design_load(regions, sites, cells)["aggregation_load"].set_index("aggregation_id")["aggregation_demand_gbps"]
    failover_load, routes = compute_mesh_failover_load(transport_mesh_G, agg_demand, core_deps_df)

    rows, flagged = [], []
    for u, v, data in transport_mesh_G.edges(data=True):
        edge = tuple(sorted((u, v)))
        distance = data["weight"]
        link_type, ceiling = classify_medium(distance)
        routed = failover_load.get(edge, 0.0)
        required = round(routed * headroom_factor, 2)
        capacity = max(floor_gbps, min(required, ceiling))
        if required > ceiling:
            flagged.append((u, v, link_type, required, ceiling))
        rows.append({"source_id": u, "source_type": "aggregation", "target_id": v, "target_type": "aggregation",
                     "link_type": link_type, "distance_km": round(distance, 2), "capacity": capacity})

    if flagged:
        print(f"{len(flagged)} mesh link(s) undersized for worst-case failover load:")
        for u, v, lt, req, ceil in flagged:
            print(f"  {u}-{v} ({lt}): needs {req} Gbps, ceiling {ceil} -> capped at {ceil}")
    return pd.DataFrame(rows), routes

def compute_mesh_failover_load(transport_mesh_G, aggregation_demand, core_deps_df):
    primary_core = build_core_role_maps(core_deps_df)
    agg_ids = list(transport_mesh_G.nodes)
    edge_load = {tuple(sorted(e)): 0.0 for e in transport_mesh_G.edges}
    routes = []

    for source, demand in aggregation_demand.items():
        if source not in transport_mesh_G:
            continue
        candidates = [a for a in agg_ids if a != source and a in primary_core]
        best_path, best_dist, gateway = None, np.inf, None
        for c in candidates:
            try:
                path = nx.shortest_path(transport_mesh_G, source, c, weight="weight")
                dist = nx.path_weight(transport_mesh_G, path, weight="weight")
            except nx.NetworkXNoPath:
                continue
            if dist < best_dist:
                best_path, best_dist, gateway = path, dist, c
        if best_path is None:
            continue  # isolated node — surfaced separately by connectivity validation

        for u, v in zip(best_path[:-1], best_path[1:]):
            edge = tuple(sorted((u, v)))
            edge_load[edge] = max(edge_load[edge], demand)
        routes.append({"source_aggregation_id": source, "gateway_aggregation_id": gateway,
                        "source_demand_gbps": round(demand, 2), "path": " -> ".join(best_path),
                        "path_km": round(best_dist, 2)})
    return pd.Series(edge_load, name="failover_load_gbps"), pd.DataFrame(routes)

def build_core_role_maps(core_deps_df):
    primary = core_deps_df[core_deps_df.dependency_role == "primary"].set_index("source_id")["target_id"].to_dict()
    return primary

def create_core_nodes(core_anchors=CORE_ANCHORS):
    return pd.DataFrame([
        {"core_id": f"core_{i+1:03d}", "latitude": lat, "longitude": lon}
        for i, (lat, lon) in enumerate(core_anchors)
    ])

def assign_core_dependencies(aggregations, core_nodes, rng=np.random.default_rng(42), max_primary_per_core=None):
    n_agg, n_core = len(aggregations), len(core_nodes)
    if max_primary_per_core is None:
        max_primary_per_core = int(np.ceil(n_agg / n_core)) + 1

    primary_slots = {cid: max_primary_per_core for cid in core_nodes["core_id"]}
    rows = []
    for idx in rng.permutation(n_agg):
        agg = aggregations.iloc[idx]
        dist = pd.Series(haversine_km(agg.latitude, agg.longitude,
                                       core_nodes["latitude"].to_numpy(), core_nodes["longitude"].to_numpy()),
                          index=core_nodes["core_id"])
        open_cores = [c for c in dist.index if primary_slots[c] > 0]
        w = 1 / (dist[open_cores]**2 + 1e-6)
        primary = rng.choice(open_cores, p=(w / w.sum()).to_numpy())
        primary_slots[primary] -= 1

        other = [c for c in dist.index if c != primary]
        w2 = 1 / (dist[other]**2 + 1e-6)
        secondary = rng.choice(other, p=(w2 / w2.sum()).to_numpy())

        for cid, role in [(primary, "primary"), (secondary, "secondary")]:
            rows.append({"source_id": agg.aggregation_id, "source_type": "aggregation", "target_id": cid,
                         "target_type": "core", "link_type": "core_uplink", "distance_km": round(dist[cid], 2),
                         "capacity": agg.capacity, "dependency_role": role})
    return pd.DataFrame(rows)

def assign_core_capacity(core_nodes, core_deps, aggregations, secondary_weight=0.5, headroom_factor=1.2):
    agg_cap = aggregations.set_index("aggregation_id")["capacity"]
    primary_load = core_deps[core_deps.dependency_role == "primary"].assign(
        c=lambda d: d.source_id.map(agg_cap)).groupby("target_id")["c"].sum()
    secondary_load = core_deps[core_deps.dependency_role == "secondary"].assign(
        c=lambda d: d.source_id.map(agg_cap)).groupby("target_id")["c"].sum()
    core_nodes = core_nodes.copy()
    core_nodes["capacity"] = core_nodes["core_id"].map(
        lambda cid: headroom_factor * (primary_load.get(cid, 0) + secondary_weight * secondary_load.get(cid, 0))
    ).round(1)
    return core_nodes

def build_transport_links_table(backhaul, transport, core_deps):
    core_physical = core_deps.drop(columns="dependency_role")
    all_links = pd.concat([backhaul, transport, core_physical], ignore_index=True, sort=False)
    all_links.insert(0, "link_id", [f"link_{i+1:04d}" for i in range(len(all_links))])
    return all_links[["link_id", "source_id", "source_type", "target_id", "target_type",
                       "link_type", "distance_km", "capacity"]]

def build_topology_table(cells, backhaul, transport, core_deps):
    hosts = cells[["cell_id", "site_id"]].rename(columns={"cell_id": "source_id", "site_id": "target_id"})
    hosts["source_type"], hosts["target_type"], hosts["relationship_type"] = "cell", "site", "hosts"
    hosts["distance_km"], hosts["dependency_role"] = 0.0, np.nan

    bh = backhaul[["source_id", "source_type", "target_id", "target_type", "distance_km"]].copy()
    bh["relationship_type"], bh["dependency_role"] = "backhaul", np.nan

    tr = transport[["source_id", "source_type", "target_id", "target_type", "distance_km"]].copy()
    tr["relationship_type"], tr["dependency_role"] = "transport", np.nan

    core = core_deps[["source_id", "source_type", "target_id", "target_type", "distance_km", "dependency_role"]].copy()
    core["relationship_type"] = "core_dependency"

    cols = ["source_id", "source_type", "target_id", "target_type", "relationship_type", "distance_km", "dependency_role"]
    return pd.concat([hosts, bh, tr, core], ignore_index=True, sort=False)[cols]

def validate_topology(topology, transport_mesh_G, aggregations):
    G = nx.from_pandas_edgelist(topology, "source_id", "target_id")
    print("Fully connected end-to-end:", nx.is_connected(G))

    print("Transport-mesh bridges (single points of failure):", list(nx.bridges(transport_mesh_G)))
    print("Transport-mesh node connectivity:", nx.node_connectivity(transport_mesh_G))

    core_edges = topology[topology.relationship_type == "core_dependency"]
    failures = []
    for agg_id, group in core_edges.groupby("source_id"):
        primary = group[group.dependency_role == "primary"].iloc[0]
        secondary = group[group.dependency_role == "secondary"].iloc[0]
        G_minus = G.copy()
        G_minus.remove_edge(primary.source_id, primary.target_id)
        if not nx.has_path(G_minus, agg_id, secondary.target_id):
            failures.append(agg_id)
    print("Aggregation nodes that lose ALL core connectivity if primary fails:", failures)

def compute_core_load_scenarios(aggregation_demand, core_deps_df):
    primary = core_deps_df[core_deps_df.dependency_role == "primary"].copy()
    primary["load"] = primary["source_id"].map(aggregation_demand)
    normal_load = primary.groupby("target_id")["load"].sum()

    secondary = core_deps_df[core_deps_df.dependency_role == "secondary"].copy()
    secondary["load"] = secondary["source_id"].map(aggregation_demand)
    
    worst_load = normal_load.copy()
    core_ids = core_deps_df["target_id"].unique()
    
    for failed_core in core_ids:
        # Find which aggregations had this core as primary
        failed_aggs = primary[primary.target_id == failed_core]["source_id"]
        
        # See which secondary cores they fail over to
        fallback = secondary[secondary.source_id.isin(failed_aggs)]
        fallback_load = fallback.groupby("target_id")["load"].sum()
        
        scenario_load = normal_load.add(fallback_load, fill_value=0)
        scenario_load[failed_core] = 0.0
        
        worst_load = np.maximum(worst_load, scenario_load)
        
    return normal_load, worst_load
