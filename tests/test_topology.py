# in a scratch cell or tests/test_topology_smoke.py
import numpy as np
from simulator.topology import (
    create_regions, 
    create_sites, 
    assign_site_coordinates, 
    assign_site_type, 
    create_cells, 
    assign_sector_and_band, 
    assign_baseline_heterogeneity, 
    assign_cell_capacity, 
    create_aggregation_nodes,
    assign_aggregation_capacity,
    assign_aggregation_nodes,
    build_backhaul_links,
    build_transport_mesh,
    build_core_role_maps,
    build_topology_table,
    build_transport_links_table,
    create_core_nodes,
    assign_core_dependencies,
    assign_core_capacity,
    size_transport_mesh,
    REGION_INFO, 
    NETWORK_CONFIG,
    rng)


print("\n\n===========================")
print("Phases of Topology Creation")
print("===========================\n\n")

print("\nCreating Regions")
regions = create_regions(REGION_INFO, 50)
print(f"Regions:\n{regions}\n\n")

print("\nCreating Sites")
sites = create_sites(regions)

print("\nAssigning Site Coordinates")
sites = assign_site_coordinates(sites, regions, rng=rng)

print("\nAssigning Site Types")
sites = assign_site_type(sites, regions)
print(f"Sites final shape: {sites.shape}")

print("\nCreating Cells")
cells = create_cells(regions, sites, rng)
print(f"Creating cells {cells.shape}")

print("\nAssigning Sector and Band")
cells = assign_sector_and_band(cells, rng)

print("\nAssigning Baseline Heterogeneity")
cells = assign_baseline_heterogeneity(cells, rng)

print("\nAssigning Cell Capacity")
cells = assign_cell_capacity(cells)
print(f"Cells final shape: {cells.shape}")

assert len(sites) == 50 and cells.cell_id.is_unique

print("Phase 1 topology OK:", sites.shape, cells.shape)

print("\n\n=================================================")
print("Phase 2: Demand Generation")
print("=================================================\n\n")

print("\nAssigning Aggregation Nodes")
sites, cluster_centers = assign_aggregation_nodes(sites, n_clusters=NETWORK_CONFIG["n_aggregation_nodes"])
aggregations = create_aggregation_nodes(cluster_centers)
aggregations = assign_aggregation_capacity(regions, sites, cells, aggregations)

print("\nBuild Backhaul mesh")
backhaul_links = build_backhaul_links(regions, sites, cells, aggregations)
transport_mesh_G = build_transport_mesh(aggregations, n_links=NETWORK_CONFIG["n_transport_links"])

print("\nBuild core nodes")
core_nodes = create_core_nodes()

print("\nAssigning core dependencies")
core_deps_df = assign_core_dependencies(aggregations, core_nodes, rng=rng)

print("\nAssigning core capacity")
core_nodes = assign_core_capacity(core_nodes, core_deps_df, aggregations)

print("\nSize transport mesh")
mesh_links, failover_routes = size_transport_mesh(transport_mesh_G, regions, sites, cells, core_deps_df)
transport_links_df = mesh_links[["source_id","source_type","target_id","target_type","link_type","distance_km","capacity"]]
transport_links = build_transport_links_table(backhaul_links, transport_links_df, core_deps_df)

print("\nBuild topology table")
topology = build_topology_table(cells, backhaul_links, transport_links_df, core_deps_df)

print(f"Topology table shape: {topology.shape}")

print("\n\n===========================")
print("Phase 2 topology OK:", (aggregations.shape, backhaul_links.shape, transport_links.shape, core_nodes.shape, topology.shape))
print("===========================")


print("\n===========================================================================")
print("Phase 3: Load and Capacity Audit")
print("===========================================================================")

from simulator.capacity import audit_capacity_chain

# ... after your Phase 2 topology creation code ...

print("\nRunning full capacity audit...")
audit_capacity_chain(regions, sites, cells, aggregations, backhaul_links, mesh_links, core_nodes, core_deps_df)