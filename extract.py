import json
import re
import os

notebook_path = 'notebooks/Telecom_Network_Intelligence_v3 (1).ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

source_lines = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_lines.extend(cell['source'])
        source_lines.append('\n\n')

# We'll use ast or simpler parsing
import ast

source_code = "".join(source_lines)
# Remove ipython magics
source_code = re.sub(r'^\s*!.*$', '', source_code, flags=re.MULTILINE)

class FunctionExtractor(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}
    
    def visit_FunctionDef(self, node):
        self.functions[node.name] = ast.get_source_segment(source_code, node)
        self.generic_visit(node)

tree = ast.parse(source_code)
extractor = FunctionExtractor()
extractor.visit(tree)

functions_map = extractor.functions

def get_code(names):
    code = []
    for name in names:
        if name in functions_map:
            code.append(functions_map[name])
        else:
            print(f"Function {name} not found!")
    return "\n\n".join(code)

topo_fns = [
    "factor_lookup", "create_regions", "create_sites", "assign_site_coordinates",
    "assign_site_type", "create_cells", "assign_sector_and_band",
    "assign_baseline_heterogeneity", "assign_cell_capacity", "haversine_km",
    "assign_aggregation_nodes", "create_aggregation_nodes",
    "add_region_demand_multiplier", "compute_design_load",
    "assign_aggregation_capacity", "build_backhaul_links", "build_transport_mesh",
    "classify_medium", "size_transport_mesh", "compute_mesh_failover_load", "build_core_role_maps",
    "create_core_nodes", "assign_core_dependencies", "assign_core_capacity",
    "build_transport_links_table", "build_topology_table", "validate_topology"
]

cap_fns = [
    "audit_capacity_chain", "_flag_outliers"
]

ops_fns = [
    "build_time_index", "diurnal_curve", "generate_traffic_demand", "inject_anomalies",
    "generate_congestion_events", "apply_congestion_to_demand", "build_neighbor_map",
    "generate_network_faults", "fault_intensity", "resolve_fault_scope",
    "generate_ran_telemetry", "compute_hierarchy_loads", "build_link_load_lookup",
    "generate_link_telemetry", "generate_transport_telemetry", "generate_core_telemetry",
    "generate_alarms", "fire_with_hysteresis", "build_evaluation_labels",
    "build_eval_fault_events", "eval_congestion_events", "eval_data_anomalies",
    "create_ml_hourly_ran"
]

labels_fns = [
    "build_fault_labels_for_cells"
]

topo_code = f"""import pandas as pd
import numpy as np
import networkx as nx
from k_means_constrained import KMeansConstrained

NETWORK_CONFIG = {{
    "n_regions": 5,
    "n_sites": 50,
    "n_aggregation_nodes": 10,
    "n_transport_links": 20,
    "n_core_nodes": 5,
    "seed": 42,
}}
REGION_INFO = {{
    "Lagos": {{"primary_hub": "Lagos", "population_pressure": "very_high", "urbanization": "very_high", "traffic_demand": "very_high"}},
    "West":  {{"primary_hub": "Ibadan", "population_pressure": "high", "urbanization": "high", "traffic_demand": "high"}},
    "North": {{"primary_hub": "Abuja/Kaduna", "population_pressure": "high", "urbanization": "medium_high", "traffic_demand": "high"}},
    "South": {{"primary_hub": "Port Harcourt", "population_pressure": "medium_high", "urbanization": "high", "traffic_demand": "high"}},
    "East":  {{"primary_hub": "Enugu", "population_pressure": "medium", "urbanization": "medium_high", "traffic_demand": "medium_high"}},
}}
REGION_ANCHORS = {{
    "Lagos": (6.5244, 3.3792),
    "West":  (7.3775, 3.9470),
    "North": (10.5, 8.5),
    "South": (4.8156, 7.0498),
    "East":  (6.4413, 7.4988),
}}
REGION_SPREAD_DEG = {{
    "Lagos": 0.12, "West": 0.25,
    "North": 1.8,
    "South": 0.20, "East": 0.30,
}}
CORE_ANCHORS = [
    (7.00, 4.50),
    (9.50, 8.20),
    (5.60, 6.20),
    (6.90, 6.00),
    (8.20, 5.30),
]
REGION_FACTOR_VALUES = {{"very_low": 0.70, "low": 0.80, "low_medium": 0.90, "medium": 1.00, "medium_high": 1.10, "high": 1.20, "very_high": 1.35}}
SITE_TYPE_MULTIPLIER = {{"urban": 1.15, "suburban": 1.00, "rural": 0.70}}
TECH_MULTIPLIER = {{"4G": 1.00, "5G": 1.15}}
rng = np.random.default_rng(NETWORK_CONFIG["seed"])
CELL_TARGET_UTILIZATION_BY_SITE_TYPE = {{"urban": 0.65, "suburban": 0.50, "rural": 0.35}}
CELL_CAPACITY_MBPS = {{
    ("4G", "700MHz"): 75, ("4G", "1800MHz"): 100, ("4G", "2100MHz"): 110, ("4G", "2600MHz"): 150,
    ("5G", "2100MHz"): 300, ("5G", "2600MHz"): 500, ("5G", "3500MHz"): 800,
}}
TRANSPORT_MEDIUM_PROFILES = [(15, "fiber_metro", 20.0), (60, "fiber_regional", 10.0), (np.inf, "microwave", 2.5)]

{get_code(topo_fns)}
"""

cap_code = f"""import pandas as pd
import numpy as np

{get_code(cap_fns)}
"""

ops_code = f"""import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

TIME_CONFIG = {{"n_days": 90, "interval_minutes": 15}}
N_STEPS = TIME_CONFIG["n_days"] * 24 * 60 // TIME_CONFIG["interval_minutes"]
SEVERITY_LEVELS = {{"low": 0.30, "medium": 0.60, "high": 1.00}}
SEVERITY_WEIGHTS = [0.50, 0.35, 0.15]
FAULT_TYPE_CONFIG = {{
    "ran_degradation":       {{"weight": 0.30, "entity_type": "cell",        "duration_hr": (1, 6)}},
    "cell_outage":           {{"weight": 0.15, "entity_type": "site",        "duration_hr": (0.5, 4)}},
    "backhaul_degradation":  {{"weight": 0.20, "entity_type": "site",        "duration_hr": (1, 8)}},
    "transport_link_failure":{{"weight": 0.15, "entity_type": "mesh_link",   "duration_hr": (1, 12)}},
    "aggregation_failure":   {{"weight": 0.10, "entity_type": "aggregation", "duration_hr": (2, 10)}},
    "core_degradation":      {{"weight": 0.10, "entity_type": "core",        "duration_hr": (1, 6)}},
}}
LINK_PROFILES = {{
    "backhaul": (3, 0.5),
    "fiber_metro": (1, 0.2),
    "fiber_regional": (2, 0.4),
    "microwave": (4, 1.2),
    "core_uplink": (2, 0.3),
}}
RADIO_BASELINE_SINR = {{
    ("4G", "urban"): 12, ("4G", "suburban"): 15, ("4G", "rural"): 18,
    ("5G", "urban"): 18, ("5G", "suburban"): 22, ("5G", "rural"): 25,
}}
MBPS_PER_ACTIVE_USER = {{"4G": 3.0, "5G": 8.0}}
ALARM_RULES = [
    ("ran", "prb_utilization_pct", ">", 90, "<", 75, 4, "high_prb_utilization"),
    ("ran", "availability_pct", "<", 50, ">", 90, 1, "cell_unavailable"),
    ("transport", "packet_loss_pct", ">", 5, "<", 2, 4, "packet_loss_threshold"),
    ("transport", "latency_ms", ">", 15, "<", 8, 4, "latency_threshold"),
    ("transport", "link_status", "==", "DOWN", "!=", "DOWN", 1, "link_down"),
    ("core", "cpu_utilization_pct", ">", 85, "<", 70, 4, "core_resource_exhaustion"),
]
FALSE_ALARM_RATE = 0.03
TRANSPORT_MEDIUM_PROFILES = [(15, "fiber_metro", 20.0), (60, "fiber_regional", 10.0), (np.inf, "microwave", 2.5)]
REGION_PEAK_SHIFT_HR = {{"Lagos": 0.5, "West": 0.0, "North": -0.5, "South": 0.2, "East": -0.3}}
SITE_TYPE_PEAK_WIDTH = {{"urban": 1.6, "suburban": 2.2, "rural": 3.0}}
WEEKDAY_MULT, WEEKEND_MULT = 1.0, 0.86
TREND_GROWTH_OVER_90D = 0.05
DEMAND_NOISE_STD = 0.03
time_index = pd.date_range(start="2025-01-01", periods=N_STEPS, freq=f"{{TIME_CONFIG['interval_minutes']}}min")
hours = time_index.hour + time_index.minute / 60
day_index = (time_index - time_index[0]).days
is_weekend = time_index.dayofweek.isin([5, 6])

{get_code(ops_fns)}
"""

labels_code = f"""import pandas as pd
import numpy as np

{get_code(labels_fns)}
"""

with open('simulator/topology.py', 'w', encoding='utf-8') as f:
    f.write(topo_code)

with open('simulator/capacity.py', 'w', encoding='utf-8') as f:
    f.write(cap_code)

with open('simulator/operational.py', 'w', encoding='utf-8') as f:
    f.write(ops_code)

with open('ml_pipeline/labels.py', 'w', encoding='utf-8') as f:
    f.write(labels_code)

print("Extraction completed!")
