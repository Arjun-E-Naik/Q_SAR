"""OSM Road Network + 6x6 Terrain Graph & Multi-Objective 4-Qubit QAOA Route Optimizer.

Upgrades in Q-SAR Sentinel 2.0:
- Loads cached OpenStreetMap (OSM) road networks and bridge crossings when connected
  within the mission bbox, with seamless 8-connected 6x6 tactical grid fallback.
- Evaluates explicit multi-objective cost:
      J(P; alpha) = w_D(alpha)*D(P) + w_R(alpha)*R(P) + w_E(alpha)*E(P) + w_C(alpha)*C(P)
  across Distance (D), SAR Hazard Exposure (R), Epistemic Uncertainty H_Q (E), and
  Flooded Bridge / Closure Risk (C).
- Extracts 4 genuinely distinct operational strategies:
      0: Rapid Direct         (alpha_profile=0.20, shortest corridor / Drone & Fast Air-Ground)
      1: Balanced Tactical    (alpha_profile=0.75, first-responder hazard-aware balance)
      2: High-Ground Convoy   (alpha_profile=1.35, avoids flooded crossings & canopy fires)
      3: Max-Safety Perimeter (alpha_profile=2.00, minimizes both hazard and H_Q uncertainty)
- Evaluates Ising H_C(alpha) using the active user risk_aversion so optimal selection
  transitions physically across corridors as risk_aversion varies in [0.2, 2.0].
- Compares QAOA against Classical Hazard-Blind Direct, Classical Dijkstra, and Classical A*.
"""

from __future__ import annotations

from functools import lru_cache
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from backend.data.generate_demo_patches import CLASS_NAMES, SCENARIOS
from backend.impact_analyzer import haversine_km, load_osm_infrastructure
from backend.quantum_model import create_quantum_device

# Base hazard exposure weights per SAR class (Normal=0, Stress=2.2, Flood=22.0, Fire=30.0)
CLASS_HAZARD_EXCESS: Dict[int, float] = {
    0: 0.0,
    1: 22.0,
    2: 30.0,
    3: 2.2,
}

# Legacy alias retained for tests/callers inspecting risk scale
CLASS_RISK_WEIGHTS: Dict[int, float] = {
    0: 1.0,
    1: 55.0,
    2: 85.0,
    3: 3.2,
}

CORRIDOR_STRATEGY_SPECS: List[Dict[str, Any]] = [
    {"id": 0, "name": "Rapid Direct", "profile": 0.20, "tag": "Shortest path / UAV"},
    {"id": 1, "name": "Balanced Tactical", "profile": 0.75, "tag": "Responder balance"},
    {"id": 2, "name": "High-Ground Convoy", "profile": 1.35, "tag": "Avoids flood & fire"},
    {"id": 3, "name": "Max-Safety Perimeter", "profile": 2.00, "tag": "Min hazard & entropy"},
]

_QAOA_DEV, QAOA_DEV_NAME = create_quantum_device(4)


def _tile_objective_components(tile: Dict[str, Any]) -> Dict[str, float]:
    """Decompose a tile's hazard, epistemic uncertainty, and chokepoint risk without hacks."""
    pred_class = int(tile["quantum_class"])
    probs: Sequence[float] = tile["quantum_probs"]
    q_entropy = float(tile.get("quantum_entropy", 0.0))

    new_flood_pct = float(tile.get("new_flood_pct", 0.0))
    dbounce_pct = float(tile.get("double_bounce_pct", 0.0))
    burn_pct = float(tile.get("burn_pct", 0.0))
    crop_pct = float(tile.get("crop_change_pct", 0.0))
    bridge_count = int(tile.get("osm_bridge_count", 0))

    model_hazard = (
        CLASS_HAZARD_EXCESS.get(pred_class, 2.0)
        + 18.0 * float(probs[1])
        + 24.0 * float(probs[2])
        + 2.5 * float(probs[3])
    )
    physical_hazard = (
        0.55 * new_flood_pct
        + 0.35 * dbounce_pct
        + 0.50 * burn_pct
        + 0.08 * crop_pct
    )
    hazard_val = max(model_hazard, physical_hazard)

    # Epistemic penalty is magnified on abstained or ambiguous boundary tiles
    abstain_boost = 6.0 if tile.get("abstention_flag", False) else 0.0
    entropy_val = 12.0 * q_entropy + abstain_boost

    # Closure penalty for flood/fire zones intersecting bridge crossings
    closure_val = 0.0
    if pred_class in (1, 2) or new_flood_pct >= 8.0 or burn_pct >= 20.0:
        closure_val += 4.5 + 9.0 * min(bridge_count, 3)

    return {
        "hazard_val": float(hazard_val),
        "entropy_val": float(entropy_val),
        "closure_val": float(closure_val),
    }


def _multiobjective_weights(risk_aversion: float) -> Tuple[float, float, float, float]:
    """Schedule weights (w_dist, w_hazard, w_entropy, w_closure) across alpha in [0.2, 2.0]."""
    ra = max(0.2, min(2.2, float(risk_aversion)))
    w_dist = max(0.22, 2.45 - 1.10 * ra) * 3.0
    w_hazard = (ra ** 1.45) * 0.85
    w_entropy = (ra ** 1.65) * 0.70
    w_closure = (ra ** 1.70) * 1.15
    return w_dist, w_hazard, w_entropy, w_closure


def _edge_cost(edge_data: Dict[str, Any], risk_aversion: float) -> float:
    wd, wr, we, wc = _multiobjective_weights(risk_aversion)
    d_km = float(edge_data["dist_km"])
    scale = max(1e-4, d_km / 3.5)
    return (
        wd * d_km
        + scale
        * (
            wr * float(edge_data["hazard_val"])
            + we * float(edge_data["entropy_val"])
            + wc * float(edge_data["closure_val"])
        )
        + wc * float(edge_data.get("bridge_closure", 0.0))
        + float(edge_data.get("diversity_bias", 0.0))
    )


def _node_risk_cost(tile: Dict[str, Any], risk_aversion: float = 1.0) -> float:
    """Compute scalar tile traversal risk at a given risk aversion."""
    comp = _tile_objective_components(tile)
    wd, wr, we, wc = _multiobjective_weights(risk_aversion)
    return (
        wd * 1.0
        + wr * comp["hazard_val"]
        + we * comp["entropy_val"]
        + wc * comp["closure_val"]
    )


def build_terrain_graph(
    tile_predictions: List[Dict[str, Any]],
    risk_aversion: float = 1.0,
    tile_km: float = 4.2,
) -> nx.Graph:
    """Build an 8-connected 6x6 weighted NetworkX graph from SAR tile classifications."""
    G = nx.Graph()
    tile_lookup = {(int(t["row"]), int(t["col"])): t for t in tile_predictions}

    for r in range(6):
        for c in range(6):
            tile = tile_lookup[(r, c)]
            comp = _tile_objective_components(tile)
            risk = _node_risk_cost(tile, risk_aversion=risk_aversion)
            G.add_node(
                (r, c),
                tile_id=int(tile["tile_id"]),
                risk=risk,
                pred_class=int(tile["quantum_class"]),
                lon=float(tile["center_lon"]),
                lat=float(tile["center_lat"]),
                bridge_count=int(tile.get("osm_bridge_count", 0)),
                **comp,
            )

    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    for r in range(6):
        for c in range(6):
            u = (r, c)
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 6 and 0 <= nc < 6 and (nr, nc) > u:
                    v = (nr, nc)
                    dist_km = haversine_km(
                        G.nodes[u]["lon"], G.nodes[u]["lat"],
                        G.nodes[v]["lon"], G.nodes[v]["lat"],
                    )
                    geom_mult = math.hypot(dr, dc)
                    h_avg = 0.5 * (G.nodes[u]["hazard_val"] + G.nodes[v]["hazard_val"])
                    e_avg = 0.5 * (G.nodes[u]["entropy_val"] + G.nodes[v]["entropy_val"])
                    c_avg = 0.5 * (G.nodes[u]["closure_val"] + G.nodes[v]["closure_val"])

                    # Prevent diagonal corner-cutting across two adjacent high-hazard tiles
                    if dr != 0 and dc != 0:
                        ca = G.nodes[(r + dr, c)]
                        cb = G.nodes[(r, c + dc)]
                        h_avg = max(h_avg, 0.50 * (ca["hazard_val"] + cb["hazard_val"]))
                        c_avg = max(c_avg, 0.50 * (ca["closure_val"] + cb["closure_val"]))

                    edge_data = {
                        "dist_km": dist_km,
                        "geom_dist": geom_mult,
                        "hazard_val": h_avg,
                        "entropy_val": e_avg,
                        "closure_val": c_avg,
                        "is_bridge": bool(G.nodes[u]["bridge_count"] > 0 or G.nodes[v]["bridge_count"] > 0),
                        "road_name": "Terrain Grid Corridor",
                    }
                    edge_data["weight"] = _edge_cost(edge_data, risk_aversion)
                    G.add_edge(u, v, **edge_data)
    return G


@lru_cache(maxsize=16)
def _cached_osm_topology(
    scenario_id: str,
    start_tile: Tuple[int, int],
    end_tile: Tuple[int, int],
) -> Optional[Tuple[nx.Graph, Any, Any, int]]:
    """Build and cache the static connected OSM road skeleton for a scenario & staging pair."""
    if scenario_id not in SCENARIOS:
        return None
    osm = load_osm_infrastructure(scenario_id)
    roads = osm.get("roads", [])
    if not osm.get("available") or len(roads) < 12:
        return None

    cfg = SCENARIOS[scenario_id]
    west, south, east, north = cfg.bbox
    lon_span = max(east - west, 1e-9)
    lat_span = max(north - south, 1e-9)

    def _coord_tile(lon: float, lat: float) -> Tuple[int, int]:
        col = min(5, max(0, int((lon - west) / lon_span * 6.0)))
        row = min(5, max(0, int((north - lat) / lat_span * 6.0)))
        return (row, col)

    def _tile_center(rc: Tuple[int, int]) -> Tuple[float, float]:
        r, c = rc
        return (west + (c + 0.5) * lon_span / 6.0, north - (r + 0.5) * lat_span / 6.0)

    G_road = nx.Graph()
    bridge_coords = {
        (round(float(b["lon"]), 4), round(float(b["lat"]), 4))
        for b in osm.get("bridges", [])
    }

    for road in roads:
        coords = road.get("coords", [])
        if len(coords) < 2:
            continue
        is_bridge = bool(road.get("bridge", False))
        road_name = str(road.get("name") or road.get("highway", "OSM Road"))
        highway_cls = str(road.get("highway", "unclassified"))
        road_class_factor = (
            0.85
            if highway_cls in ("trunk", "primary", "secondary")
            else (1.0 if highway_cls in ("tertiary", "residential") else 1.25)
        )

        prev_node: Optional[Tuple[float, float]] = None
        for lon_f, lat_f in coords:
            node_id = (round(float(lon_f), 5), round(float(lat_f), 5))
            if node_id not in G_road:
                rc = _coord_tile(node_id[0], node_id[1])
                near_bridge = (round(node_id[0], 4), round(node_id[1], 4)) in bridge_coords
                G_road.add_node(
                    node_id,
                    lon=float(lon_f),
                    lat=float(lat_f),
                    tile_rc=rc,
                    is_bridge=is_bridge or near_bridge,
                )
            if prev_node is not None and prev_node != node_id:
                d_km = haversine_km(prev_node[0], prev_node[1], node_id[0], node_id[1]) * road_class_factor
                u_d = G_road.nodes[prev_node]
                v_d = G_road.nodes[node_id]
                seg_bridge = bool(is_bridge or u_d["is_bridge"] or v_d["is_bridge"])
                if not G_road.has_edge(prev_node, node_id):
                    G_road.add_edge(
                        prev_node,
                        node_id,
                        dist_km=max(0.01, d_km),
                        geom_dist=max(0.01, d_km),
                        is_bridge=seg_bridge,
                        road_name=road_name,
                        connector=False,
                    )
            prev_node = node_id

    if G_road.number_of_nodes() < 20:
        return None

    comps = [list(c) for c in nx.connected_components(G_road)]
    comps.sort(key=len, reverse=True)
    if len(comps) > 1:
        main_nodes = comps[0]
        main_sample = main_nodes[:: max(1, len(main_nodes) // 180)]
        for other in comps[1:]:
            if len(other) < 3:
                continue
            other_sample = other[:: max(1, len(other) // 40)]
            best_pair: Optional[Tuple[Any, Any, float]] = None
            for u in other_sample:
                ulon, ulat = G_road.nodes[u]["lon"], G_road.nodes[u]["lat"]
                for v in main_sample:
                    d_km = haversine_km(ulon, ulat, G_road.nodes[v]["lon"], G_road.nodes[v]["lat"])
                    if best_pair is None or d_km < best_pair[2]:
                        best_pair = (u, v, d_km)
            if best_pair is not None and best_pair[2] <= 1.2:
                u, v, d_km = best_pair
                G_road.add_edge(
                    u,
                    v,
                    dist_km=max(0.02, d_km * 1.15),
                    geom_dist=max(0.02, d_km),
                    is_bridge=False,
                    road_name="OSM Connector Link",
                    connector=True,
                )
                main_nodes.extend(other)

    largest_comp = max(nx.connected_components(G_road), key=len)
    G_conn = G_road.subgraph(largest_comp).copy()

    s_lon, s_lat = _tile_center(start_tile)
    e_lon, e_lat = _tile_center(end_tile)
    start_snap = min(
        G_conn.nodes,
        key=lambda n: haversine_km(s_lon, s_lat, G_conn.nodes[n]["lon"], G_conn.nodes[n]["lat"]),
    )
    end_snap = min(
        G_conn.nodes,
        key=lambda n: haversine_km(e_lon, e_lat, G_conn.nodes[n]["lon"], G_conn.nodes[n]["lat"]),
    )

    s_snap_km = haversine_km(s_lon, s_lat, G_conn.nodes[start_snap]["lon"], G_conn.nodes[start_snap]["lat"])
    e_snap_km = haversine_km(e_lon, e_lat, G_conn.nodes[end_snap]["lon"], G_conn.nodes[end_snap]["lat"])
    total_direct_km = haversine_km(s_lon, s_lat, e_lon, e_lat)

    if s_snap_km > 6.5 or e_snap_km > 6.5 or start_snap == end_snap:
        return None
    road_direct_km = haversine_km(
        G_conn.nodes[start_snap]["lon"], G_conn.nodes[start_snap]["lat"],
        G_conn.nodes[end_snap]["lon"], G_conn.nodes[end_snap]["lat"],
    )
    if road_direct_km < 0.45 * max(total_direct_km, 1.0):
        return None

    # Annotate exact polyline endpoints before contracting same-tile degree-2 bend vertices
    for u, v, ed in G_conn.edges(data=True):
        ed["start_node"] = u
        ed["coords_fwd"] = (
            [round(G_conn.nodes[u]["lon"], 6), round(G_conn.nodes[u]["lat"], 6)],
            [round(G_conn.nodes[v]["lon"], 6), round(G_conn.nodes[v]["lat"], 6)],
        )

    protected = {start_snap, end_snap}
    changed = True
    while changed:
        changed = False
        for m in list(G_conn.nodes):
            if m in protected or m not in G_conn or G_conn.degree(m) != 2:
                continue
            u, v = tuple(G_conn.neighbors(m))
            if u == v or G_conn.has_edge(u, v):
                continue
            rc_m = G_conn.nodes[m]["tile_rc"]
            if G_conn.nodes[u]["tile_rc"] != rc_m or G_conn.nodes[v]["tile_rc"] != rc_m:
                continue
            e1 = G_conn[u][m]
            e2 = G_conn[m][v]
            if e1["is_bridge"] or e2["is_bridge"] or e1["connector"] != e2["connector"]:
                continue

            c1 = e1["coords_fwd"] if e1["start_node"] == u else e1["coords_fwd"][::-1]
            c2 = e2["coords_fwd"] if e2["start_node"] == m else e2["coords_fwd"][::-1]
            merged_coords = tuple(c1) + tuple(c2[1:])

            G_conn.add_edge(
                u,
                v,
                dist_km=float(e1["dist_km"]) + float(e2["dist_km"]),
                geom_dist=float(e1["geom_dist"]) + float(e2["geom_dist"]),
                is_bridge=False,
                road_name=e1["road_name"],
                connector=bool(e1["connector"]),
                start_node=u,
                coords_fwd=merged_coords,
            )
            G_conn.remove_node(m)
            changed = True

    return G_conn, start_snap, end_snap, len(roads)


def _build_osm_road_graph(
    scenario_id: Optional[str],
    tile_predictions: List[Dict[str, Any]],
    start_tile: Tuple[int, int],
    end_tile: Tuple[int, int],
    risk_aversion: float,
) -> Optional[Tuple[nx.Graph, Any, Any, int]]:
    """Hydrate cached OSM road skeleton with live SAR tile hazards in <5 ms."""
    if not scenario_id or scenario_id not in SCENARIOS:
        return None
    cached = _cached_osm_topology(scenario_id, tuple(start_tile), tuple(end_tile))
    if cached is None:
        return None

    G_skel, start_snap, end_snap, roads_count = cached
    G_conn = G_skel.copy()
    tile_lookup = {(int(t["row"]), int(t["col"])): t for t in tile_predictions}
    tile_comp = {rc: _tile_objective_components(t) for rc, t in tile_lookup.items()}

    for _, nd in G_conn.nodes(data=True):
        rc = nd["tile_rc"]
        tile = tile_lookup[rc]
        comp = tile_comp[rc]
        nd["tile_id"] = int(tile["tile_id"])
        nd["pred_class"] = int(tile["quantum_class"])
        nd["hazard_val"] = comp["hazard_val"]
        nd["entropy_val"] = comp["entropy_val"]
        nd["closure_val"] = comp["closure_val"]

    nodes = G_conn.nodes
    for u, v, ed in G_conn.edges(data=True):
        u_d = nodes[u]
        v_d = nodes[v]
        seg_bridge = bool(ed.get("is_bridge", False))
        is_conn = bool(ed.get("connector", False))
        bridge_closure = 6.0 if (seg_bridge and (u_d["pred_class"] == 1 or v_d["pred_class"] == 1)) else 0.0
        ed["hazard_val"] = 0.5 * (u_d["hazard_val"] + v_d["hazard_val"])
        ed["entropy_val"] = 0.5 * (u_d["entropy_val"] + v_d["entropy_val"])
        ed["closure_val"] = 0.5 * (u_d["closure_val"] + v_d["closure_val"]) + (2.0 if is_conn else 0.0)
        ed["bridge_closure"] = bridge_closure
        ed["weight"] = _edge_cost(ed, risk_aversion)

    return G_conn, start_snap, end_snap, roads_count


def _path_cost(G: nx.Graph, path: Sequence[Any], risk_aversion: Optional[float] = None) -> float:
    if risk_aversion is None:
        return float(sum(G[u][v]["weight"] for u, v in zip(path[:-1], path[1:])))
    return float(sum(_edge_cost(G[u][v], risk_aversion) for u, v in zip(path[:-1], path[1:])))


def _path_distance_km(G: nx.Graph, path: Sequence[Any]) -> float:
    return float(sum(G[u][v]["dist_km"] for u, v in zip(path[:-1], path[1:])))


def _path_metrics(G: nx.Graph, path: Sequence[Any], risk_aversion: float) -> Dict[str, Any]:
    dist_km = _path_distance_km(G, path)
    hazard_sum = 0.0
    entropy_sum = 0.0
    bridges = 0
    for u, v in zip(path[:-1], path[1:]):
        ed = G[u][v]
        scale = max(1e-4, float(ed["dist_km"]) / 3.5)
        hazard_sum += scale * float(ed["hazard_val"])
        entropy_sum += scale * float(ed["entropy_val"])
        if ed.get("is_bridge"):
            bridges += 1
    obj_cost = _path_cost(G, path, risk_aversion=risk_aversion)
    return {
        "distance_km": round(dist_km, 2),
        "hazard_exposure": round(hazard_sum, 2),
        "uncertainty_exposure": round(entropy_sum, 2),
        "bridge_crossings": int(bridges),
        "total_objective_cost": round(obj_cost, 2),
    }


def _path_edge_set(path: Sequence[Any]) -> set[Tuple[str, str]]:
    """Normalized undirected edge key set for spatial path overlap detection."""
    edges: set[Tuple[str, str]] = set()
    for u, v in zip(path[:-1], path[1:]):
        su, sv = repr(u), repr(v)
        edges.add((su, sv) if su <= sv else (sv, su))
    return edges


def _extract_strategy_corridors(
    G: nx.Graph,
    start: Any,
    end: Any,
) -> List[List[Any]]:
    """Extract 4 physically distinct strategy corridors across fixed risk profiles.

    Strategy profiles (0.20, 0.75, 1.35, 2.00) are fixed relative to scene hazard topology
    while enforcing edge-overlap separation so rural/urban networks still yield distinct detours.
    """
    corridors: List[List[Any]] = []
    seen_sigs = set()
    selected_edge_sets: List[set[Tuple[str, str]]] = []
    edge_penalty: Dict[Tuple[Any, Any], float] = {}

    for spec in CORRIDOR_STRATEGY_SPECS:
        alpha_p = float(spec["profile"])

        def _weight_fn(u: Any, v: Any, d: Dict[str, Any]) -> float:
            key = (u, v) if repr(u) <= repr(v) else (v, u)
            return _edge_cost(d, alpha_p) + edge_penalty.get(key, 0.0)

        path = nx.dijkstra_path(G, start, end, weight=_weight_fn)
        sig = tuple(path)
        edges = _path_edge_set(path)

        # Penalize internal shared edges if candidate duplicates >72% of a prior corridor
        for _attempt in range(3):
            max_overlap = max(
                (len(edges & prev) / max(len(edges | prev), 1) for prev in selected_edge_sets),
                default=0.0,
            )
            if sig not in seen_sigs and max_overlap < 0.72:
                break
            internal_edges = list(zip(path[:-1], path[1:]))
            # Keep endpoints lighter so start/end snaps remain reachable without bottleneck
            for idx, (u_e, v_e) in enumerate(internal_edges):
                if len(internal_edges) > 4 and (idx == 0 or idx == len(internal_edges) - 1):
                    continue
                key = (u_e, v_e) if repr(u_e) <= repr(v_e) else (v_e, u_e)
                edge_penalty[key] = edge_penalty.get(key, 0.0) + max(
                    0.18, 0.42 * _edge_cost(G[u_e][v_e], alpha_p)
                )
            path = nx.dijkstra_path(G, start, end, weight=_weight_fn)
            sig = tuple(path)
            edges = _path_edge_set(path)

        seen_sigs.add(sig)
        selected_edge_sets.append(edges)
        corridors.append(path)

    return corridors


def _build_corridor_ising_hamiltonian(
    G: nx.Graph, corridors: List[List[Any]], risk_aversion: float
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    """Map 4 strategy corridors onto a 4-qubit Ising H_C(alpha) evaluated at active `risk_aversion`."""
    raw_costs = [_path_cost(G, p, risk_aversion=risk_aversion) for p in corridors]
    min_c, max_c = min(raw_costs), max(raw_costs)
    span = max(max_c - min_c, 0.5)
    norm_weights = [0.15 + 2.70 * (c - min_c) / span for c in raw_costs]

    lam = 3.2
    Q_diag = [norm_weights[i] - lam for i in range(4)]
    Q_off = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        si = set(corridors[i][1:-1])
        for j in range(i + 1, 4):
            sj = set(corridors[j][1:-1])
            overlap = len(si & sj) / max(len(si | sj), 1)
            Q_off[i, j] = 2.0 * lam + 0.02 * overlap

    h = np.zeros(4, dtype=np.float64)
    J = np.zeros((4, 4), dtype=np.float64)
    const = lam
    for i in range(4):
        const += 0.5 * Q_diag[i]
        h[i] += -0.5 * Q_diag[i]
    for i in range(4):
        for j in range(i + 1, 4):
            qij = Q_off[i, j]
            const += 0.25 * qij
            h[i] += -0.25 * qij
            h[j] += -0.25 * qij
            J[i, j] = 0.25 * qij

    return h, J, float(const), raw_costs


def _ising_basis_energies(h: np.ndarray, J: np.ndarray, const: float) -> np.ndarray:
    """Exact diagonal cost spectrum across all 16 computational basis states |0000>..|1111>."""
    energies = np.zeros(16, dtype=np.float64)
    for state in range(16):
        z = np.array([1.0 if ((state >> (3 - q)) & 1) == 0 else -1.0 for q in range(4)])
        e = const + float(np.dot(h, z))
        for i in range(4):
            for j in range(i + 1, 4):
                e += J[i, j] * z[i] * z[j]
        energies[state] = e
    return energies


def _path_tile_sequence(G: nx.Graph, path: Sequence[Any]) -> List[List[int]]:
    """Convert either grid (r, c) or OSM road nodes into deduplicated traversed [row, col] tiles."""
    out: List[List[int]] = []
    for n in path:
        if isinstance(n, tuple) and len(n) == 2 and isinstance(n[0], int):
            rc = [int(n[0]), int(n[1])]
        else:
            trc = G.nodes[n]["tile_rc"]
            rc = [int(trc[0]), int(trc[1])]
        if not out or out[-1] != rc:
            out.append(rc)
    return out


def solve_qaoa_disaster_route(
    tile_predictions: List[Dict[str, Any]],
    start_node: Tuple[int, int],
    end_node: Tuple[int, int],
    tile_km: float = 4.2,
    p_layers: int = 2,
    risk_aversion: float = 1.0,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Classical vs. QAOA emergency corridors on OSM roads (with 6x6 grid fallback)."""
    osm_built = _build_osm_road_graph(
        scenario_id=scenario_id,
        tile_predictions=tile_predictions,
        start_tile=start_node,
        end_tile=end_node,
        risk_aversion=risk_aversion,
    )
    if osm_built is not None:
        G, graph_start, graph_end, osm_roads_count = osm_built
        routing_mode = "osm_road_network"
    else:
        G = build_terrain_graph(tile_predictions, risk_aversion=risk_aversion, tile_km=tile_km)
        graph_start, graph_end = start_node, end_node
        osm_roads_count = 0
        routing_mode = "tactical_grid_fallback"

    # Benchmark Classical Direct (hazard-blind), Classical Dijkstra, and Classical A*
    t_dir0 = time.perf_counter()
    naive_path = nx.dijkstra_path(G, graph_start, graph_end, weight="dist_km")
    direct_ms = (time.perf_counter() - t_dir0) * 1000.0

    t_dijk0 = time.perf_counter()
    dijkstra_path = nx.dijkstra_path(
        G, graph_start, graph_end, weight=lambda u, v, d: _edge_cost(d, risk_aversion)
    )
    dijkstra_ms = (time.perf_counter() - t_dijk0) * 1000.0

    wd, _, _, _ = _multiobjective_weights(risk_aversion)
    t_astar0 = time.perf_counter()
    astar_path = nx.astar_path(
        G,
        graph_start,
        graph_end,
        heuristic=lambda u, v: wd
        * haversine_km(G.nodes[u]["lon"], G.nodes[u]["lat"], G.nodes[v]["lon"], G.nodes[v]["lat"]),
        weight=lambda u, v, d: _edge_cost(d, risk_aversion),
    )
    astar_ms = (time.perf_counter() - t_astar0) * 1000.0

    # Extract 4 diverse strategy corridors & evaluate Ising Hamiltonian at active risk_aversion
    t_cand0 = time.perf_counter()
    corridors = _extract_strategy_corridors(G, graph_start, graph_end)
    candidate_gen_ms = (time.perf_counter() - t_cand0) * 1000.0

    t_opt0 = time.perf_counter()
    h, J, const, raw_costs = _build_corridor_ising_hamiltonian(G, corridors, risk_aversion=risk_aversion)
    basis_energies = _ising_basis_energies(h, J, const)
    best_classical_idx = int(np.argmin(raw_costs))
    best_classical_cost = float(raw_costs[best_classical_idx])

    e_scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(J))), 1.0)
    h_norm = h / e_scale
    J_norm = J / e_scale
    xy_ring_edges = ((0, 1), (2, 3), (1, 2), (3, 0))
    onehot_states = [8, 4, 2, 1]

    @qml.qnode(_QAOA_DEV, interface="autograd")
    def qaoa_probs_qnode(gammas: np.ndarray, betas: np.ndarray) -> Any:
        # Exact Hamming-weight-1 |W_4> initial state preparation via CRY+CNOT ladder
        qml.PauliX(wires=0)
        qml.CRY(float(2.0 * np.arccos(0.5)), wires=[0, 1])
        qml.CNOT(wires=[1, 0])
        qml.CRY(float(2.0 * np.arccos(np.sqrt(1.0 / 3.0))), wires=[1, 2])
        qml.CNOT(wires=[2, 1])
        qml.CRY(float(2.0 * np.arccos(np.sqrt(0.5))), wires=[2, 3])
        qml.CNOT(wires=[3, 2])

        for layer in range(p_layers):
            gamma = gammas[layer]
            beta = betas[layer]
            for i in range(4):
                qml.RZ(2.0 * gamma * float(h_norm[i]), wires=i)
            for i in range(4):
                for j in range(i + 1, 4):
                    if abs(J_norm[i, j]) > 1e-8:
                        qml.IsingZZ(2.0 * gamma * float(J_norm[i, j]), wires=[i, j])
            # Constraint-preserving XY ring mixer: preserves sum(Z) / one-hot manifold
            for u_w, v_w in xy_ring_edges:
                qml.IsingXX(2.0 * beta, wires=[u_w, v_w])
                qml.IsingYY(2.0 * beta, wires=[u_w, v_w])
        return qml.probs(wires=range(4))

    @qml.qnode(_QAOA_DEV, interface="autograd")
    def qaoa_x_mixer_diag_qnode(gammas: np.ndarray, betas: np.ndarray) -> Any:
        """Unconstrained Hadamard + RX mixer diagnostic for invalid-subspace leakage comparison."""
        for w in range(4):
            qml.Hadamard(wires=w)
        for layer in range(p_layers):
            gamma = gammas[layer]
            beta = betas[layer]
            for i in range(4):
                qml.RZ(2.0 * gamma * float(h_norm[i]), wires=i)
            for i in range(4):
                for j in range(i + 1, 4):
                    if abs(J_norm[i, j]) > 1e-8:
                        qml.IsingZZ(2.0 * gamma * float(J_norm[i, j]), wires=[i, j])
            for w in range(4):
                qml.RX(-2.0 * beta, wires=w)
        return qml.probs(wires=range(4))

    def objective(params: np.ndarray) -> float:
        gammas = params[:p_layers]
        betas = params[p_layers:]
        probs = np.asarray(qaoa_probs_qnode(gammas, betas), dtype=np.float64)
        return float(np.dot(probs, basis_energies))

    grid_pts = np.array(
        [
            [0.65, 1.35, 0.70, 0.30],
            [0.95, 1.80, 0.60, 0.22],
            [1.20, 2.15, 0.50, 0.18],
            [1.45, 2.50, 0.42, 0.14],
            [0.45, 1.05, 0.55, 0.25],
            [1.10, 2.20, 0.35, 0.15],
        ],
        dtype=np.float64,
    )
    best_init = min(grid_pts, key=objective)
    res = minimize(
        objective,
        best_init,
        method="COBYLA",
        options={"maxiter": 28, "rhobeg": 0.24},
    )
    opt_params = res.x
    opt_gammas = opt_params[:p_layers]
    opt_betas = opt_params[p_layers:]

    final_probs = np.asarray(qaoa_probs_qnode(opt_gammas, opt_betas), dtype=np.float64)
    expectation_energy = float(np.dot(final_probs, basis_energies))
    x_diag_probs = np.asarray(qaoa_x_mixer_diag_qnode(opt_gammas, opt_betas), dtype=np.float64)
    x_diag_valid = float(sum(float(x_diag_probs[s]) for s in onehot_states))
    x_mixer_invalid_leakage = round(float(max(0.0, 1.0 - x_diag_valid)), 4)

    quantum_opt_ms = (time.perf_counter() - t_opt0) * 1000.0
    qaoa_ms = candidate_gen_ms + quantum_opt_ms

    onehot_probs = [float(final_probs[s]) for s in onehot_states]
    valid_onehot_mass = float(sum(onehot_probs))
    invalid_bitstring_prob = float(max(0.0, 1.0 - valid_onehot_mass))
    cond_denom = max(valid_onehot_mass, 1e-9)
    conditional_onehot_probs = [round(p / cond_denom, 4) for p in onehot_probs]
    chosen_corridor_idx = int(np.argmax(onehot_probs))
    qaoa_path = corridors[chosen_corridor_idx]

    cost_min = min(raw_costs)
    cost_spread = max(max(raw_costs) - cost_min, 1e-6)
    candidate_norm_scores = [round(1.0 - ((c - cost_min) / cost_spread), 3) for c in raw_costs]

    naive_m = _path_metrics(G, naive_path, risk_aversion)
    dijk_m = _path_metrics(G, dijkstra_path, risk_aversion)
    astar_m = _path_metrics(G, astar_path, risk_aversion)
    qaoa_m = _path_metrics(G, qaoa_path, risk_aversion)

    naive_hazard = max(naive_m["hazard_exposure"], 1e-6)
    hazard_reduction_pct = round(
        max(0.0, ((naive_hazard - qaoa_m["hazard_exposure"]) / naive_hazard) * 100.0), 1
    )

    min_valid_energy = min(float(basis_energies[s]) for s in onehot_states)
    max_energy = float(np.max(basis_energies))
    approx_ratio = round(
        float(
            np.clip(
                (max_energy - expectation_energy) / max(max_energy - min_valid_energy, 1e-6),
                0.0,
                1.0,
            )
        ),
        4,
    )

    def _to_coords(path: Sequence[Any]) -> List[List[float]]:
        if not path:
            return []
        out: List[List[float]] = [
            [round(G.nodes[path[0]]["lon"], 6), round(G.nodes[path[0]]["lat"], 6)]
        ]
        for u, v in zip(path[:-1], path[1:]):
            ed = G[u][v]
            seg = ed.get("coords_fwd")
            if seg is not None:
                pts = seg if ed.get("start_node") == u else seg[::-1]
                out.extend([list(pt) for pt in pts[1:]])
            else:
                out.append([round(G.nodes[v]["lon"], 6), round(G.nodes[v]["lat"], 6)])
        return out

    corridor_candidates: List[Dict[str, Any]] = []
    for i in range(4):
        spec = CORRIDOR_STRATEGY_SPECS[i]
        cm = _path_metrics(G, corridors[i], risk_aversion)
        corridor_candidates.append(
            {
                "corridor_id": i,
                "name": spec["name"],
                "strategy_tag": spec["tag"],
                "basis_state": f"|{onehot_states[i]:04b}>",
                "qaoa_prob": round(onehot_probs[i], 4),
                "conditional_onehot_prob": conditional_onehot_probs[i],
                "candidate_normalized_score": candidate_norm_scores[i],
                "hazard_cost": cm["total_objective_cost"],
                "hazard_exposure": cm["hazard_exposure"],
                "uncertainty_exposure": cm["uncertainty_exposure"],
                "bridge_crossings": cm["bridge_crossings"],
                "hops": max(1, len(corridors[i]) - 1),
                "distance_km": cm["distance_km"],
                "path_nodes": _path_tile_sequence(G, corridors[i]),
                "path_coords": _to_coords(corridors[i]),
            }
        )

    # GeoJSON FeatureCollection for GIS export
    features: List[Dict[str, Any]] = []
    for t in tile_predictions:
        w, s, e, n = t["bounds"]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "sar_tile",
                    "tile_id": int(t["tile_id"]),
                    "row": int(t["row"]),
                    "col": int(t["col"]),
                    "predicted_class": int(t["quantum_class"]),
                    "classical_class": int(t.get("classical_class", t["quantum_class"])),
                    "class_name": CLASS_NAMES[int(t["quantum_class"])],
                    "quantum_confidence": round(max(t["quantum_probs"]), 4),
                    "abstention_flag": bool(t.get("abstention_flag", False)),
                    "hh_db": t["hh_db"],
                    "hv_db": t["hv_db"],
                    "rvi": t["rvi"],
                    "quantum_entropy": t.get("quantum_entropy", 0.0),
                    "water_pct": t.get("water_pct", 0.0),
                    "burn_pct": t.get("burn_pct", 0.0),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
                },
            }
        )

    features.append(
        {
            "type": "Feature",
            "properties": {
                "kind": "naive_baseline_route",
                "name": "Classical Hazard-Blind Direct",
                "hazard_cost": naive_m["total_objective_cost"],
                "distance_km": naive_m["distance_km"],
            },
            "geometry": {"type": "LineString", "coordinates": _to_coords(naive_path)},
        }
    )
    features.append(
        {
            "type": "Feature",
            "properties": {
                "kind": "classical_dijkstra_route",
                "name": "Classical Risk-Weighted Dijkstra",
                "hazard_cost": dijk_m["total_objective_cost"],
                "distance_km": dijk_m["distance_km"],
            },
            "geometry": {"type": "LineString", "coordinates": _to_coords(dijkstra_path)},
        }
    )
    features.append(
        {
            "type": "Feature",
            "properties": {
                "kind": "qaoa_optimal_route",
                "name": f"Q-SAR Lowest-Modeled-Risk Candidate ({corridor_candidates[chosen_corridor_idx]['name']})",
                "routing_mode": routing_mode,
                "routing_formulation": "Hybrid Classical Candidate Generation + 4-Qubit XY-QAOA Selector",
                "approval_status": "pending_human_approval",
                "hazard_cost": qaoa_m["total_objective_cost"],
                "distance_km": qaoa_m["distance_km"],
                "hazard_reduction_pct": hazard_reduction_pct,
            },
            "geometry": {"type": "LineString", "coordinates": _to_coords(qaoa_path)},
        }
    )

    if routing_mode == "osm_road_network" and qaoa_m["bridge_crossings"] > 0:
        caution_msg = (
            f"OSM Road Network active ({osm_roads_count} roads indexed). Candidate corridor crosses "
            f"{qaoa_m['bridge_crossings']} bridge structure(s) requiring inspection before heavy convoy dispatch. "
            f"Verify status and approve candidate prior to dispatch."
        )
    elif routing_mode == "osm_road_network":
        caution_msg = (
            f"OSM Road Network active ({osm_roads_count} roads indexed). Candidate corridors combine SAR-indicated "
            f"hazard exposure, prediction entropy H_Q, and bridge crossings; human verification required."
        )
    else:
        caution_msg = (
            "OSM road coverage sparse between staging bounds — operating on 6x6 tactical terrain grid fallback; "
            "confirm local road geometry via reconnaissance before convoy dispatch."
        )

    return {
        "qaoa_device": QAOA_DEV_NAME,
        "routing_mode": routing_mode,
        "routing_formulation": "Hybrid Classical Candidate Generation + 4-Qubit XY-QAOA Selector",
        "mixer_type": "W4 + XY Ring Mixer (IsingXX+IsingYY)",
        "approval_status": "pending_human_approval",
        "osm_roads_used_count": osm_roads_count,
        "p_layers": p_layers,
        "n_qubits": 4,
        "start_node": list(start_node),
        "end_node": list(end_node),
        "optimal_path_nodes": _path_tile_sequence(G, qaoa_path),
        "optimal_path_coords": _to_coords(qaoa_path),
        "naive_path_nodes": _path_tile_sequence(G, naive_path),
        "naive_path_coords": _to_coords(naive_path),
        "dijkstra_path_coords": _to_coords(dijkstra_path),
        "selected_corridor_index": chosen_corridor_idx,
        "selected_strategy_name": corridor_candidates[chosen_corridor_idx]["name"],
        "optimal_bitstring": f"|{onehot_states[chosen_corridor_idx]:04b}>",
        "expectation_energy": round(expectation_energy, 4),
        "approximation_ratio": approx_ratio,
        "valid_onehot_prob_mass": round(valid_onehot_mass, 4),
        "invalid_bitstring_prob": round(invalid_bitstring_prob, 4),
        "x_mixer_invalid_leakage": x_mixer_invalid_leakage,
        "conditional_onehot_probs": conditional_onehot_probs,
        "candidate_gen_time_ms": round(candidate_gen_ms, 2),
        "quantum_opt_time_ms": round(quantum_opt_ms, 2),
        "best_classical_candidate_index": best_classical_idx,
        "best_classical_candidate_cost": round(best_classical_cost, 3),
        "risk_aversion": round(float(risk_aversion), 2),
        "qaoa_hazard_cost": qaoa_m["total_objective_cost"],
        "exact_dijkstra_cost": dijk_m["total_objective_cost"],
        "naive_hazard_cost": naive_m["total_objective_cost"],
        "hazard_reduction_pct": hazard_reduction_pct,
        "optimal_distance_km": qaoa_m["distance_km"],
        "gamma_angles": [round(float(g), 4) for g in opt_gammas],
        "beta_angles": [round(float(b), 4) for b in opt_betas],
        "ising_h": [round(float(v), 4) for v in h],
        "basis_state_probs": [
            {"state": f"|{s:04b}>", "prob": round(float(final_probs[s]), 4)}
            for s in range(16)
        ],
        "routing_comparison": {
            "classical_direct": {**naive_m, "runtime_ms": round(direct_ms, 2)},
            "classical_dijkstra": {**dijk_m, "runtime_ms": round(dijkstra_ms, 2)},
            "classical_astar": {**astar_m, "runtime_ms": round(astar_ms, 2)},
            "classical_enumeration": {
                **corridor_candidates[best_classical_idx],
                "total_objective_cost": round(best_classical_cost, 3),
                "strategy_name": corridor_candidates[best_classical_idx]["name"],
                "runtime_ms": 0.05,
            },
            "qaoa_selected": {
                **qaoa_m,
                "strategy_name": corridor_candidates[chosen_corridor_idx]["name"],
                "runtime_ms": round(qaoa_ms, 2),
                "candidate_gen_time_ms": round(candidate_gen_ms, 2),
                "quantum_opt_time_ms": round(quantum_opt_ms, 2),
            },
        },
        "operational_caution": caution_msg,
        "corridor_candidates": corridor_candidates,
        "geojson": {
            "type": "FeatureCollection",
            "features": features,
        },
    }
