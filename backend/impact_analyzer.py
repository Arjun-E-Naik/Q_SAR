"""Geospatial & Humanitarian Impact Engine for Q-SAR Sentinel 2.0.

Computes:
- Exact cosine-latitude geodesic surface areas (km^2 and agricultural hectares)
- Flooded vs. permanent river vs. flooded-vegetation double-bounce areas
- Cached OpenStreetMap (OSM) road, bridge, waterway, and settlement hazard intersection
- Critical logistics chokepoints & estimated exposed population
- Plain-language operational & radar explainable briefs (scene + tile level)
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

OSM_CACHE_DIR = Path(__file__).resolve().parent / "data" / "osm_cache"
KM_PER_DEG_LAT = 111.132

DEFAULT_SETTLEMENT_POP: Dict[str, int] = {
    "city": 45000,
    "town": 12000,
    "suburb": 6000,
    "village": 1800,
    "hamlet": 250,
}


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle geodesic distance in kilometers between two WGS84 coordinates."""
    r_earth = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi * 0.5) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(dlam * 0.5) ** 2)
    )
    return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


def geodesic_bounds_area_km2(bounds: Sequence[float]) -> float:
    """Compute cosine-latitude geodesic tile area in km^2 from [west, south, east, north]."""
    west, south, east, north = [float(v) for v in bounds]
    mid_lat = 0.5 * (south + north)
    lat_km = abs(north - south) * KM_PER_DEG_LAT
    lon_km = abs(east - west) * KM_PER_DEG_LAT * max(0.05, math.cos(math.radians(mid_lat)))
    return float(lat_km * lon_km)


@lru_cache(maxsize=8)
def load_osm_infrastructure(scenario_id: str) -> Dict[str, Any]:
    """Load cached OpenStreetMap Overpass infrastructure JSON for a scenario."""
    cache_path = OSM_CACHE_DIR / f"{scenario_id}.json"
    if not cache_path.exists():
        return {
            "scenario_id": scenario_id,
            "available": False,
            "source": "Synthetic Fallback Grid (OSM cache absent)",
            "counts": {"roads": 0, "bridges": 0, "settlements": 0, "waterways": 0},
            "roads": [],
            "bridges": [],
            "settlements": [],
            "waterways": [],
        }

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["available"] = True
    return data


def _point_in_bounds(lon: float, lat: float, bounds: Sequence[float]) -> bool:
    west, south, east, north = bounds
    return (west <= lon <= east) and (south <= lat <= north)


def _segment_intersects_bounds(
    p0: Sequence[float], p1: Sequence[float], bounds: Sequence[float]
) -> bool:
    """Fast Liang-Barsky-style 2D bounding-box check for a WGS84 line segment."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    west, south, east, north = bounds
    if max(x0, x1) < west or min(x0, x1) > east or max(y0, y1) < south or min(y0, y1) > north:
        return False
    if _point_in_bounds(x0, y0, bounds) or _point_in_bounds(x1, y1, bounds):
        return True
    for frac in (0.25, 0.5, 0.75):
        xm = x0 + frac * (x1 - x0)
        ym = y0 + frac * (y1 - y0)
        if _point_in_bounds(xm, ym, bounds):
            return True
    return False


def _polyline_length_km(coords: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for i in range(len(coords) - 1):
        total += haversine_km(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
    return total


def _coord_to_tile_id(lon: float, lat: float, bbox: Sequence[float]) -> Optional[int]:
    west, south, east, north = bbox
    if not (west <= lon <= east and south <= lat <= north):
        return None
    col = min(5, max(0, int((lon - west) / max(east - west, 1e-9) * 6.0)))
    row = min(5, max(0, int((north - lat) / max(north - south, 1e-9) * 6.0)))
    return row * 6 + col


def _build_tile_explanation(
    tile: Dict[str, Any],
    co_label: str,
    cr_label: str,
    bridge_count: int,
    road_count: int,
    facility_count: int = 0,
) -> Dict[str, str]:
    """Generate a concise plain-language SAR explanation card for a single tile."""
    cls = int(tile["quantum_class"])
    cls_name = str(tile["quantum_label"])
    c_cls = int(tile.get("classical_class", cls))
    d_co = float(tile.get("delta_hh_db", 0.0))
    d_cr = float(tile.get("delta_hv_db", 0.0))
    d_rvi = float(tile.get("delta_rvi", 0.0))
    new_f = float(tile.get("new_flood_pct", 0.0))
    perm_w = float(tile.get("permanent_water_pct", 0.0))
    dbounce = float(tile.get("double_bounce_pct", 0.0))
    burn = float(tile.get("burn_pct", 0.0))
    crop = float(tile.get("crop_change_pct", 0.0))
    abstain = bool(tile.get("abstention_flag", False))

    if cls == 1:
        if dbounce >= new_f and dbounce >= 15.0:
            what = (
                f"SAR-indicated flooded riparian/paddy signature: {co_label} backscatter surged "
                f"{d_co:+.1f} dB via water-trunk double-bounce ({dbounce:.0f}% sub-blocks)."
            )
        else:
            what = (
                f"SAR-indicated surface inundation: {new_f:.0f}% newly darkened water area "
                f"({perm_w:.0f}% pre-existing channel; {co_label} {d_co:+.1f} dB)."
            )
    elif cls == 2:
        what = (
            f"SAR-indicated canopy structural loss: {cr_label} volume backscatter dropped {d_cr:+.1f} dB "
            f"(ΔRVI {d_rvi:+.2f}, {burn:.0f}% sub-blocks disturbed without standing water)."
        )
    elif cls == 3:
        what = (
            f"SAR-indicated crop/moisture shift: {crop:.0f}% surface sub-blocks shifted >2 dB "
            f"({co_label} {d_co:+.1f} dB, {cr_label} {d_cr:+.1f} dB)."
        )
    else:
        if perm_w >= 12.0:
            what = (
                f"Unchanged baseline river corridor ({perm_w:.0f}% pre-existing water; "
                f"{co_label} {d_co:+.1f} dB, no expansion hazard detected)."
            )
        else:
            what = (
                f"Stable reference terrain ({co_label} {d_co:+.1f} dB, {cr_label} {d_cr:+.1f} dB "
                f"within seasonal baseline)."
            )

    infra_note = ""
    if bridge_count > 0 or road_count > 0 or facility_count > 0:
        infra_note = (
            f" Intersects {road_count} OSM road(s), {bridge_count} bridge(s) requiring inspection"
            + (f", and {facility_count} critical facility(ies)." if facility_count > 0 else ".")
        )

    if abstain:
        conf_note = (
            f"Flagged Unknown/Verify (confidence {tile['quantum_confidence']:.0%}, "
            f"entropy {tile['quantum_entropy']:.2f}) — mixed scattering requires analyst review."
        )
    elif c_cls != cls:
        conf_note = (
            f"Hybrid Q-SAR predicts {cls_name} ({tile['quantum_confidence']:.0%}) vs. "
            f"Classical CNN ({tile.get('classical_label', 'Alt')}); verify boundary contrast."
        )
    else:
        conf_note = (
            f"Hybrid Q-SAR and Classical CNN agree on {cls_name} "
            f"({tile['quantum_confidence']:.0%} confidence)."
        )

    return {
        "what_changed": what + infra_note,
        "supporting_evidence": (
            f"Pre->Post {co_label}: {tile.get('pre_hh_db', tile['hh_db']):.1f}->{tile['hh_db']:.1f} dB ({d_co:+.1f}) | "
            f"{cr_label}: {tile.get('pre_hv_db', tile['hv_db']):.1f}->{tile['hv_db']:.1f} dB ({d_cr:+.1f})"
        ),
        "confidence_note": conf_note,
    }


@lru_cache(maxsize=8)
def _cached_osm_spatial_index(
    scenario_id: str,
    tile_bounds_key: Tuple[Tuple[int, Tuple[float, float, float, float]], ...],
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """Precompute static OSM road/bridge/settlement spatial intersections per scenario."""
    osm = load_osm_infrastructure(scenario_id)
    tile_road_count = [0] * 36
    tile_bridge_count = [0] * 36
    road_records: List[Tuple[float, Tuple[int, ...], str, str, bool]] = []
    total_roads_km = 0.0

    for road in osm.get("roads", []):
        coords = road.get("coords", [])
        if len(coords) < 2:
            continue
        length_km = _polyline_length_km(coords)
        total_roads_km += length_km

        # Fast spatial prefilter via road bounding box vs tile bounds
        rx0 = min(float(p[0]) for p in coords)
        rx1 = max(float(p[0]) for p in coords)
        ry0 = min(float(p[1]) for p in coords)
        ry1 = max(float(p[1]) for p in coords)

        hit_tiles: List[int] = []
        for idx, b in tile_bounds_key:
            west, south, east, north = b
            if rx1 < west or rx0 > east or ry1 < south or ry0 > north:
                continue
            for j in range(len(coords) - 1):
                if _segment_intersects_bounds(coords[j], coords[j + 1], b):
                    hit_tiles.append(idx)
                    tile_road_count[idx] += 1
                    break

        road_records.append(
            (
                length_km,
                tuple(hit_tiles),
                str(road.get("name") or road.get("highway", "Road")),
                str(road.get("highway", "road")),
                bool(road.get("bridge", False)),
            )
        )

    bridge_records: List[Tuple[int, str, str, float, float]] = []
    for bridge in osm.get("bridges", []):
        tid = _coord_to_tile_id(float(bridge["lon"]), float(bridge["lat"]), bbox)
        if tid is None:
            continue
        tile_bridge_count[tid] += 1
        bridge_records.append(
            (
                tid,
                str(bridge.get("name") or "River/Drainage Bridge"),
                str(bridge.get("highway", "road")),
                float(bridge["lon"]),
                float(bridge["lat"]),
            )
        )

    settlement_records: List[Tuple[int, str, str, int, float, float]] = []
    tagged_pop_count = 0
    for st in osm.get("settlements", []):
        tid = _coord_to_tile_id(float(st["lon"]), float(st["lat"]), bbox)
        if tid is None:
            continue
        raw_pop = st.get("population")
        if raw_pop is not None and str(raw_pop).isdigit():
            pop = int(str(raw_pop))
            tagged_pop_count += 1
        else:
            pop = DEFAULT_SETTLEMENT_POP.get(str(st.get("place", "village")), 600)
        settlement_records.append(
            (
                tid,
                str(st.get("name") or "Unnamed Settlement"),
                str(st.get("place", "settlement")),
                pop,
                float(st["lon"]),
                float(st["lat"]),
            )
        )

    tile_facility_count = [0] * 36
    facility_records: List[Tuple[int, str, str, float, float]] = []
    for fac in osm.get("facilities", []):
        tid = _coord_to_tile_id(float(fac["lon"]), float(fac["lat"]), bbox)
        if tid is None:
            continue
        tile_facility_count[tid] += 1
        facility_records.append(
            (
                tid,
                str(fac.get("name") or fac.get("amenity", "Critical Facility")),
                str(fac.get("amenity", "facility")),
                float(fac["lon"]),
                float(fac["lat"]),
            )
        )

    return {
        "osm": osm,
        "tile_road_count": tuple(tile_road_count),
        "tile_bridge_count": tuple(tile_bridge_count),
        "tile_facility_count": tuple(tile_facility_count),
        "road_records": tuple(road_records),
        "total_roads_km": float(total_roads_km),
        "bridge_records": tuple(bridge_records),
        "settlement_records": tuple(settlement_records),
        "facility_records": tuple(facility_records),
        "tagged_pop_count": tagged_pop_count,
    }


def compute_humanitarian_and_explainable_impact(
    scenario_cfg: Any,
    tiles: List[Dict[str, Any]],
    data_provenance: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Aggregate geodesic hazard areas, OSM exposure metrics, and operational briefs."""
    scenario_id = scenario_cfg.scenario_id
    bbox = tuple(float(v) for v in scenario_cfg.bbox)
    co_label = str(data_provenance.get("co_pol_label", "VV"))
    cr_label = str(data_provenance.get("cross_pol_label", "VH"))

    tile_by_id = {int(t["tile_id"]): t for t in tiles}
    tile_bounds_key = tuple(
        (tid, tuple(round(float(v), 6) for v in tile_by_id[tid]["bounds"]))
        for tid in sorted(tile_by_id)
    )
    spatial = _cached_osm_spatial_index(scenario_id, tile_bounds_key, bbox)
    osm = spatial["osm"]
    tile_road_count = spatial["tile_road_count"]
    tile_bridge_count = spatial["tile_bridge_count"]
    tile_facility_count = spatial.get("tile_facility_count", (0,) * 36)
    total_roads_km = spatial["total_roads_km"]
    tagged_pop_count = spatial["tagged_pop_count"]

    road_hazard_km = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    flagged_roads: List[Dict[str, Any]] = []

    for length_km, hit_tiles, r_name, r_hw, r_bridge in spatial["road_records"]:
        if hit_tiles:
            worst_cls = max(int(tile_by_id[idx]["quantum_class"]) for idx in hit_tiles)
            if any(int(tile_by_id[idx]["quantum_class"]) == 1 for idx in hit_tiles):
                worst_cls = 1
            elif any(int(tile_by_id[idx]["quantum_class"]) == 2 for idx in hit_tiles):
                worst_cls = 2
            road_hazard_km[worst_cls] += length_km
            if worst_cls in (1, 2) and len(flagged_roads) < 12:
                flagged_roads.append(
                    {
                        "name": r_name,
                        "highway": r_hw,
                        "bridge": r_bridge,
                        "hazard_class": worst_cls,
                        "length_km": round(length_km, 2),
                        "tile_ids": list(hit_tiles[:4]),
                    }
                )

    bridges_at_risk: List[Dict[str, Any]] = []
    for tid, b_name, b_hw, b_lon, b_lat in spatial["bridge_records"]:
        cls = int(tile_by_id[tid]["quantum_class"])
        if cls in (1, 2) or float(tile_by_id[tid].get("water_pct", 0.0)) >= 10.0:
            bridges_at_risk.append(
                {
                    "name": b_name,
                    "highway": b_hw,
                    "tile_id": tid,
                    "hazard_class": cls,
                    "lon": b_lon,
                    "lat": b_lat,
                }
            )

    settlements_exposed: List[Dict[str, Any]] = []
    tile_exposed_pop = [0] * 36
    estimated_exposed_pop = 0
    for tid, s_name, s_place, pop, s_lon, s_lat in spatial["settlement_records"]:
        t = tile_by_id[tid]
        cls = int(t["quantum_class"])
        if cls in (1, 2, 3):
            exposure_factor = 0.55 if cls in (1, 2) else 0.25
            exp_pop = int(round(pop * exposure_factor))
            tile_exposed_pop[tid] += exp_pop
            estimated_exposed_pop += exp_pop
            settlements_exposed.append(
                {
                    "name": s_name,
                    "place": s_place,
                    "tile_id": tid,
                    "hazard_class": cls,
                    "estimated_pop": exp_pop,
                    "lon": s_lon,
                    "lat": s_lat,
                }
            )

    facilities_exposed: List[Dict[str, Any]] = []
    for tid, f_name, f_amenity, f_lon, f_lat in spatial.get("facility_records", ()):
        t = tile_by_id[tid]
        cls = int(t["quantum_class"])
        if cls in (1, 2, 3) or bool(t.get("abstention_flag", False)):
            facilities_exposed.append(
                {
                    "name": f_name,
                    "amenity": f_amenity,
                    "tile_id": tid,
                    "hazard_class": cls,
                    "lon": f_lon,
                    "lat": f_lat,
                }
            )

    # Geodesic area calculations across the 36 tiles
    total_scene_km2 = 0.0
    flooded_km2 = 0.0
    burn_km2 = 0.0
    crop_stress_km2 = 0.0
    new_flood_sub_km2 = 0.0
    perm_water_sub_km2 = 0.0
    double_bounce_sub_km2 = 0.0
    chokepoint_tiles: List[Dict[str, Any]] = []

    enriched_tiles: List[Dict[str, Any]] = []
    for t in tiles:
        tid = int(t["tile_id"])
        area_km2 = geodesic_bounds_area_km2(t["bounds"])
        total_scene_km2 += area_km2
        cls = int(t["quantum_class"])

        if cls == 1:
            flooded_km2 += area_km2
        elif cls == 2:
            burn_km2 += area_km2
        elif cls == 3:
            crop_stress_km2 += area_km2

        new_flood_sub_km2 += area_km2 * (float(t.get("new_flood_pct", 0.0)) / 100.0)
        perm_water_sub_km2 += area_km2 * (float(t.get("permanent_water_pct", 0.0)) / 100.0)
        double_bounce_sub_km2 += area_km2 * (float(t.get("double_bounce_pct", 0.0)) / 100.0)

        t_enriched = dict(t)
        t_enriched["area_km2"] = round(area_km2, 2)
        t_enriched["osm_road_count"] = tile_road_count[tid]
        t_enriched["osm_bridge_count"] = tile_bridge_count[tid]
        t_enriched["osm_facility_count"] = tile_facility_count[tid]
        t_enriched["tile_explanation"] = _build_tile_explanation(
            t_enriched,
            co_label,
            cr_label,
            tile_bridge_count[tid],
            tile_road_count[tid],
            tile_facility_count[tid],
        )
        enriched_tiles.append(t_enriched)

        if cls in (1, 2) and (tile_bridge_count[tid] > 0 or tile_road_count[tid] >= 4):
            chokepoint_tiles.append(
                {
                    "tile_id": tid,
                    "row": int(t["row"]),
                    "col": int(t["col"]),
                    "hazard_class": cls,
                    "hazard_label": t["quantum_label"],
                    "osm_roads": tile_road_count[tid],
                    "osm_bridges": tile_bridge_count[tid],
                    "confidence": float(t["quantum_confidence"]),
                }
            )

    chokepoint_tiles.sort(key=lambda item: (item["osm_bridges"], item["osm_roads"]), reverse=True)
    chokepoint_tiles = chokepoint_tiles[:6]

    affected_km2 = flooded_km2 + burn_km2 + crop_stress_km2
    agricultural_ha = crop_stress_km2 * 100.0

    # Transparent weighted priority queue across hazard, critical facilities, bridges, population & uncertainty
    priority_queue: List[Dict[str, Any]] = []
    severity_weights = {1: 42.0, 2: 38.0, 3: 18.0, 0: 0.0}
    for t in enriched_tiles:
        tid = int(t["tile_id"])
        cls = int(t["quantum_class"])
        abstain = bool(t.get("abstention_flag", False))
        fac_cnt = tile_facility_count[tid]
        br_cnt = tile_bridge_count[tid]
        rd_cnt = tile_road_count[tid]
        pop_est = tile_exposed_pop[tid]

        if cls == 0 and not abstain and fac_cnt == 0 and br_cnt == 0:
            continue

        sev_pts = severity_weights.get(cls, 0.0)
        fac_pts = min(30.0, fac_cnt * 12.0)
        br_pts = min(24.0, br_cnt * 6.0)
        rd_pts = min(8.0, rd_cnt * 1.5)
        pop_pts = min(18.0, round(pop_est / 1200.0, 1))
        unc_pts = 12.0 if abstain else 0.0
        score = round(sev_pts + fac_pts + br_pts + rd_pts + pop_pts + unc_pts, 1)

        if score >= 65.0 or (cls in (1, 2) and (fac_cnt > 0 or br_cnt >= 2)):
            tier = "CRITICAL"
        elif score >= 42.0 or cls in (1, 2):
            tier = "HIGH"
        elif score >= 18.0 or abstain or cls == 3:
            tier = "MEDIUM"
        else:
            tier = "MONITOR"

        reasons: List[str] = []
        if cls in (1, 2, 3):
            reasons.append(f"SAR-indicated {t['quantum_label']}")
        else:
            reasons.append("Nominal terrain")
        if fac_cnt > 0:
            reasons.append(f"{fac_cnt} critical facility(ies)")
        if br_cnt > 0:
            reasons.append(f"{br_cnt} bridge(s)")
        if pop_est > 0:
            reasons.append(f"~{pop_est:,} est. exposed")
        if abstain:
            reasons.append("High uncertainty (Unknown/Verify)")

        priority_queue.append(
            {
                "tile_id": tid,
                "row": int(t["row"]),
                "col": int(t["col"]),
                "priority_tier": tier,
                "priority_score": score,
                "hazard_class": cls,
                "hazard_label": t["quantum_label"],
                "osm_bridges": br_cnt,
                "osm_roads": rd_cnt,
                "osm_facilities": fac_cnt,
                "estimated_pop": pop_est,
                "abstention_flag": abstain,
                "reason": " · ".join(reasons),
            }
        )

    priority_queue.sort(key=lambda item: (item["priority_score"], item["osm_facilities"], item["osm_bridges"]), reverse=True)
    priority_queue = priority_queue[:12]

    # Scene-wide dominant radiometric scattering signature
    mean_dco = sum(float(t.get("delta_hh_db", 0.0)) for t in tiles) / max(1, len(tiles))
    mean_dcr = sum(float(t.get("delta_hv_db", 0.0)) for t in tiles) / max(1, len(tiles))
    mean_drvi = sum(float(t.get("delta_rvi", 0.0)) for t in tiles) / max(1, len(tiles))
    abstain_count = sum(1 for t in tiles if t.get("abstention_flag", False))
    mean_conf = sum(float(t.get("quantum_confidence", 0.0)) for t in tiles) / max(1, len(tiles))

    if flooded_km2 >= max(burn_km2, crop_stress_km2) and flooded_km2 > 0:
        dominant_shift = (
            f"SAR-Indicated Inundation & Double-Bounce (Δ{co_label} {mean_dco:+.2f} dB, "
            f"{new_flood_sub_km2:.1f} km² newly darkened water sub-pixels)"
        )
        headline = (
            f"SAR-indicated flood alert across {flooded_km2:.1f} km² of tiles with "
            f"{len(bridges_at_risk)} OSM bridge crossing(s) requiring inspection."
        )
        what_changed = (
            f"Between {scenario_cfg.pre_date} and {scenario_cfg.post_date}, {co_label}/{cr_label} SAR indicates "
            f"{flooded_km2:.1f} km² of flood-flagged tiles ({double_bounce_sub_km2:.1f} km² flooded-vegetation "
            f"double-bounce + {new_flood_sub_km2:.1f} km² newly darkened water; {perm_water_sub_km2:.1f} km² unchanged channel)."
        )
    elif burn_km2 >= max(flooded_km2, crop_stress_km2) and burn_km2 > 0:
        dominant_shift = (
            f"SAR-Indicated Canopy Volume Collapse (Δ{cr_label} {mean_dcr:+.2f} dB, ΔRVI {mean_drvi:+.3f})"
        )
        headline = (
            f"SAR-indicated wildfire burn disturbance across {burn_km2:.1f} km² with "
            f"{road_hazard_km[2]:.1f} km of roads flagged for verification."
        )
        what_changed = (
            f"Between {scenario_cfg.pre_date} and {scenario_cfg.post_date}, cross-polarized {cr_label} volume "
            f"scattering collapsed by {mean_dcr:+.2f} dB (mean ΔRVI {mean_drvi:+.3f}) across {burn_km2:.1f} km² "
            f"of non-water terrain, separating defoliated canopy signatures from permanent river channels."
        )
    elif crop_stress_km2 > 0:
        dominant_shift = (
            f"Dielectric & Canopy Moisture Shift (Δ{co_label} {mean_dco:+.2f} dB, Δ{cr_label} {mean_dcr:+.2f} dB)"
        )
        headline = (
            f"SAR-indicated agricultural & moisture shift across {agricultural_ha:,.0f} ha "
            f"({crop_stress_km2:.1f} km²) along key rural corridors."
        )
        what_changed = (
            f"Between {scenario_cfg.pre_date} and {scenario_cfg.post_date}, dual-pol {co_label}/{cr_label} "
            f"shifts indicated {agricultural_ha:,.0f} ha of crop/soil stress and {flooded_km2:.1f} km² of localized ponding."
        )
    else:
        dominant_shift = f"Nominal Seasonal Backscatter Baseline (Δ{co_label} {mean_dco:+.2f} dB)"
        headline = "Nominal terrain conditions across analyzed scene; no high-severity hazard zone dominant."
        what_changed = (
            f"Minimal structural radar divergence between {scenario_cfg.pre_date} and {scenario_cfg.post_date}."
        )

    next_action = (
        f"Dispatch verification to {abstain_count} high-uncertainty boundary tile(s) and "
        f"prioritize inspection at {', '.join('T' + str(c['tile_id']) for c in priority_queue[:3])}."
        if (abstain_count > 0 and priority_queue)
        else (
            f"Inspect {len(bridges_at_risk)} OSM bridge crossing(s) and review lowest-modeled-risk corridor before dispatch."
            if bridges_at_risk
            else "Review lowest-modeled-risk corridor candidate while monitoring mixed-pixel boundary tiles."
        )
    )

    humanitarian_impact = {
        "total_scene_area_km2": round(total_scene_km2, 2),
        "affected_area_km2": round(affected_km2, 2),
        "flooded_area_km2": round(flooded_km2, 2),
        "burn_scar_area_km2": round(burn_km2, 2),
        "crop_stress_area_km2": round(crop_stress_km2, 2),
        "agricultural_area_ha": round(agricultural_ha, 1),
        "new_flood_sub_km2": round(new_flood_sub_km2, 2),
        "permanent_water_sub_km2": round(perm_water_sub_km2, 2),
        "double_bounce_sub_km2": round(double_bounce_sub_km2, 2),
        "osm_available": bool(osm.get("available", False)),
        "total_osm_roads_km": round(total_roads_km, 2),
        "roads_flood_exposed_km": round(road_hazard_km[1], 2),
        "roads_fire_exposed_km": round(road_hazard_km[2], 2),
        "roads_at_risk_km": round(road_hazard_km[1] + road_hazard_km[2], 2),
        "bridges_total": int(osm.get("counts", {}).get("bridges", 0)),
        "bridges_at_risk_count": len(bridges_at_risk),
        "bridges_at_risk": bridges_at_risk[:8],
        "settlements_exposed_count": len(settlements_exposed),
        "settlements_exposed": settlements_exposed[:8],
        "facilities_total": int(osm.get("counts", {}).get("facilities", len(spatial.get("facility_records", ())))),
        "facilities_exposed_count": len(facilities_exposed),
        "facilities_exposed": facilities_exposed[:12],
        "estimated_exposed_population": int(estimated_exposed_pop),
        "population_method": (
            f"OSM population tags ({tagged_pop_count} tagged) + settlement class density prior (estimate only)"
            if settlements_exposed
            else "No hazard-intersecting OSM settlements detected"
        ),
        "critical_chokepoint_tiles": chokepoint_tiles,
        "priority_queue": priority_queue,
        "uncertain_tiles_count": int(abstain_count),
        "dominant_scattering_shift": dominant_shift,
    }

    explainable_brief = {
        "headline": headline,
        "what_changed": what_changed,
        "supporting_channels": (
            f"{data_provenance.get('sensor_band', 'Dual-Pol SAR')} ({co_label}+{cr_label}) | "
            f"Scene Mean Δ{co_label}: {mean_dco:+.2f} dB, Δ{cr_label}: {mean_dcr:+.2f} dB, ΔRVI: {mean_drvi:+.3f}"
        ),
        "confidence_summary": (
            f"Mean Hybrid Q-SAR confidence {mean_conf:.1%}; {abstain_count}/36 tiles flagged "
            f"Unknown/Verify for operator review ({36 - abstain_count}/36 decisive)."
        ),
        "next_verification_action": next_action,
        "flagged_roads_sample": flagged_roads[:6],
    }

    osm_summary = {
        "available": bool(osm.get("available", False)),
        "source": osm.get("source", "OpenStreetMap Overpass API"),
        "counts": osm.get("counts", {}),
        "roads": [
            {
                "osm_id": r["osm_id"],
                "name": r.get("name", ""),
                "highway": r.get("highway", ""),
                "bridge": bool(r.get("bridge", False)),
                "coords": r.get("coords", []),
            }
            for r in osm.get("roads", [])[:120]
        ],
        "bridges": osm.get("bridges", [])[:50],
        "settlements": osm.get("settlements", [])[:50],
        "facilities": osm.get("facilities", [])[:150],
    }

    return humanitarian_impact, enriched_tiles, {"explainable_brief": explainable_brief, "osm_summary": osm_summary}
