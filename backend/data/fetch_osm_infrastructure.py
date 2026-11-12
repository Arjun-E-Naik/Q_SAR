"""Fetch and cache real OpenStreetMap road networks, bridges, waterways, and settlements.

Caches lightweight JSON under `backend/data/osm_cache/{scenario_id}.json` so
geospatial exposure assessment and road-network QAOA routing execute 100% offline
with sub-millisecond load latency during `/api/analyze`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.data.generate_demo_patches import SCENARIOS

OSM_CACHE_DIR = Path(__file__).resolve().parent / "osm_cache"

OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HIGHWAY_FILTER = "motorway|trunk|primary|secondary|tertiary"
FACILITY_AMENITY_FILTER = "hospital|clinic|doctors|fire_station|police|school|college|shelter|community_centre"


def _post_overpass(query: str, timeout_s: int = 15) -> Dict[str, Any]:
    headers = {
        "User-Agent": "Q-SAR-Sentinel/2.0 (NASA-Space-Apps-2026)",
        "Accept": "application/json",
    }
    last_err: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS[:2]:
        try:
            resp = requests.post(endpoint, data={"data": query}, headers=headers, timeout=timeout_s)
            if resp.status_code == 200 and resp.text.lstrip().startswith("{"):
                return resp.json()
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Overpass endpoints failed: {last_err}")


def _query_overpass(bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
    west, south, east, north = bbox
    primary_query = (
        f"[out:json][timeout:15];("
        f'way["highway"~"{HIGHWAY_FILTER}"]({south},{west},{north},{east});'
        f'way["waterway"~"river|canal"]({south},{west},{north},{east});'
        f'node["place"~"city|town|village|hamlet"]({south},{west},{north},{east});'
        f'nwr["amenity"~"{FACILITY_AMENITY_FILTER}"]({south},{west},{north},{east});'
        f");out body center;>;out skel qt;"
    )
    try:
        return _post_overpass(primary_query, timeout_s=18)
    except Exception:
        # Lightweight arterial fallback for dense mega-cities where all-tertiary Overpass times out
        light_query = (
            f"[out:json][timeout:12];("
            f'way["highway"~"motorway|trunk|primary|secondary"]({south},{west},{north},{east});'
            f'node["place"~"city|town|village"]({south},{west},{north},{east});'
            f'nwr["amenity"~"hospital|clinic|fire_station|shelter"]({south},{west},{north},{east});'
            f");out body center;>;out skel qt;"
        )
        return _post_overpass(light_query, timeout_s=15)


def _query_facilities_only(bbox: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
    west, south, east, north = bbox
    query = (
        f"[out:json][timeout:35];("
        f'nwr["amenity"~"{FACILITY_AMENITY_FILTER}"]({south},{west},{north},{east});'
        f");out center qt;"
    )
    raw = _post_overpass(query)
    return _parse_facilities_from_elements(raw.get("elements", []), bbox)


def _parse_facilities_from_elements(
    elements: List[Dict[str, Any]], bbox: Tuple[float, float, float, float]
) -> List[Dict[str, Any]]:
    west, south, east, north = bbox
    facilities: List[Dict[str, Any]] = []
    seen = set()
    priority_rank = {
        "hospital": 0,
        "clinic": 1,
        "fire_station": 2,
        "shelter": 3,
        "police": 4,
        "community_centre": 5,
        "school": 6,
        "college": 7,
        "doctors": 8,
    }
    for el in elements:
        tags = el.get("tags") or {}
        amenity = str(tags.get("amenity", "")).lower().strip()
        if not amenity or amenity not in priority_rank:
            continue
        if el.get("type") == "node" and "lon" in el and "lat" in el:
            lon, lat = float(el["lon"]), float(el["lat"])
        elif isinstance(el.get("center"), dict):
            lon, lat = float(el["center"]["lon"]), float(el["center"]["lat"])
        else:
            continue
        if not (west <= lon <= east and south <= lat <= north):
            continue
        key = (amenity, round(lon, 4), round(lat, 4))
        if key in seen:
            continue
        seen.add(key)
        raw_name = str(tags.get("name") or tags.get("operator") or f"{amenity.replace('_', ' ').title()}")
        facilities.append(
            {
                "osm_id": int(el["id"]),
                "name": raw_name,
                "amenity": amenity,
                "lon": round(lon, 6),
                "lat": round(lat, 6),
            }
        )
    facilities.sort(key=lambda f: (priority_rank.get(f["amenity"], 9), f["name"]))
    return facilities


def _parse_osm_elements(
    scenario_id: str, bbox: Tuple[float, float, float, float], raw: Dict[str, Any]
) -> Dict[str, Any]:
    elements = raw.get("elements", [])
    node_coords: Dict[int, Tuple[float, float]] = {}
    settlements: List[Dict[str, Any]] = []

    for el in elements:
        if el.get("type") == "node":
            nid = int(el["id"])
            lon = float(el["lon"])
            lat = float(el["lat"])
            node_coords[nid] = (lon, lat)
            tags = el.get("tags") or {}
            if "place" in tags and "name" in tags:
                pop_raw = tags.get("population")
                pop_val = None
                if pop_raw:
                    try:
                        pop_val = int(str(pop_raw).replace(",", "").strip())
                    except ValueError:
                        pop_val = None
                settlements.append(
                    {
                        "osm_id": nid,
                        "name": str(tags["name"]),
                        "place": str(tags["place"]),
                        "lon": round(lon, 6),
                        "lat": round(lat, 6),
                        "population": pop_val,
                    }
                )

    roads: List[Dict[str, Any]] = []
    bridges: List[Dict[str, Any]] = []
    waterways: List[Dict[str, Any]] = []

    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        nids = [int(n) for n in el.get("nodes", []) if int(n) in node_coords]
        if len(nids) < 2:
            continue
        coords = [[round(node_coords[n][0], 6), round(node_coords[n][1], 6)] for n in nids]

        if "highway" in tags:
            is_bridge = str(tags.get("bridge", "no")).lower() not in ("", "no", "false", "0")
            name = str(tags.get("name") or tags.get("ref") or f"{tags['highway'].title()} Road")
            road_obj = {
                "osm_id": int(el["id"]),
                "name": name,
                "highway": str(tags["highway"]),
                "bridge": is_bridge,
                "node_ids": nids,
                "coords": coords,
            }
            roads.append(road_obj)
            if is_bridge:
                mid = coords[len(coords) // 2]
                bridges.append(
                    {
                        "osm_id": int(el["id"]),
                        "name": name,
                        "highway": str(tags["highway"]),
                        "lon": mid[0],
                        "lat": mid[1],
                        "coords": coords,
                    }
                )
        elif "waterway" in tags:
            waterways.append(
                {
                    "osm_id": int(el["id"]),
                    "name": str(tags.get("name") or tags["waterway"].title()),
                    "waterway": str(tags["waterway"]),
                    "coords": coords,
                }
            )

    # Deduplicate settlements by name and sort priority places first
    place_rank = {"city": 0, "town": 1, "village": 2, "hamlet": 3}
    settlements.sort(key=lambda s: (place_rank.get(s["place"], 9), s["name"]))
    facilities = _parse_facilities_from_elements(elements, bbox)

    return {
        "scenario_id": scenario_id,
        "bbox": list(bbox),
        "source": "OpenStreetMap Overpass API",
        "available": True,
        "counts": {
            "roads": len(roads),
            "bridges": len(bridges),
            "settlements": len(settlements),
            "waterways": len(waterways),
            "facilities": len(facilities),
        },
        "roads": roads,
        "bridges": bridges,
        "settlements": settlements,
        "waterways": waterways,
        "facilities": facilities,
    }


def enrich_existing_osm_caches_with_facilities() -> None:
    """Safely fetch critical facilities (hospitals, clinics, shelters, schools) into cached OSM JSONs."""
    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sid, cfg in SCENARIOS.items():
        out_path = OSM_CACHE_DIR / f"{sid}.json"
        if out_path.exists():
            cached = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            cached = _parse_osm_elements(sid, cfg.bbox, _query_overpass(cfg.bbox))

        facilities = _query_facilities_only(cfg.bbox)
        cached["facilities"] = facilities
        cached.setdefault("counts", {})["facilities"] = len(facilities)
        out_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
        print(
            f"[OSM-Facilities:{sid}] facilities={len(facilities)} "
            f"(roads={cached['counts'].get('roads', 0)}, bridges={cached['counts'].get('bridges', 0)})"
        )


def ensure_osm_cache_for_bbox(scenario_id: str, bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
    """Fetch and cache OSM infrastructure for an arbitrary bbox if not already cached."""
    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OSM_CACHE_DIR / f"{scenario_id}.json"
    if out_path.exists():
        try:
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("available", True) and cached.get("roads"):
                cached.setdefault("available", True)
                return cached
        except Exception:
            pass

    try:
        raw = _query_overpass(bbox)
        parsed = _parse_osm_elements(scenario_id, bbox, raw)
        out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        return parsed
    except Exception as exc:
        fallback = {
            "scenario_id": scenario_id,
            "bbox": list(bbox),
            "source": f"Overpass unavailable ({exc}); using 6x6 tactical terrain fallback",
            "available": False,
            "counts": {"roads": 0, "bridges": 0, "settlements": 0, "waterways": 0, "facilities": 0},
            "roads": [],
            "bridges": [],
            "settlements": [],
            "waterways": [],
            "facilities": [],
        }
        out_path.write_text(json.dumps(fallback, indent=2), encoding="utf-8")
        return fallback


def fetch_all_osm_caches() -> None:
    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sid, cfg in SCENARIOS.items():
        raw = _query_overpass(cfg.bbox)
        parsed = _parse_osm_elements(sid, cfg.bbox, raw)
        out_path = OSM_CACHE_DIR / f"{sid}.json"
        out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        print(
            f"[OSM:{sid}] roads={parsed['counts']['roads']} "
            f"bridges={parsed['counts']['bridges']} "
            f"settlements={parsed['counts']['settlements']} "
            f"waterways={parsed['counts']['waterways']} "
            f"facilities={parsed['counts']['facilities']}"
        )


if __name__ == "__main__":
    if "--enrich-facilities" in sys.argv:
        enrich_existing_osm_caches_with_facilities()
    else:
        fetch_all_osm_caches()
