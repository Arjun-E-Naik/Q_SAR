"""FastAPI Mission Control Backend for Q-SAR Sentinel.

Exposes:
- GET  /api/health
- GET  /api/scenarios
- POST /api/analyze
- GET  /api/quantum-circuit
- GET  /api/scenarios/{scenario_id}/geojson
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
from pydantic import BaseModel, Field
import torch

# Allow launching either from repo root (`uvicorn backend.main:app`) or `backend/`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.data.generate_demo_patches import CLASS_NAMES, SCENARIOS, register_custom_scenario
from backend.data.fetch_osm_infrastructure import ensure_osm_cache_for_bbox
from backend.data.fetch_real_sar_geotiffs import fetch_custom_sar_scene
from backend.impact_analyzer import (
    _cached_osm_spatial_index,
    compute_humanitarian_and_explainable_impact,
    load_osm_infrastructure,
)
from backend.qaoa_router import _cached_osm_topology, solve_qaoa_disaster_route
from backend.quantum_model import (
    DEFAULT_WEIGHTS_PATH,
    QDEV_NAME,
    HybridQSARNet,
    build_bitemporal_input,
    compute_uncertainty_and_abstention,
    get_circuit_specification,
    load_benchmark_metrics,
    load_or_init_hybrid_model,
)
from backend.sar_processor import clear_sar_scene_cache, list_scenarios, load_sar_scene


class AnalyzeRequest(BaseModel):
    scenario_id: str = Field(default="nagpur_flood", description="NASA SAR scenario ID")
    look_count: int = Field(default=4, ge=1, le=32, description="SAR multi-look Gamma speckle parameter L")
    start_node: Optional[Tuple[int, int]] = Field(default=None, description="Optional (row, col) relief origin")
    end_node: Optional[Tuple[int, int]] = Field(default=None, description="Optional (row, col) rescue destination")
    query_nasa_cmr: bool = Field(default=False, description="Optionally query live NASA CMR via earthaccess")
    seed: int = Field(default=42, description="Deterministic scene seed")
    data_source: Literal["real_s1_rtc", "nisar_sim"] = Field(
        default="real_s1_rtc",
        description="Use real cached Sentinel-1 RTC GeoTIFF crops ('real_s1_rtc') or simulated L-band ('nisar_sim')",
    )
    risk_aversion: float = Field(
        default=1.0,
        ge=0.2,
        le=2.0,
        description="Multi-objective QAOA risk aversion alpha (0.2=speed priority, 1.0=balanced, 1.8=max hazard avoidance)",
    )


_STATE: Dict[str, Any] = {}
_VERIFICATION_LOCK = threading.Lock()
_VERIFICATION_PATH = Path(__file__).resolve().parent / "data" / "verification_state.json"


class VerificationUpdateRequest(BaseModel):
    scenario_id: str = Field(description="Scenario ID")
    tile_id: Optional[int] = Field(default=None, ge=0, le=35, description="Optional tile ID to verify")
    tile_status: Optional[Literal["unreviewed", "confirmed_hazard", "false_alarm", "needs_inspection"]] = Field(
        default=None, description="Operator ground-truth triage status for this tile"
    )
    verified_label: Optional[int] = Field(
        default=None, ge=0, le=3, description="Optional manually confirmed class index (0..3)"
    )
    tile_note: Optional[str] = Field(default=None, max_length=240)
    route_approval: Optional[Literal["pending_human_approval", "approved_for_recon", "rejected"]] = Field(
        default=None, description="Human-in-the-loop sign-off state for the active candidate corridor"
    )
    route_note: Optional[str] = Field(default=None, max_length=240)


def _read_verification_store() -> Dict[str, Any]:
    with _VERIFICATION_LOCK:
        if not _VERIFICATION_PATH.exists():
            return {"scenarios": {}}
        try:
            data = json.loads(_VERIFICATION_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("scenarios"), dict):
                return data
        except Exception:
            pass
        return {"scenarios": {}}


def _write_verification_store(store: Dict[str, Any]) -> None:
    with _VERIFICATION_LOCK:
        _VERIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        _VERIFICATION_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t0 = time.perf_counter()
    model = load_or_init_hybrid_model(DEFAULT_WEIGHTS_PATH)
    benchmarks = load_benchmark_metrics(model)
    for sc_id, sc_cfg in SCENARIOS.items():
        _cached_osm_topology(sc_id, sc_cfg.start_node, sc_cfg.end_node)
    _STATE["model"] = model
    _STATE["benchmarks"] = benchmarks
    _STATE["boot_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    yield
    _STATE.clear()


app = FastAPI(
    title="Q-SAR Sentinel API",
    description="Hybrid Quantum-Classical NISAR SAR Anomaly Detection & QAOA Disaster Routing",
    version="1.0.0",
    lifespan=lifespan,
)

_raw_origins = os.environ.get("QSAR_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _get_model() -> HybridQSARNet:
    model = _STATE.get("model")
    if model is None:
        model = load_or_init_hybrid_model(DEFAULT_WEIGHTS_PATH)
        _STATE["model"] = model
    return model


def _validate_node(node: Optional[Tuple[int, int]], default: Tuple[int, int]) -> Tuple[int, int]:
    if node is None:
        return default
    r, c = int(node[0]), int(node[1])
    if not (0 <= r < 6 and 0 <= c < 6):
        raise HTTPException(status_code=422, detail=f"Grid node {node} must have coordinates in [0, 5].")
    return (r, c)


def _run_full_analysis(req: AnalyzeRequest) -> Dict[str, Any]:
    if req.scenario_id not in SCENARIOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{req.scenario_id}'. Valid scenarios: {list(SCENARIOS)}",
        )

    t_total_start = time.perf_counter()
    model = _get_model()
    scene = load_sar_scene(
        scenario_id=req.scenario_id,
        look_count=req.look_count,
        query_nasa_cmr=req.query_nasa_cmr,
        seed=req.seed,
        data_source=req.data_source,
    )
    cfg = scene["scenario"]
    provenance = scene.get("data_provenance", {})
    gt_available = bool(provenance.get("ground_truth_available", True))

    bitemporal_tensor = build_bitemporal_input(
        scene["pre_patches"], scene["post_patches"]
    )
    t_inf_start = time.perf_counter()
    with torch.no_grad():
        (
            q_logits,
            c_logits,
            angles,
            q_expvals,
            sv_probs_np,
            bloch_np,
            entropy_np,
        ) = model.infer_with_telemetry(bitemporal_tensor)
        q_probs = torch.softmax(q_logits, dim=-1).numpy()
        c_probs = torch.softmax(c_logits, dim=-1).numpy()
        angles_np = angles.numpy()
        expvals_np = q_expvals.numpy()
    inference_ms = (time.perf_counter() - t_inf_start) * 1000.0
    telemetry_ms = 0.0  # Computed jointly inside batched statevector telemetry pass

    tile_predictions: List[Dict[str, Any]] = []
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    q_preds_list: List[int] = []
    c_preds_list: List[int] = []
    ref_list: List[int] = []
    q_correct = 0
    c_correct = 0
    agree_count = 0
    abstain_count = 0

    for idx, t_meta in enumerate(scene["tile_metrics"]):
        q_p = q_probs[idx]
        c_p = c_probs[idx]
        q_cls = int(np.argmax(q_p))
        c_cls = int(np.argmax(c_p))
        ref_cls = int(t_meta["reference_class"])
        q_conf = float(np.max(q_p))
        c_conf = float(np.max(c_p))
        unc = compute_uncertainty_and_abstention(q_p, float(entropy_np[idx]))

        q_preds_list.append(q_cls)
        c_preds_list.append(c_cls)
        ref_list.append(ref_cls)

        class_counts[q_cls] += 1
        if q_cls == c_cls:
            agree_count += 1
        if unc["abstention_flag"]:
            abstain_count += 1
        if ref_cls >= 0:
            if q_cls == ref_cls:
                q_correct += 1
            if c_cls == ref_cls:
                c_correct += 1

        new_flood_pct = float(t_meta.get("new_flood_pct", 0.0))
        perm_water_pct = float(t_meta.get("permanent_water_pct", 0.0))
        burn_pct = float(t_meta.get("burn_pct", 0.0))
        dbounce_pct = float(t_meta.get("double_bounce_pct", 0.0))
        if new_flood_pct >= 6.0 or float(t_meta.get("water_pct", 0.0)) >= 18.0:
            phys_flag = "Active Open Flood"
        elif dbounce_pct >= 28.0:
            phys_flag = "Flooded Veg Double-Bounce"
        elif burn_pct >= 20.0:
            phys_flag = "Canopy Burn Loss"
        elif perm_water_pct >= 12.0:
            phys_flag = "Permanent River Channel"
        else:
            phys_flag = "Nominal"

        tile_predictions.append(
            {
                **t_meta,
                "quantum_class": q_cls,
                "quantum_label": CLASS_NAMES[q_cls],
                "quantum_class_name": CLASS_NAMES[q_cls],
                "quantum_confidence": round(q_conf, 4),
                "quantum_probs": [round(float(v), 4) for v in q_p],
                "classical_class": c_cls,
                "classical_label": CLASS_NAMES[c_cls],
                "classical_class_name": CLASS_NAMES[c_cls],
                "classical_confidence": round(c_conf, 4),
                "classical_probs": [round(float(v), 4) for v in c_p],
                "models_disagree": bool(q_cls != c_cls),
                "confidence_delta": round(q_conf - c_conf, 4),
                "quantum_entropy": round(float(entropy_np[idx]), 4),
                "abstention_flag": unc["abstention_flag"],
                "uncertainty_tier": unc["uncertainty_tier"],
                "physical_anomaly": phys_flag,
                "quantum_angles_rad": [round(float(v), 4) for v in angles_np[idx]],
                "pauli_z_expvals": [round(float(v), 4) for v in expvals_np[idx]],
                "statevector_probs": [round(float(v), 4) for v in sv_probs_np[idx]],
                "bloch_vectors": [
                    [round(float(coord), 4) for coord in q_xyz]
                    for q_xyz in bloch_np[idx]
                ],
            }
        )

    # Compute geodesic areas, OSM road/bridge intersections, and plain-language briefs
    humanitarian_impact, tile_predictions, extra_impact = compute_humanitarian_and_explainable_impact(
        scenario_cfg=cfg,
        tiles=tile_predictions,
        data_provenance=provenance,
    )

    start_node = _validate_node(req.start_node, cfg.start_node)
    end_node = _validate_node(req.end_node, cfg.end_node)
    if start_node == end_node:
        raise HTTPException(
            status_code=422,
            detail="Origin (start_node) and destination (end_node) must be different tiles.",
        )

    t_qaoa_start = time.perf_counter()
    qaoa_result = solve_qaoa_disaster_route(
        tile_predictions=tile_predictions,
        start_node=start_node,
        end_node=end_node,
        tile_km=cfg.tile_km,
        risk_aversion=req.risk_aversion,
        scenario_id=cfg.scenario_id,
    )
    qaoa_ms = (time.perf_counter() - t_qaoa_start) * 1000.0
    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    # Geodesic area breakdown (falls back cleanly from OSM-aware tile areas)
    total_area_km2 = float(humanitarian_impact.get("total_scene_area_km2", 36 * cfg.tile_km * cfg.tile_km))
    affected_area_km2 = {
        CLASS_NAMES[0]: round(
            max(0.0, total_area_km2 - float(humanitarian_impact["affected_area_km2"])),
            2,
        ),
        CLASS_NAMES[1]: float(humanitarian_impact["flooded_area_km2"]),
        CLASS_NAMES[2]: float(humanitarian_impact["burn_scar_area_km2"]),
        CLASS_NAMES[3]: float(humanitarian_impact["crop_stress_area_km2"]),
    }

    def _scene_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
        yt = np.asarray(y_true, dtype=np.int64)
        yp = np.asarray(y_pred, dtype=np.int64)
        present = [c for c in range(4) if np.any(yt == c) or np.any(yp == c)]
        if not present:
            return 1.0
        f1s: List[float] = []
        for c in present:
            tp = int(np.sum((yt == c) & (yp == c)))
            fp = int(np.sum((yt != c) & (yp == c)))
            fn = int(np.sum((yt == c) & (yp != c)))
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
        return round(float(np.mean(f1s)), 4)

    benchmarks = _STATE.get("benchmarks") or load_benchmark_metrics(model)
    mean_entropy = round(float(np.mean(entropy_np)), 4)

    return {
        "scenario": {
            "scenario_id": cfg.scenario_id,
            "title": cfg.title,
            "subtitle": cfg.subtitle,
            "region": cfg.region,
            "sensor_mode": cfg.sensor_mode,
            "nasa_collections": list(cfg.nasa_collections),
            "bbox": list(cfg.bbox),
            "pre_date": cfg.pre_date,
            "post_date": cfg.post_date,
            "tile_km": cfg.tile_km,
            "look_count": req.look_count,
            "data_source": provenance.get("data_source", req.data_source),
            "risk_aversion": round(float(req.risk_aversion), 2),
        },
        "data_provenance": provenance,
        "data_quality": scene.get(
            "data_quality",
            {
                "status": "PASS",
                "crs": "EPSG:4326",
                "crs_aligned": True,
                "shape_aligned": True,
                "transform_aligned": True,
                "temporal_pairs_complete": True,
                "georeferenced_reprojection": False,
                "valid_pixel_pct": 100.0,
                "radiometric_in_bounds": True,
                "db_range": [-32.0, 2.0],
                "unsuitable_reason": None,
            },
        ),
        "nasa_metadata": scene["nasa_metadata"],
        "timing_ms": {
            "quantum_inference_ms": round(inference_ms, 2),
            "statevector_telemetry_ms": round(telemetry_ms, 2),
            "qaoa_routing_ms": round(qaoa_ms, 2),
            "total_pipeline_ms": round(total_ms, 2),
        },
        "model_status": {
            "quantum_device": QDEV_NAME,
            "weight_source": model.weight_source,
            "ground_truth_available": gt_available,
            "ground_truth_category": provenance.get("ground_truth_category", "radiometric_heuristic_reference"),
            "ground_truth_type": provenance.get("ground_truth_type", "Reference"),
            "is_verified_manual_ground_truth": bool(provenance.get("is_verified_manual_ground_truth", False)),
            "metric_label": (
                "Accuracy (Synthetic GT)"
                if provenance.get("is_verified_manual_ground_truth", False)
                else "Agreement with Heuristic Reference"
            ),
            "is_heuristic_agreement": not bool(provenance.get("is_verified_manual_ground_truth", False)),
            "scene_quantum_heuristic_agreement": round(q_correct / 36.0, 4) if gt_available else None,
            "scene_classical_heuristic_agreement": round(c_correct / 36.0, 4) if gt_available else None,
            "scene_quantum_accuracy": round(q_correct / 36.0, 4) if gt_available else None,
            "scene_classical_accuracy": round(c_correct / 36.0, 4) if gt_available else None,
            "scene_quantum_macro_f1": _scene_macro_f1(ref_list, q_preds_list) if gt_available else None,
            "scene_classical_macro_f1": _scene_macro_f1(ref_list, c_preds_list) if gt_available else None,
            "accuracy_delta": round((q_correct - c_correct) / 36.0, 4) if gt_available else None,
            "quantum_classical_agreement": round(agree_count / 36.0, 4),
            "disagreement_count": int(36 - agree_count),
            "abstention_count": int(abstain_count),
            "mean_quantum_entropy": mean_entropy,
        },
        "class_names": CLASS_NAMES,
        "affected_area_km2": affected_area_km2,
        "humanitarian_impact": humanitarian_impact,
        "explainable_brief": extra_impact["explainable_brief"],
        "osm_infrastructure": extra_impact["osm_summary"],
        "tiles": tile_predictions,
        "qaoa_routing": qaoa_result,
        "images": scene["images"],
        "benchmarks": benchmarks,
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    model = _get_model()
    weight_bytes = DEFAULT_WEIGHTS_PATH.stat().st_size if DEFAULT_WEIGHTS_PATH.exists() else 0
    return {
        "status": "ok",
        "quantum_device": QDEV_NAME,
        "weight_source": model.weight_source,
        "weights_size_kb": round(weight_bytes / 1024.0, 2),
        "boot_ms": _STATE.get("boot_ms", 0.0),
        "torch_version": torch.__version__,
    }


@app.get("/api/scenarios")
def get_scenarios() -> Dict[str, Any]:
    return {"scenarios": list_scenarios(), "class_names": CLASS_NAMES}


@app.post("/api/analyze")
def analyze_sar_scene(req: AnalyzeRequest) -> Dict[str, Any]:
    return _run_full_analysis(req)


@app.get("/api/quantum-circuit")
def get_quantum_circuit() -> Dict[str, Any]:
    model = _get_model()
    return get_circuit_specification(model)


@app.get("/api/scenarios/{scenario_id}/geojson")
def export_scenario_geojson(
    scenario_id: str,
    look_count: int = Query(default=4, ge=1, le=32),
    data_source: Literal["real_s1_rtc", "nisar_sim"] = Query(default="real_s1_rtc"),
    risk_aversion: float = Query(default=1.0, ge=0.2, le=2.0),
) -> Response:
    result = _run_full_analysis(
        AnalyzeRequest(
            scenario_id=scenario_id,
            look_count=look_count,
            data_source=data_source,
            risk_aversion=risk_aversion,
        )
    )
    geojson_str = json.dumps(result["qaoa_routing"]["geojson"], indent=2)
    return Response(
        content=geojson_str,
        media_type="application/geo+json",
        headers={
            "Content-Disposition": f'attachment; filename="{scenario_id}_qsar_sentinel.geojson"'
        },
    )


@app.get("/api/verifications")
def get_verifications(scenario_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    store = _read_verification_store()
    scenarios = store.get("scenarios", {})
    if scenario_id is not None:
        if scenario_id not in SCENARIOS:
            raise HTTPException(status_code=404, detail=f"Unknown scenario '{scenario_id}'.")
        entry = scenarios.get(
            scenario_id,
            {"tiles": {}, "route_approval": {"status": "pending_human_approval", "note": "", "updated_at": None}},
        )
        return {"scenario_id": scenario_id, **entry}
    return {"scenarios": scenarios}


@app.post("/api/verifications")
def update_verification(req: VerificationUpdateRequest) -> Dict[str, Any]:
    if req.scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown scenario '{req.scenario_id}'.")

    store = _read_verification_store()
    scenarios = store.setdefault("scenarios", {})
    sc_entry = scenarios.setdefault(
        req.scenario_id,
        {
            "tiles": {},
            "route_approval": {"status": "pending_human_approval", "note": "", "updated_at": None},
        },
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    if req.tile_id is not None:
        tiles_map = sc_entry.setdefault("tiles", {})
        tkey = str(req.tile_id)
        if req.tile_status == "unreviewed":
            tiles_map.pop(tkey, None)
        else:
            existing = tiles_map.get(tkey, {})
            tiles_map[tkey] = {
                "status": req.tile_status or existing.get("status", "needs_inspection"),
                "verified_label": req.verified_label if req.verified_label is not None else existing.get("verified_label"),
                "note": req.tile_note if req.tile_note is not None else existing.get("note", ""),
                "updated_at": now_iso,
            }

    if req.route_approval is not None or req.route_note is not None:
        cur_route = sc_entry.setdefault(
            "route_approval",
            {"status": "pending_human_approval", "note": "", "updated_at": None},
        )
        sc_entry["route_approval"] = {
            "status": req.route_approval or cur_route.get("status", "pending_human_approval"),
            "note": req.route_note if req.route_note is not None else cur_route.get("note", ""),
            "updated_at": now_iso,
        }

    _write_verification_store(store)
    return {"scenario_id": req.scenario_id, **sc_entry}


_COORD_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*([NSns])?\s*[,\s]\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*([EWew])?\s*$"
)


def _parse_lat_lon_query(q: str) -> Optional[Tuple[float, float]]:
    """Parse direct 'lat, lon' or 'lat lon' coordinate strings."""
    m = _COORD_RE.match(q or "")
    if not m:
        return None
    lat = float(m.group(1))
    lon = float(m.group(3))
    if m.group(2) and m.group(2).upper() == "S":
        lat = -abs(lat)
    elif m.group(2) and m.group(2).upper() == "N":
        lat = abs(lat)
    if m.group(4) and m.group(4).upper() == "W":
        lon = -abs(lon)
    elif m.group(4) and m.group(4).upper() == "E":
        lon = abs(lon)
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return (round(lat, 5), round(lon, 5))
    return None


def _nominatim_geocode(q: str, limit: int = 6) -> List[Dict[str, Any]]:
    """Resolve a place name or coordinate string into candidate geographic points."""
    cleaned = (q or "").strip()
    if not cleaned:
        return []
    coords = _parse_lat_lon_query(cleaned)
    if coords is not None:
        lat, lon = coords
        return [
            {
                "label": f"Coordinates ({lat:.4f}, {lon:.4f})",
                "short_name": f"{lat:.2f}, {lon:.2f}",
                "lat": lat,
                "lon": lon,
                "type": "coordinates",
                "source": "direct_coords",
            }
        ]

    params = urlencode({"format": "jsonv2", "limit": str(limit), "q": cleaned})
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = Request(
        url,
        headers={
            "User-Agent": "Q-SAR-Sentinel/2.0 (NASA Space Apps Disaster Intelligence)",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Geocoding lookup failed ({exc}). You can also enter 'lat, lon' directly.",
        ) from exc

    results: List[Dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            lat = round(float(item["lat"]), 5)
            lon = round(float(item["lon"]), 5)
        except (KeyError, TypeError, ValueError):
            continue
        display_name = str(item.get("display_name") or item.get("name") or f"{lat}, {lon}")
        short_name = str(item.get("name") or display_name.split(",")[0]).strip()
        results.append(
            {
                "label": display_name,
                "short_name": short_name[:32],
                "lat": lat,
                "lon": lon,
                "type": str(item.get("type") or item.get("class") or "place"),
                "source": "nominatim",
            }
        )
    return results


def _validate_iso_date(val: Optional[str], field_name: str) -> Optional[str]:
    if val is None or not str(val).strip():
        return None
    s = str(val).strip()
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be YYYY-MM-DD (got '{s}').",
        ) from exc
    return s


def _build_location_bbox(lat: float, lon: float, span_deg: float) -> Tuple[float, float, float, float]:
    clamped_lat = max(-80.0, min(80.0, float(lat)))
    half_lat = span_deg / 2.0
    cos_lat = max(0.35, float(np.cos(np.deg2rad(clamped_lat))))
    half_lon = min(span_deg * 1.25, half_lat / cos_lat)
    west = round(max(-179.95, float(lon) - half_lon), 4)
    east = round(min(179.95, float(lon) + half_lon), 4)
    south = round(max(-83.95, clamped_lat - half_lat), 4)
    north = round(min(83.95, clamped_lat + half_lat), 4)
    return (west, south, east, north)


def _slugify_location_id(
    label: str, lat: float, lon: float, pre_date: Optional[str], post_date: Optional[str]
) -> str:
    head = label.split(",")[0].strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", head).strip("_")[:18] or "aoi"
    digest_src = f"{lat:.4f},{lon:.4f}|{pre_date or ''}|{post_date or ''}"
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:6]
    return f"loc_{slug}_{digest}"


class PrepareLocationRequest(BaseModel):
    query: Optional[str] = Field(default=None, max_length=200)
    lat: Optional[float] = Field(default=None, ge=-82.0, le=82.0)
    lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    label: Optional[str] = Field(default=None, max_length=160)
    pre_date: Optional[str] = Field(default=None, description="Optional YYYY-MM-DD pre-event target date")
    post_date: Optional[str] = Field(default=None, description="Optional YYYY-MM-DD post-event target date")
    span_deg: float = Field(default=0.24, ge=0.12, le=0.45, description="AOI degree span (~13-50 km)")


@app.get("/api/geocode")
def geocode_location(q: str = Query(..., min_length=2, max_length=200)) -> Dict[str, Any]:
    return {"query": q, "results": _nominatim_geocode(q)}


@app.post("/api/locations/prepare")
def prepare_custom_location(req: PrepareLocationRequest) -> Dict[str, Any]:
    pre_target = _validate_iso_date(req.pre_date, "pre_date")
    post_target = _validate_iso_date(req.post_date, "post_date")
    if pre_target and post_target and pre_target >= post_target:
        raise HTTPException(
            status_code=422,
            detail=f"pre_date ({pre_target}) must be earlier than post_date ({post_target}).",
        )

    if req.lat is not None and req.lon is not None:
        lat = float(req.lat)
        lon = float(req.lon)
        resolved_label = (req.label or req.query or f"AOI ({lat:.4f}, {lon:.4f})").strip()
    elif req.query and req.query.strip():
        candidates = _nominatim_geocode(req.query.strip(), limit=1)
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail=f"No geographic match found for '{req.query.strip()}'. Try 'City, Country' or 'lat, lon'.",
            )
        lat = float(candidates[0]["lat"])
        lon = float(candidates[0]["lon"])
        resolved_label = (req.label or candidates[0]["label"]).strip()
    else:
        raise HTTPException(status_code=422, detail="Provide either a search 'query' or 'lat'/'lon' coordinates.")

    bbox = _build_location_bbox(lat, lon, req.span_deg)
    scenario_id = _slugify_location_id(resolved_label, lat, lon, pre_target, post_target)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sar_fut = pool.submit(
                fetch_custom_sar_scene,
                scenario_id,
                bbox,
                pre_date=pre_target,
                post_date=post_target,
                location_name=resolved_label,
            )
            osm_fut = pool.submit(ensure_osm_cache_for_bbox, scenario_id, bbox)
            stac_meta = sar_fut.result()
            osm_payload = osm_fut.result()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Live Sentinel-1 / Overpass provisioning error: {exc}",
        ) from exc

    actual_pre = str(
        stac_meta.get("pre_date") or stac_meta.get("pre", {}).get("datetime", pre_target or "")
    )[:10]
    actual_post = str(
        stac_meta.get("post_date") or stac_meta.get("post", {}).get("datetime", post_target or "")
    )[:10]
    orbit_val = stac_meta.get("relative_orbit") if stac_meta.get("relative_orbit") is not None else stac_meta.get("orbit")
    pre_items = stac_meta.get("pre_item_ids") or stac_meta.get("pre", {}).get("item_ids", [])
    post_items = stac_meta.get("post_item_ids") or stac_meta.get("post", {}).get("item_ids", [])
    short_title = resolved_label.split(",")[0].strip()[:36] or scenario_id

    cfg = register_custom_scenario(
        scenario_id=scenario_id,
        title=short_title,
        region=resolved_label[:120],
        bbox=bbox,
        pre_date=actual_pre,
        post_date=actual_post,
    )

    # Flush cached scene / OSM / topology lookups so newly fetched GeoTIFFs & OSM attach cleanly
    clear_sar_scene_cache()
    load_osm_infrastructure.cache_clear()
    _cached_osm_spatial_index.cache_clear()
    _cached_osm_topology.cache_clear()
    _cached_osm_topology(cfg.scenario_id, cfg.start_node, cfg.end_node)

    return {
        "status": "ready",
        "scenario": {
            "scenario_id": cfg.scenario_id,
            "title": cfg.title,
            "subtitle": cfg.subtitle,
            "region": cfg.region,
            "sensor_mode": cfg.sensor_mode,
            "nasa_collections": cfg.nasa_collections,
            "bbox": list(cfg.bbox),
            "pre_date": cfg.pre_date,
            "post_date": cfg.post_date,
            "start_node": list(cfg.start_node),
            "end_node": list(cfg.end_node),
            "start_label": cfg.start_label,
            "end_label": cfg.end_label,
            "tile_km": cfg.tile_km,
            "has_real_s1_cache": True,
            "is_custom": True,
        },
        "osm_counts": {
            "roads": len(osm_payload.get("roads", [])),
            "bridges": len(osm_payload.get("bridges", [])),
            "settlements": len(osm_payload.get("settlements", [])),
            "facilities": len(osm_payload.get("facilities", [])),
            "available": bool(osm_payload.get("available", False)),
        },
        "stac_meta": {
            "orbit": orbit_val,
            "pre_datetime": actual_pre,
            "post_datetime": actual_post,
            "pre_items": pre_items,
            "post_items": post_items,
        },
    }
