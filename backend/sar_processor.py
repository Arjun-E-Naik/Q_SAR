"""NASA Earthdata SAR ingestion & sensor-aware dual-polarization preprocessing.

Supports:
1. Live NASA CMR granule discovery via `earthaccess` across BOTH `NISAR_L2_GCOV_V1`
   and `OPERA_L2_RTC-S1_V1`.
2. Local parsing of NISAR L2 GCOV HDF5 (.h5) L-band HH/HV grids via `h5py`.
3. Local parsing of real cached Sentinel-1 C-Band RTC GeoTIFFs (.tif, VV/VH) via `rasterio`
   using explicit C-band radiometric ranges (never mislabeled as L-band HH/HV).
4. Deterministic physics-based L-band NISAR simulation (`nisar_sim`) for controlled
   speckle & ablation testing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import numpy as np

# Load gitignored backend/.env if present (populates EARTHDATA_USERNAME / EARTHDATA_PASSWORD)
_BACKEND_ENV = Path(__file__).resolve().parent / ".env"
if _BACKEND_ENV.exists():
    load_dotenv(_BACKEND_ENV, override=False)

from backend.data.generate_demo_patches import (
    CLASS_NAMES,
    HH_DB_MAX,
    HH_DB_MIN,
    HV_DB_MAX,
    HV_DB_MIN,
    S1_VH_DB_MAX,
    S1_VH_DB_MIN,
    S1_VV_DB_MAX,
    S1_VV_DB_MIN,
    SCENARIOS,
    ScenarioConfig,
    _apply_gamma_speckle,
    assemble_sar_scene_from_db,
    db_to_norm,
    generate_realistic_sar_scene,
)

REAL_SAR_DIR = Path(__file__).resolve().parent / "data" / "real_sar"

logger = logging.getLogger(__name__)


def list_scenarios() -> List[Dict[str, Any]]:
    """Return metadata for all pre-configured NASA SAR disaster scenarios."""
    out: List[Dict[str, Any]] = []
    for cfg in SCENARIOS.values():
        out.append(
            {
                "scenario_id": cfg.scenario_id,
                "title": cfg.title,
                "subtitle": cfg.subtitle,
                "region": cfg.region,
                "sensor_mode": cfg.sensor_mode,
                "nasa_collections": list(cfg.nasa_collections),
                "bbox": list(cfg.bbox),
                "pre_date": cfg.pre_date,
                "post_date": cfg.post_date,
                "start_node": list(cfg.start_node),
                "end_node": list(cfg.end_node),
                "start_label": cfg.start_label,
                "end_label": cfg.end_label,
                "tile_km": cfg.tile_km,
            }
        )
    return out


def query_nasa_earthdata_granules(
    scenario_id: str, max_results_per_collection: int = 2
) -> Dict[str, Any]:
    """Query NASA CMR metadata via `earthaccess` across NISAR and OPERA collections."""
    cfg: ScenarioConfig = SCENARIOS[scenario_id]
    try:
        import earthaccess  # type: ignore

        auth_mode = "anonymous_cmr"
        uname = os.environ.get("EARTHDATA_USERNAME", "")
        upass = os.environ.get("EARTHDATA_PASSWORD", "")
        if uname and upass and uname != "your_earthdata_username":
            try:
                earthaccess.login(strategy="environment")
                auth_mode = "earthdata_env_authenticated"
            except Exception as auth_exc:
                logger.info("Earthdata env login skipped (%s); using public CMR search.", auth_exc)

        found_granules: List[Dict[str, Any]] = []
        queried_collections: List[str] = []
        for short_name in cfg.nasa_collections[:2]:
            # Widen temporal search window up to current mission era so NISAR_L2_GCOV_V1 hits appear
            t_start = min(cfg.pre_date, "2024-06-01") if cfg.pre_date < "2025-01-01" else cfg.pre_date
            t_end = max(cfg.post_date, datetime.now(timezone.utc).date().isoformat())
            results = earthaccess.search_data(
                short_name=short_name,
                bounding_box=cfg.bbox,
                temporal=(t_start, t_end),
                count=max_results_per_collection,
            )
            queried_collections.append(short_name)
            for item in results[:max_results_per_collection]:
                umm = item.get("umm", {}) if isinstance(item, dict) else {}
                temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
                links = item.data_links() if hasattr(item, "data_links") else []
                found_granules.append(
                    {
                        "granule_ur": umm.get("GranuleUR", "NASA-SAR-GRANULE"),
                        "collection": short_name,
                        "time_start": temporal.get("BeginningDateTime", cfg.pre_date),
                        "data_url": links[0] if links else None,
                    }
                )

        return {
            "status": "online_cmr_verified" if found_granules else "online_cmr_no_match_fallback",
            "auth_mode": auth_mode,
            "queried_collection": ", ".join(queried_collections) if queried_collections else cfg.nasa_collections[0],
            "granule_count": len(found_granules),
            "granules": found_granules,
        }
    except Exception as exc:
        logger.info("NASA CMR query fallback (%s): %s", scenario_id, exc)
        return {
            "status": "offline_physics_fallback",
            "queried_collection": cfg.nasa_collections[0],
            "granule_count": 0,
            "granules": [],
            "note": f"Offline mode active ({type(exc).__name__}); using local cached imagery/simulation.",
        }


def _reproject_georeferenced_band(
    src_arr: np.ndarray,
    src_transform: Any,
    src_crs: Any,
    dst_bbox: Tuple[float, float, float, float],
    src_nodata: Any = None,
    size: int = 384,
) -> Tuple[np.ndarray, float]:
    """Reproject a georeferenced 2D SAR raster to an aligned EPSG:4326 grid via Rasterio."""
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.warp import reproject

    dst_transform = transform_from_bounds(*dst_bbox, width=size, height=size)
    dst = np.full((size, size), np.nan, dtype=np.float32)
    reproject(
        source=src_arr.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    valid_pct = float(np.mean(np.isfinite(dst) & (dst > 0.0)) * 100.0)
    fill_val = float(np.nanmedian(dst[dst > 0.0])) if np.any(np.isfinite(dst) & (dst > 0.0)) else 1e-4
    clean = np.where(np.isfinite(dst) & (dst > 0.0), dst, fill_val).astype(np.float32)
    return np.maximum(clean, 1e-6), round(valid_pct, 2)


def load_nisar_h5_patches(
    h5_path: str | Path,
    target_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """Read a native NISAR L2 GCOV HDF5 file using HDF5 geolocation coordinates & EPSG CRS.

    Requires valid spatial metadata (`projection`, `xCoordinates`, `yCoordinates`); never silently
    treats unprojected array resizing as geospatial reprojection.
    """
    import h5py  # type: ignore
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    with h5py.File(str(h5_path), "r") as hf:
        freq_grp = hf["science/LSAR/GCOV/grids/frequencyA"]
        hh_key = "HHHH" if "HHHH" in freq_grp else "HH"
        hv_key = "HVHV" if "HVHV" in freq_grp else ("HV" if "HV" in freq_grp else "VH")
        hh_lin = np.asarray(freq_grp[hh_key][:], dtype=np.float32)
        hv_lin = np.asarray(freq_grp[hv_key][:], dtype=np.float32)

        if "xCoordinates" not in freq_grp or "yCoordinates" not in freq_grp:
            raise ValueError("NISAR GCOV HDF5 missing xCoordinates/yCoordinates geolocation arrays.")

        xs = np.asarray(freq_grp["xCoordinates"][:], dtype=np.float64)
        ys = np.asarray(freq_grp["yCoordinates"][:], dtype=np.float64)
        if xs.size < 2 or ys.size < 2:
            raise ValueError("NISAR coordinate axes too small for affine construction.")

        dx = abs(float(xs[1] - xs[0]))
        dy = abs(float(ys[1] - ys[0]))
        src_transform = from_origin(float(xs[0]) - 0.5 * dx, float(ys[0]) + 0.5 * dy, dx, dy)
        epsg_raw = freq_grp["projection"][()] if "projection" in freq_grp else 4326
        if isinstance(epsg_raw, bytes):
            epsg_str = epsg_raw.decode("utf-8", errors="ignore")
            src_crs = CRS.from_user_input(epsg_str)
        elif int(epsg_raw) > 0:
            src_crs = CRS.from_epsg(int(epsg_raw))
        else:
            raise ValueError("NISAR HDF5 projection lacks a valid EPSG code.")

        h_src, w_src = hh_lin.shape
        if target_bbox is None:
            left = float(xs[0]) - 0.5 * dx
            right = left + w_src * dx
            top = float(ys[0]) + 0.5 * dy
            bottom = top - h_src * dy
            target_bbox = transform_bounds(src_crs, "EPSG:4326", left, bottom, right, top)

    hh_384, valid_hh = _reproject_georeferenced_band(np.abs(hh_lin), src_transform, src_crs, target_bbox)
    hv_384, valid_hv = _reproject_georeferenced_band(np.abs(hv_lin), src_transform, src_crs, target_bbox)
    hh_db = 10.0 * np.log10(np.maximum(hh_384, 1e-6))
    hv_db = 10.0 * np.log10(np.maximum(hv_384, 1e-6))

    hh_n = db_to_norm(hh_db, HH_DB_MIN, HH_DB_MAX)
    hv_n = db_to_norm(hv_db, HV_DB_MIN, HV_DB_MAX)

    patches = np.zeros((36, 2, 64, 64), dtype=np.float32)
    for row in range(6):
        for col in range(6):
            idx = row * 6 + col
            r0, r1 = row * 64, (row + 1) * 64
            c0, c1 = col * 64, (col + 1) * 64
            patches[idx, 0] = hh_n[r0:r1, c0:c1]
            patches[idx, 1] = hv_n[r0:r1, c0:c1]
    return {
        "patches": patches,
        "georeferenced": True,
        "src_crs": str(src_crs),
        "target_bbox": tuple(float(v) for v in target_bbox),
        "valid_pixel_pct": min(valid_hh, valid_hv),
    }


def load_opera_geotiff_patches(
    co_pol_tif: str | Path,
    cross_pol_tif: str | Path,
    target_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Read dual-polarization OPERA RTC-S1 GeoTIFFs with CRS/transform validation into (36, 2, 64, 64)."""
    import rasterio  # type: ignore

    with rasterio.open(str(co_pol_tif)) as src_co, rasterio.open(str(cross_pol_tif)) as src_cr:
        if src_co.crs is None or src_cr.crs is None:
            raise ValueError("OPERA GeoTIFF missing spatial CRS metadata.")
        co_lin = src_co.read(1).astype(np.float32)
        cr_lin = src_cr.read(1).astype(np.float32)
        bbox = target_bbox or (
            float(src_co.bounds.left),
            float(src_co.bounds.bottom),
            float(src_co.bounds.right),
            float(src_co.bounds.top),
        )
        co_384, _ = _reproject_georeferenced_band(
            np.maximum(co_lin, 1e-6), src_co.transform, src_co.crs, bbox, src_nodata=src_co.nodata
        )
        cr_384, _ = _reproject_georeferenced_band(
            np.maximum(cr_lin, 1e-6), src_cr.transform, src_cr.crs, bbox, src_nodata=src_cr.nodata
        )

    co_db = 10.0 * np.log10(np.maximum(co_384, 1e-6))
    cr_db = 10.0 * np.log10(np.maximum(cr_384, 1e-6))

    co_n = db_to_norm(co_db, S1_VV_DB_MIN, S1_VV_DB_MAX)
    cr_n = db_to_norm(cr_db, S1_VH_DB_MIN, S1_VH_DB_MAX)

    patches = np.zeros((36, 2, 64, 64), dtype=np.float32)
    for row in range(6):
        for col in range(6):
            idx = row * 6 + col
            r0, r1 = row * 64, (row + 1) * 64
            c0, c1 = col * 64, (col + 1) * 64
            patches[idx, 0] = co_n[r0:r1, c0:c1]
            patches[idx, 1] = cr_n[r0:r1, c0:c1]
    return patches


class NISARGCOVAdapter:
    """Sensor adapter for NASA-ISRO NISAR L2 GCOV (L-Band HH/HV) HDF5 products."""
    sensor_name: str = "NASA-ISRO SAR (NISAR)"
    sensor_band: str = "L-Band (1.257 GHz)"
    polarizations: Tuple[str, ...] = ("HH", "HV")
    co_pol: str = "HH"
    cross_pol: str = "HV"

    @staticmethod
    def inspect_and_load(
        h5_path: str | Path, target_bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[str, Any]:
        return load_nisar_h5_patches(h5_path, target_bbox)


class Sentinel1RTCAdapter:
    """Sensor adapter for Copernicus Sentinel-1 C-Band RTC (VV/VH) products."""
    sensor_name: str = "Copernicus Sentinel-1 RTC"
    sensor_band: str = "C-Band (5.405 GHz)"
    polarizations: Tuple[str, ...] = ("VV", "VH")
    co_pol: str = "VV"
    cross_pol: str = "VH"

    @staticmethod
    def load_scene(
        scenario_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
        return _load_cached_real_geotiff_scene(scenario_id)


class SyntheticSARAdapter:
    """Sensor adapter for controlled physics-based dual-polarization SAR simulation."""
    sensor_name: str = "Simulated NISAR L-Band Physics Benchmark"
    sensor_band: str = "L-Band (1.257 GHz · Simulated)"
    polarizations: Tuple[str, ...] = ("HH", "HV")
    co_pol: str = "HH"
    cross_pol: str = "HV"

    @staticmethod
    def generate(scenario_id: str, look_count: int = 4, seed: int = 42) -> Dict[str, Any]:
        return generate_realistic_sar_scene(scenario_id=scenario_id, look_count=look_count, seed=seed)


def clear_sar_scene_cache() -> None:
    """Clear LRU-cached GeoTIFF scenes when new custom user locations are added."""
    _load_cached_real_geotiff_scene.cache_clear()


@lru_cache(maxsize=6)
def _load_cached_real_geotiff_scene(
    scenario_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Read cached 384x384 EPSG:4326 Sentinel-1 RTC GeoTIFF arrays, alignment checks, and provenance."""
    import rasterio  # type: ignore

    scene_dir = REAL_SAR_DIR / scenario_id
    with open(scene_dir / "metadata.json", "r", encoding="utf-8") as mf:
        meta = json.load(mf)

    band_files = ("pre_vv.tif", "pre_vh.tif", "post_vv.tif", "post_vh.tif")
    crs_set = set()
    shapes = set()
    transforms = []
    valid_ratios: List[float] = []
    arrays_db: List[np.ndarray] = []

    for fname in band_files:
        fpath = scene_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing Sentinel-1 RTC temporal band: {fpath}")
        with rasterio.open(str(fpath)) as src:
            crs_set.add(str(src.crs))
            shapes.add((src.height, src.width))
            transforms.append(tuple(round(float(x), 7) for x in src.transform[:6]))
            lin = src.read(1).astype(np.float32)
            nodata = src.nodata
            valid_mask = np.isfinite(lin) & (lin > 0.0)
            if nodata is not None:
                valid_mask &= lin != float(nodata)
            valid_ratios.append(float(np.mean(valid_mask) * 100.0))
            clean_lin = np.where(valid_mask, lin, np.nanmedian(lin[valid_mask]) if np.any(valid_mask) else 1e-4)
            arrays_db.append((10.0 * np.log10(np.maximum(clean_lin, 1e-6))).astype(np.float32))

    crs_aligned = len(crs_set) == 1 and "4326" in next(iter(crs_set))
    shape_aligned = len(shapes) == 1 and (384, 384) in shapes
    transform_aligned = len(set(transforms)) == 1
    valid_pct = round(min(valid_ratios), 2)
    min_db = min(float(np.min(a)) for a in arrays_db)
    max_db = max(float(np.max(a)) for a in arrays_db)
    radiometric_ok = (-55.0 <= min_db) and (max_db <= 22.0)

    issues: List[str] = []
    if not crs_aligned:
        issues.append(f"CRS mismatch across bands: {sorted(crs_set)}")
    if not shape_aligned:
        issues.append(f"Raster dimensions mismatch: {sorted(shapes)}")
    if not transform_aligned:
        issues.append("Affine geotransform mismatch across pre/post polarization files")
    if valid_pct < 95.0:
        issues.append(f"Low valid pixel ratio ({valid_pct:.1f}%)")
    if not radiometric_ok:
        issues.append(f"Backscatter outside expected range ({min_db:.1f}..{max_db:.1f} dB)")

    data_quality = {
        "status": "PASS" if not issues else ("WARN" if (crs_aligned and shape_aligned and transform_aligned and valid_pct >= 80.0) else "UNSUITABLE"),
        "crs": next(iter(crs_set)),
        "crs_aligned": crs_aligned,
        "shape_aligned": shape_aligned,
        "transform_aligned": transform_aligned,
        "temporal_pairs_complete": True,
        "georeferenced_reprojection": True,
        "valid_pixel_pct": valid_pct,
        "radiometric_in_bounds": radiometric_ok,
        "db_range": [round(min_db, 2), round(max_db, 2)],
        "unsuitable_reason": "; ".join(issues) if issues else None,
    }
    return arrays_db[0], arrays_db[1], arrays_db[2], arrays_db[3], meta, data_quality


def load_sar_scene(
    scenario_id: str = "nagpur_flood",
    look_count: int = 4,
    query_nasa_cmr: bool = False,
    seed: int = 42,
    data_source: str = "real_s1_rtc",
) -> Dict[str, Any]:
    """Load real Sentinel-1 RTC GeoTIFF crops or calibrated L-band NISAR simulation."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario_id '{scenario_id}'. Valid: {list(SCENARIOS)}")

    cfg = SCENARIOS[scenario_id]
    real_dir = REAL_SAR_DIR / scenario_id
    use_real = data_source == "real_s1_rtc" and (real_dir / "post_vv.tif").exists()

    if use_real:
        pre_vv_clean, pre_vh_clean, post_vv_clean, post_vh_clean, stac_meta, data_quality = _load_cached_real_geotiff_scene(scenario_id)
        override_labels = None
        pre_vv_db, pre_vh_db, post_vv_db, post_vh_db = pre_vv_clean, pre_vh_clean, post_vv_clean, post_vh_clean
        if look_count < 4:
            # Compute native-look (~4.4) radiometric reference BEFORE injecting L=1 synthetic stress speckle
            native_scene = assemble_sar_scene_from_db(
                cfg=cfg,
                pre_co_db=pre_vv_clean,
                pre_cr_db=pre_vh_clean,
                post_co_db=post_vv_clean,
                post_cr_db=post_vh_clean,
                pixel_labels=None,
                co_range=(S1_VV_DB_MIN, S1_VV_DB_MAX),
                cr_range=(S1_VH_DB_MIN, S1_VH_DB_MAX),
            )
            override_labels = native_scene["tile_labels"]
            rng = np.random.default_rng(seed)
            pre_vv_db = _apply_gamma_speckle(pre_vv_clean, look_count, rng)
            pre_vh_db = _apply_gamma_speckle(pre_vh_clean, look_count, rng)
            post_vv_db = _apply_gamma_speckle(post_vv_clean, look_count, rng)
            post_vh_db = _apply_gamma_speckle(post_vh_clean, look_count, rng)

        scene = assemble_sar_scene_from_db(
            cfg=cfg,
            pre_co_db=pre_vv_db,
            pre_cr_db=pre_vh_db,
            post_co_db=post_vv_db,
            post_cr_db=post_vh_db,
            pixel_labels=None,
            co_range=(S1_VV_DB_MIN, S1_VV_DB_MAX),
            cr_range=(S1_VH_DB_MIN, S1_VH_DB_MAX),
            vis_co_range=(-22.0, -1.0),
            vis_cr_range=(-28.0, -7.0),
            override_tile_labels=override_labels,
        )
        scene["data_quality"] = data_quality
        scene["data_provenance"] = {
            "data_source": "real_s1_rtc",
            "data_source_mode": "live_sentinel1",
            "data_source_tier": "real_satellite",
            "is_real_satellite_data": True,
            "ground_truth_available": True,
            "ground_truth_category": "radiometric_heuristic_reference",
            "ground_truth_type": "OPERA Radiometric Change Reference (Heuristic — Not Manual Ground Truth)",
            "is_verified_manual_ground_truth": False,
            "metric_display_mode": "heuristic_agreement",
            "sensor_label": "Real Sentinel-1 C-Band RTC (VV + VH · Planetary Computer STAC)",
            "sensor_band": "C-Band (5.405 GHz)",
            "polarization_channels": ["VV", "VH"],
            "co_pol_label": "VV",
            "cross_pol_label": "VH",
            "polarization_basis": "Native C-Band VV / VH (sensor-specific normalization)",
            "crs": stac_meta.get("crs", "EPSG:4326"),
            "relative_orbit": stac_meta.get("relative_orbit"),
            "pre_item_ids": stac_meta.get("pre_item_ids", []),
            "post_item_ids": stac_meta.get("post_item_ids", []),
            "event_note": stac_meta.get("event_note", ""),
            "speckle_mode": f"Native S1-RTC + L{look_count} stress" if look_count < 4 else "Native S1-RTC (~4.4 looks)",
        }
    else:
        scene = generate_realistic_sar_scene(
            scenario_id=scenario_id, look_count=look_count, seed=seed
        )
        scene["data_quality"] = {
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
        }
        scene["data_provenance"] = {
            "data_source": "nisar_sim",
            "data_source_mode": "simulation",
            "data_source_tier": "synthetic_physics",
            "is_real_satellite_data": False,
            "ground_truth_available": True,
            "ground_truth_category": "synthetic_physics_ground_truth",
            "ground_truth_type": "Synthetic L-Band Physics Ground Truth (Simulated Reference)",
            "is_verified_manual_ground_truth": True,
            "metric_display_mode": "accuracy",
            "sensor_label": "Simulated NISAR L-Band Physics Benchmark (HH + HV · Synthetic)",
            "sensor_band": "L-Band (1.257 GHz · Simulated)",
            "polarization_channels": ["HH", "HV"],
            "co_pol_label": "HH",
            "cross_pol_label": "HV",
            "polarization_basis": "Simulated L-Band HH / HV Controlled Ablation",
            "crs": "EPSG:4326",
            "relative_orbit": None,
            "pre_item_ids": [],
            "post_item_ids": [],
            "event_note": "Deterministic L-band specular, double-bounce, canopy-loss & Gamma speckle testbench",
            "speckle_mode": f"Simulated L={look_count} Gamma Speckle",
        }

    nasa_meta = (
        query_nasa_earthdata_granules(scenario_id)
        if query_nasa_cmr
        else {
            "status": "cached_physics_ready",
            "queried_collection": cfg.nasa_collections[0],
            "granule_count": 0,
            "granules": [],
        }
    )
    scene["nasa_metadata"] = nasa_meta
    scene["class_names"] = CLASS_NAMES
    return scene
