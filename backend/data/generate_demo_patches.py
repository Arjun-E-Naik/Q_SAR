"""Physics-accurate pre/post dual-polarization SAR patch generator & temporal intelligence.

Generates spatially continuous 384x384 (6x6 grid of 64x64 patches) dual-pol
radar backscatter scenes modeling:
- Specular reflection on open floodwater vs. permanent river channels
- Dihedral double-bounce in flooded riparian vegetation (high co-pol increase)
- Cross-pol volume scattering collapse in wildfire burn scars
- Dielectric & structural shifts in agricultural crop/soil stress
- Multiplicative Gamma L-look radar speckle noise in linear power domain
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

CLASS_NAMES: Dict[int, str] = {
    0: "Normal Terrain",
    1: "Flood Inundation",
    2: "Wildfire Burn Scar",
    3: "Crop/Soil Stress",
}

HH_DB_MIN, HH_DB_MAX = -30.0, 2.0
HV_DB_MIN, HV_DB_MAX = -35.0, -5.0
S1_VV_DB_MIN, S1_VV_DB_MAX = -24.0, 0.0
S1_VH_DB_MIN, S1_VH_DB_MAX = -30.0, -6.0


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    title: str
    subtitle: str
    region: str
    sensor_mode: str
    nasa_collections: Tuple[str, ...]
    bbox: Tuple[float, float, float, float]  # (west, south, east, north)
    pre_date: str
    post_date: str
    start_node: Tuple[int, int]
    end_node: Tuple[int, int]
    start_label: str
    end_label: str
    tile_km: float


SCENARIOS: Dict[str, ScenarioConfig] = {
    "nagpur_flood": ScenarioConfig(
        scenario_id="nagpur_flood",
        title="Wainganga / Gosikhurd Monsoon Flood Basin",
        subtitle="Multi-Temporal Dual-Pol SAR Flood & Flooded-Vegetation Mapping",
        region="Wainganga–Gosikhurd Basin, Nagpur, Maharashtra, India",
        sensor_mode="Sentinel-1 C-Band RTC (VV+VH) / NISAR L-Band (HH+HV)",
        nasa_collections=(
            "NISAR_L2_GCOV_V1",
            "OPERA_L2_RTC-S1_V1",
            "NISAR_L2_GCOV_PROVISIONAL_V1",
            "NISAR_L2_GCOV_BETA_V1",
        ),
        bbox=(79.48, 20.80, 79.72, 21.04),
        pre_date="2026-05-23",
        post_date="2026-08-22",
        start_node=(0, 0),
        end_node=(5, 5),
        start_label="Kuhi Relief Staging (NW)",
        end_label="Gosikhurd / Pauni Flood Sector (SE)",
        tile_km=4.2,
    ),
    "california_wildfire": ScenarioConfig(
        scenario_id="california_wildfire",
        title="Butte–Tehama Park Fire Burn Scar",
        subtitle="Multi-Temporal Cross-Pol Canopy Disturbance & Perimeter Mapping",
        region="Cohasset–Butte Creek Foothills, Northern California, USA",
        sensor_mode="Sentinel-1 C-Band RTC (VV+VH) / NISAR L-Band (HH+HV)",
        nasa_collections=(
            "NISAR_L2_GCOV_V1",
            "OPERA_L2_RTC-S1_V1",
            "NISAR_L2_GCOV_PROVISIONAL_V1",
            "NISAR_L2_GCOV_BETA_V1",
        ),
        bbox=(-121.84, 39.88, -121.60, 40.12),
        pre_date="2024-07-06",
        post_date="2024-08-23",
        start_node=(0, 1),
        end_node=(5, 5),
        start_label="Pine Creek Staging Base (NW)",
        end_label="Butte Creek Safe Zone (SE)",
        tile_km=4.0,
    ),
    "vidarbha_crop_stress": ScenarioConfig(
        scenario_id="vidarbha_crop_stress",
        title="Vidarbha Kharif Crop & Soil Moisture Stress",
        subtitle="Multi-Temporal Agricultural Surface & Wetland Change Tracking",
        region="Wardha–Yavatmal Agrarian Belt, Maharashtra, India",
        sensor_mode="Sentinel-1 C-Band RTC (VV+VH) / NISAR L-Band (HH+HV)",
        nasa_collections=(
            "NISAR_L2_GCOV_V1",
            "OPERA_L2_RTC-S1_V1",
            "NISAR_L3_SME2_PROVISIONAL_V1",
            "NISAR_L2_GCOV_PROVISIONAL_V1",
        ),
        bbox=(78.40, 20.68, 78.64, 20.92),
        pre_date="2026-06-28",
        post_date="2026-10-21",
        start_node=(0, 0),
        end_node=(5, 4),
        start_label="Wardha Agri-Extension Hub (NW)",
        end_label="Hinganghat Field Cluster (SE)",
        tile_km=4.2,
    ),
}

_CUSTOM_SCENARIOS_PATH = Path(__file__).resolve().parent / "custom_scenarios.json"


def register_custom_scenario(
    scenario_id: str,
    region: str,
    bbox: Tuple[float, float, float, float],
    pre_date: str,
    post_date: str,
    title: Optional[str] = None,
    persist: bool = True,
) -> ScenarioConfig:
    """Register (and optionally persist) a user-selected custom ROI into `SCENARIOS`."""
    west, south, east, north = [round(float(v), 5) for v in bbox]
    mid_lat = 0.5 * (south + north)
    tile_km = round(max(0.5, (abs(north - south) * 111.132) / 6.0), 2)
    clean_region = region.strip() or f"{mid_lat:.3f}°, {(west + east) * 0.5:.3f}°"
    cfg = ScenarioConfig(
        scenario_id=scenario_id,
        title=title or f"Custom SAR Watch — {clean_region.split(',')[0][:42]}",
        subtitle="On-Demand Bi-Temporal Dual-Pol SAR Anomaly & Emergency Routing",
        region=clean_region,
        sensor_mode="Sentinel-1 C-Band RTC (VV+VH) / NISAR L-Band (HH+HV)",
        nasa_collections=(
            "OPERA_L2_RTC-S1_V1",
            "NISAR_L2_GCOV_V1",
        ),
        bbox=(west, south, east, north),
        pre_date=pre_date,
        post_date=post_date,
        start_node=(0, 0),
        end_node=(5, 5),
        start_label=f"NW Staging ({north:.3f}°, {west:.3f}°)",
        end_label=f"SE Objective ({south:.3f}°, {east:.3f}°)",
        tile_km=tile_km,
    )
    SCENARIOS[scenario_id] = cfg

    if persist:
        existing: Dict[str, Any] = {}
        if _CUSTOM_SCENARIOS_PATH.exists():
            try:
                existing = json.loads(_CUSTOM_SCENARIOS_PATH.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing[scenario_id] = {
            "scenario_id": cfg.scenario_id,
            "title": cfg.title,
            "region": cfg.region,
            "bbox": list(cfg.bbox),
            "pre_date": cfg.pre_date,
            "post_date": cfg.post_date,
        }
        _CUSTOM_SCENARIOS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CUSTOM_SCENARIOS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return cfg


def _load_persisted_custom_scenarios() -> None:
    if not _CUSTOM_SCENARIOS_PATH.exists():
        return
    try:
        saved = json.loads(_CUSTOM_SCENARIOS_PATH.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            for sid, item in saved.items():
                if sid not in SCENARIOS and isinstance(item, dict) and len(item.get("bbox", [])) == 4:
                    register_custom_scenario(
                        scenario_id=str(sid),
                        region=str(item.get("region", sid)),
                        bbox=tuple(float(v) for v in item["bbox"]),  # type: ignore[arg-type]
                        pre_date=str(item.get("pre_date", "")),
                        post_date=str(item.get("post_date", "")),
                        title=str(item.get("title") or "") or None,
                        persist=False,
                    )
    except Exception:
        pass


_load_persisted_custom_scenarios()


def db_to_norm(db_val: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """Normalize radar backscatter in dB into [0, 1]."""
    return np.clip((db_val - db_min) / (db_max - db_min), 0.0, 1.0).astype(np.float32)


def norm_to_db(norm_val: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """Invert [0, 1] normalization back to dB."""
    return (norm_val * (db_max - db_min) + db_min).astype(np.float32)


def _apply_gamma_speckle(
    db_clean: np.ndarray, look_count: int, rng: np.random.Generator
) -> np.ndarray:
    """Convert clean dB field to linear power, apply L-look Gamma speckle, return dB."""
    lin_clean = np.power(10.0, db_clean / 10.0)
    looks = max(1, int(look_count))
    speckle = rng.gamma(shape=float(looks), scale=1.0 / float(looks), size=db_clean.shape)
    lin_noisy = np.maximum(lin_clean * speckle, 1e-6)
    return (10.0 * np.log10(lin_noisy)).astype(np.float32)


def _fbm_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth multi-octave deterministic spatial texture in [-1, 1]."""
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    field = np.zeros((h, w), dtype=np.float32)
    for freq, amp in ((3.0, 0.55), (7.0, 0.30), (15.0, 0.15)):
        px, py = rng.uniform(0, 2 * np.pi, size=2)
        field += amp * np.sin(2 * np.pi * freq * xx + px) * np.cos(2 * np.pi * freq * yy + py)
    return field


def _array_to_png_b64(img_uint8: np.ndarray) -> str:
    """Encode an HxW (grayscale) or HxWx3 (RGB) uint8 array as a fast data-URI PNG."""
    mode = "L" if img_uint8.ndim == 2 else "RGB"
    sampled = img_uint8[::2, ::2] if img_uint8.shape[0] >= 384 else img_uint8
    pil_img = Image.fromarray(sampled, mode=mode)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", compress_level=1)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _derive_radiometric_reference_class(
    new_flood_pct: float,
    water_pct: float,
    double_bounce_pct: float,
    burn_pct: float,
    crop_change_pct: float,
    delta_co_db: float,
    delta_cr_db: float,
    delta_rvi: float,
) -> int:
    """OPERA-style radiometric change reference label (0..3) from real pre/post shifts.

    Uses 4x4 linear-power multi-looked sub-masks to separate new flood & strict double-bounce
    inundation (Class 1), non-water cross-pol canopy collapse (Class 2), and significant
    agricultural moisture/structural shifts (Class 3) from unchanged land/river (Class 0).
    """
    db_excess = delta_co_db - delta_cr_db
    if (
        new_flood_pct >= 5.5
        or (water_pct >= 18.0 and (db_excess >= 2.0 or new_flood_pct >= 2.5))
        or (double_bounce_pct >= 16.0 and db_excess >= 1.75 and delta_cr_db < 0.6)
    ):
        return 1
    if (burn_pct >= 22.0 and water_pct < 12.0 and double_bounce_pct < 15.0) or (
        delta_cr_db <= -1.7 and delta_cr_db <= delta_co_db - 0.55 and water_pct < 10.0
    ):
        return 2
    if (
        crop_change_pct >= 22.0
        or (delta_co_db >= 2.1 and delta_cr_db >= 1.1)
        or (abs(delta_co_db) >= 1.5 and delta_rvi >= 0.05)
        or (12.0 <= burn_pct < 22.0 and water_pct < 12.0)
    ):
        return 3
    return 0


@lru_cache(maxsize=8)
def _cached_scene_fields(
    scenario_id: str, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache deterministic clean terrain fields while applying fresh Gamma speckle per request."""
    return _build_scene_fields(scenario_id, np.random.default_rng(seed))


def _build_scene_fields(
    scenario_id: str, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct spatially continuous pre/post clean dB fields and per-pixel class map."""
    h = w = 384
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    tex = _fbm_field(h, w, rng)

    pre_hh_db = -10.5 + 1.8 * tex
    pre_hv_db = -14.2 + 1.5 * tex
    post_hh_db = pre_hh_db.copy()
    post_hv_db = pre_hv_db.copy()
    class_map = np.zeros((h, w), dtype=np.int64)

    if scenario_id == "nagpur_flood":
        river_center = 0.45 + 0.14 * np.sin(2.0 * np.pi * xx + 0.3) + 0.05 * tex
        dist_river = np.abs((yy - (0.18 + 0.68 * xx)) * 0.72 + (yy - river_center) * 0.28)
        depot_clear = ((xx < 0.18) & (yy < 0.18)) | ((xx > 0.82) & (yy > 0.82))

        open_flood = (dist_river < 0.095) & (~depot_clear)
        double_bounce_rim = (dist_river >= 0.095) & (dist_river < 0.165) & (~depot_clear)
        crop_stress = (dist_river >= 0.165) & (dist_river < 0.255) & (xx > 0.20) & (~depot_clear)

        pre_channel = dist_river < 0.032
        pre_hh_db[pre_channel] = -21.5 + 1.2 * tex[pre_channel]
        pre_hv_db[pre_channel] = -28.5 + 1.2 * tex[pre_channel]

        post_hh_db[open_flood] = -23.5 + 1.2 * tex[open_flood]
        post_hv_db[open_flood] = -30.2 + 1.0 * tex[open_flood]

        post_hh_db[double_bounce_rim] = -3.2 + 1.3 * tex[double_bounce_rim]
        post_hv_db[double_bounce_rim] = -22.5 + 1.1 * tex[double_bounce_rim]

        row_mod = 0.9 * np.sin(2.0 * np.pi * 24.0 * xx)
        post_hh_db[crop_stress] = -15.6 + row_mod[crop_stress] + 0.9 * tex[crop_stress]
        post_hv_db[crop_stress] = -20.8 + 0.9 * tex[crop_stress]

        class_map[open_flood | double_bounce_rim] = 1
        class_map[crop_stress] = 3

    elif scenario_id == "california_wildfire":
        d1 = np.sqrt(((xx - 0.44) / 0.24) ** 2 + ((yy - 0.46) / 0.21) ** 2) + 0.06 * tex
        d2 = np.sqrt(((xx - 0.64) / 0.19) ** 2 + ((yy - 0.62) / 0.18) ** 2) + 0.06 * tex
        depot_clear = ((xx > 0.80) & (yy < 0.20)) | ((xx < 0.20) & (yy > 0.80))

        burn_scar = ((d1 < 1.0) | (d2 < 1.0)) & (~depot_clear)
        dry_chaparral = (~burn_scar) & ((d1 < 1.32) | (d2 < 1.32)) & (~depot_clear)
        creek_flood = (np.abs(yy - (0.78 - 0.18 * xx)) < 0.045) & (xx > 0.25) & (xx < 0.72) & (~burn_scar)

        pre_hh_db = -9.5 + 1.6 * tex
        pre_hv_db = -12.8 + 1.4 * tex
        post_hh_db = pre_hh_db.copy()
        post_hv_db = pre_hv_db.copy()

        post_hh_db[burn_scar] = -11.4 + 2.4 * tex[burn_scar]
        post_hv_db[burn_scar] = -25.2 + 1.4 * tex[burn_scar]

        post_hh_db[dry_chaparral] = -15.2 + 1.1 * tex[dry_chaparral]
        post_hv_db[dry_chaparral] = -20.4 + 1.1 * tex[dry_chaparral]

        post_hh_db[creek_flood] = -22.8 + 1.0 * tex[creek_flood]
        post_hv_db[creek_flood] = -29.5 + 0.9 * tex[creek_flood]

        class_map[burn_scar] = 2
        class_map[dry_chaparral] = 3
        class_map[creek_flood] = 1

    else:  # vidarbha_crop_stress
        parcel_x = np.floor(xx * 6.0)
        parcel_y = np.floor(yy * 6.0)
        stressed_mask = ((parcel_x + parcel_y) % 2 == 1) | (
            (xx > 0.25) & (xx < 0.72) & (yy > 0.30) & (yy < 0.70)
        )
        depot_clear = ((xx < 0.18) & (yy < 0.18)) | ((xx > 0.66) & (xx < 0.84) & (yy > 0.82))
        stressed = stressed_mask & (~depot_clear)

        nullah = (np.abs(yy - (0.50 + 0.15 * np.sin(2.0 * np.pi * xx))) < 0.065) & (xx > 0.18) & (xx < 0.85)
        small_burn = np.sqrt((xx - 0.76) ** 2 + (yy - 0.26) ** 2) < 0.11

        row_mod = 1.1 * np.sin(2.0 * np.pi * 22.0 * yy)
        post_hh_db[stressed] = -15.8 + row_mod[stressed] + 0.8 * tex[stressed]
        post_hv_db[stressed] = -21.0 + 0.9 * tex[stressed]

        post_hh_db[nullah] = -23.0 + 1.0 * tex[nullah]
        post_hv_db[nullah] = -29.6 + 1.0 * tex[nullah]

        post_hh_db[small_burn] = -11.8 + 2.0 * tex[small_burn]
        post_hv_db[small_burn] = -24.8 + 1.2 * tex[small_burn]

        class_map[stressed] = 3
        class_map[nullah] = 1
        class_map[small_burn] = 2

    return pre_hh_db, pre_hv_db, post_hh_db, post_hv_db, class_map


def _multilook_4x4_db(patch_db: np.ndarray) -> np.ndarray:
    """Apply 4x4 linear-power multi-looking (64x64 -> 16x16) to suppress native speckle before thresholding."""
    lin = np.power(10.0, np.clip(patch_db, -45.0, 15.0) / 10.0)
    pooled = lin.reshape(16, 4, 16, 4).mean(axis=(1, 3))
    return (10.0 * np.log10(np.maximum(pooled, 1e-6))).astype(np.float32)


def assemble_sar_scene_from_db(
    cfg: ScenarioConfig,
    pre_co_db: np.ndarray,
    pre_cr_db: np.ndarray,
    post_co_db: np.ndarray,
    post_cr_db: np.ndarray,
    pixel_labels: Optional[np.ndarray] = None,
    co_range: Tuple[float, float] = (HH_DB_MIN, HH_DB_MAX),
    cr_range: Tuple[float, float] = (HV_DB_MIN, HV_DB_MAX),
    vis_co_range: Optional[Tuple[float, float]] = None,
    vis_cr_range: Optional[Tuple[float, float]] = None,
    override_tile_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build (36, 2, 64, 64) pre/post dual-pol tensors, temporal telemetry & PNG overlays."""
    v_co_range = vis_co_range or co_range
    v_cr_range = vis_cr_range or cr_range

    pre_co_m = db_to_norm(pre_co_db, co_range[0], co_range[1])
    pre_cr_m = db_to_norm(pre_cr_db, cr_range[0], cr_range[1])
    post_co_m = db_to_norm(post_co_db, co_range[0], co_range[1])
    post_cr_m = db_to_norm(post_cr_db, cr_range[0], cr_range[1])

    pre_patches = np.zeros((36, 2, 64, 64), dtype=np.float32)
    post_patches = np.zeros((36, 2, 64, 64), dtype=np.float32)
    tile_labels = np.full((36,), -1, dtype=np.int64)
    tile_metrics: List[Dict[str, Any]] = []

    west, south, east, north = cfg.bbox
    lon_step = (east - west) / 6.0
    lat_step = (north - south) / 6.0
    reference_type = "synthetic_ground_truth" if pixel_labels is not None else "opera_radiometric_change_heuristic"

    for row in range(6):
        for col in range(6):
            idx = row * 6 + col
            r0, r1 = row * 64, (row + 1) * 64
            c0, c1 = col * 64, (col + 1) * 64

            pre_patches[idx, 0] = pre_co_m[r0:r1, c0:c1]
            pre_patches[idx, 1] = pre_cr_m[r0:r1, c0:c1]
            post_patches[idx, 0] = post_co_m[r0:r1, c0:c1]
            post_patches[idx, 1] = post_cr_m[r0:r1, c0:c1]

            post_co_t = post_co_db[r0:r1, c0:c1]
            post_cr_t = post_cr_db[r0:r1, c0:c1]
            pre_co_t = pre_co_db[r0:r1, c0:c1]
            pre_cr_t = pre_cr_db[r0:r1, c0:c1]

            co_mean_db = float(np.mean(post_co_t))
            cr_mean_db = float(np.mean(post_cr_t))
            pre_co_mean_db = float(np.mean(pre_co_t))
            pre_cr_mean_db = float(np.mean(pre_cr_t))
            delta_co_db = co_mean_db - pre_co_mean_db
            delta_cr_db = cr_mean_db - pre_cr_mean_db

            pre_co_lin = float(np.mean(np.power(10.0, pre_co_t / 10.0)))
            pre_cr_lin = float(np.mean(np.power(10.0, pre_cr_t / 10.0)))
            pre_rvi = float((4.0 * pre_cr_lin) / max(pre_co_lin + pre_cr_lin, 1e-6))

            co_lin = float(np.mean(np.power(10.0, post_co_t / 10.0)))
            cr_lin = float(np.mean(np.power(10.0, post_cr_t / 10.0)))
            rvi = float((4.0 * cr_lin) / max(co_lin + cr_lin, 1e-6))
            delta_rvi = rvi - pre_rvi

            # 4x4 multi-looked sub-block change decomposition (256 superpixels per tile)
            pco_ml = _multilook_4x4_db(pre_co_t)
            pcr_ml = _multilook_4x4_db(pre_cr_t)
            qco_ml = _multilook_4x4_db(post_co_t)
            qcr_ml = _multilook_4x4_db(post_cr_t)
            dco_ml = qco_ml - pco_ml
            dcr_ml = qcr_ml - pcr_ml

            permanent_water_mask = (pco_ml < -14.0) & (qco_ml < -14.0)
            new_flood_mask = (pco_ml >= -13.5) & ((qco_ml < -14.0) | (dco_ml < -2.6))
            total_water_mask = permanent_water_mask | new_flood_mask
            # Strict emergent vegetation double-bounce / wind-ruffled water: strong +Co surge without matching +Cross surge
            double_bounce_mask = (
                (dco_ml > 2.2)
                & (qco_ml > -12.5)
                & ((dco_ml - dcr_ml) > 1.8)
                & (dcr_ml < 0.8)
                & (~permanent_water_mask)
            )
            burn_mask = (
                (dcr_ml < -2.0)
                & (dcr_ml <= dco_ml - 0.4)
                & (~total_water_mask)
                & (~double_bounce_mask)
                & (qco_ml >= -14.5)
            )
            crop_change_mask = (
                ((np.abs(dco_ml) > 2.0) | (np.abs(dcr_ml) > 2.0))
                & (~total_water_mask)
                & (~burn_mask)
                & (~double_bounce_mask)
            )

            permanent_water_pct = float(np.mean(permanent_water_mask) * 100.0)
            new_flood_pct = float(np.mean(new_flood_mask) * 100.0)
            water_pct = float(np.mean(total_water_mask) * 100.0)
            double_bounce_pct = float(np.mean(double_bounce_mask) * 100.0)
            burn_pct = float(np.mean(burn_mask) * 100.0)
            crop_change_pct = float(np.mean(crop_change_mask) * 100.0)

            if override_tile_labels is not None:
                tile_labels[idx] = int(override_tile_labels[idx])
                reference_type = "opera_radiometric_change_heuristic" if pixel_labels is None else "synthetic_ground_truth"
            elif pixel_labels is not None:
                patch_lbls = pixel_labels[r0:r1, c0:c1].ravel()
                counts = np.bincount(patch_lbls, minlength=4)
                if counts[1] > 0.22 * 4096:
                    tile_labels[idx] = 1
                elif counts[2] > 0.22 * 4096:
                    tile_labels[idx] = 2
                elif counts[3] > 0.30 * 4096:
                    tile_labels[idx] = 3
                else:
                    tile_labels[idx] = int(np.argmax(counts))
            else:
                tile_labels[idx] = _derive_radiometric_reference_class(
                    new_flood_pct=new_flood_pct,
                    water_pct=water_pct,
                    double_bounce_pct=double_bounce_pct,
                    burn_pct=burn_pct,
                    crop_change_pct=crop_change_pct,
                    delta_co_db=delta_co_db,
                    delta_cr_db=delta_cr_db,
                    delta_rvi=delta_rvi,
                )

            tile_west = west + col * lon_step
            tile_east = tile_west + lon_step
            tile_north = north - row * lat_step
            tile_south = tile_north - lat_step

            tile_metrics.append(
                {
                    "tile_id": idx,
                    "row": row,
                    "col": col,
                    "center_lon": round((tile_west + tile_east) * 0.5, 6),
                    "center_lat": round((tile_south + tile_north) * 0.5, 6),
                    "bounds": [
                        round(tile_west, 6),
                        round(tile_south, 6),
                        round(tile_east, 6),
                        round(tile_north, 6),
                    ],
                    "hh_db": round(co_mean_db, 2),
                    "hv_db": round(cr_mean_db, 2),
                    "pre_hh_db": round(pre_co_mean_db, 2),
                    "pre_hv_db": round(pre_cr_mean_db, 2),
                    "delta_hh_db": round(delta_co_db, 2),
                    "delta_hv_db": round(delta_cr_db, 2),
                    "cross_pol_ratio_db": round(cr_mean_db - co_mean_db, 2),
                    "rvi": round(float(np.clip(rvi, 0.0, 1.5)), 3),
                    "delta_rvi": round(float(np.clip(delta_rvi, -1.0, 1.0)), 3),
                    "water_pct": round(water_pct, 1),
                    "permanent_water_pct": round(permanent_water_pct, 1),
                    "new_flood_pct": round(new_flood_pct, 1),
                    "burn_pct": round(burn_pct, 1),
                    "double_bounce_pct": round(double_bounce_pct, 1),
                    "crop_change_pct": round(crop_change_pct, 1),
                    "reference_class": int(tile_labels[idx]),
                    "radiometric_reference_label": int(tile_labels[idx]),
                    "verified_ground_truth_label": None,
                    "reference_type": reference_type,
                }
            )

    pre_co_v = db_to_norm(pre_co_db, v_co_range[0], v_co_range[1])
    post_co_v = db_to_norm(post_co_db, v_co_range[0], v_co_range[1])
    post_cr_v = db_to_norm(post_cr_db, v_cr_range[0], v_cr_range[1])

    hh_pre_u8 = np.clip(pre_co_v * 255.0, 0, 255).astype(np.uint8)
    hh_post_u8 = np.clip(post_co_v * 255.0, 0, 255).astype(np.uint8)
    hv_post_u8 = np.clip(post_cr_v * 255.0, 0, 255).astype(np.uint8)

    asf_b = np.clip(post_co_v - post_cr_v + 0.45, 0.0, 1.0)
    asf_rgb_u8 = np.clip(np.stack([post_co_v, post_cr_v, asf_b], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    change_rgb_u8 = np.clip(np.stack([post_co_v, pre_co_v, pre_co_v], axis=-1) * 255.0, 0, 255).astype(np.uint8)

    return {
        "scenario": cfg,
        "pre_patches": pre_patches,
        "post_patches": post_patches,
        "tile_labels": tile_labels,
        "tile_metrics": tile_metrics,
        "images": {
            "hh_pre": _array_to_png_b64(hh_pre_u8),
            "hh_post": _array_to_png_b64(hh_post_u8),
            "hv_post": _array_to_png_b64(hv_post_u8),
            "asf_rgb": _array_to_png_b64(asf_rgb_u8),
            "change_rgb": _array_to_png_b64(change_rgb_u8),
        },
    }


def generate_realistic_sar_scene(
    scenario_id: str = "nagpur_flood",
    look_count: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a 6x6 grid (36 x 2 x 64 x 64) of pre/post dual-pol SAR backscatter patches."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario_id '{scenario_id}'. Valid: {list(SCENARIOS)}")

    cfg = SCENARIOS[scenario_id]
    rng = np.random.default_rng(seed)

    pre_hh_clean, pre_hv_clean, post_hh_clean, post_hv_clean, pixel_labels = _cached_scene_fields(
        scenario_id, seed
    )

    pre_hh_db = _apply_gamma_speckle(pre_hh_clean, look_count, rng)
    pre_hv_db = _apply_gamma_speckle(pre_hv_clean, look_count, rng)
    post_hh_db = _apply_gamma_speckle(post_hh_clean, look_count, rng)
    post_hv_db = _apply_gamma_speckle(post_hv_clean, look_count, rng)

    return assemble_sar_scene_from_db(
        cfg=cfg,
        pre_co_db=pre_hh_db,
        pre_cr_db=pre_hv_db,
        post_co_db=post_hh_db,
        post_cr_db=post_hv_db,
        pixel_labels=pixel_labels,
        co_range=(HH_DB_MIN, HH_DB_MAX),
        cr_range=(HV_DB_MIN, HV_DB_MAX),
    )


def generate_bitemporal_training_batch(
    num_samples: int = 240,
    look_count: Union[int, Tuple[int, ...]] = 4,
    seed: int = 123,
    include_cband: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired pre/post dual-pol training patches across L-band and C-band regimes.

    Returns (pre_patches, post_patches, labels) where pre/post have shape (N, 2, 64, 64).
    """
    rng = np.random.default_rng(seed)
    pre_patches = np.zeros((num_samples, 2, 64, 64), dtype=np.float32)
    post_patches = np.zeros((num_samples, 2, 64, 64), dtype=np.float32)
    labels = np.zeros((num_samples,), dtype=np.int64)
    yy, xx = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64), indexing="ij")
    look_choices = (look_count,) if isinstance(look_count, int) else tuple(look_count)

    for i in range(num_samples):
        cls = i % 4
        labels[i] = cls
        tex = _fbm_field(64, 64, rng)
        is_cband = include_cband and ((i // 4) % 2 == 1)
        patch_looks = int(look_choices[(i // 8) % len(look_choices)])

        if is_cband:
            co_range, cr_range = (S1_VV_DB_MIN, S1_VV_DB_MAX), (S1_VH_DB_MIN, S1_VH_DB_MAX)
            pre_co_c = rng.uniform(-10.8, -7.2) + 1.4 * tex
            pre_cr_c = rng.uniform(-17.0, -13.2) + 1.2 * tex
        else:
            co_range, cr_range = (HH_DB_MIN, HH_DB_MAX), (HV_DB_MIN, HV_DB_MAX)
            pre_co_c = rng.uniform(-11.5, -8.5) + 1.7 * tex
            pre_cr_c = rng.uniform(-15.5, -12.0) + 1.4 * tex

        post_co_c = pre_co_c.copy()
        post_cr_c = pre_cr_c.copy()

        if cls == 0:  # Normal terrain OR unchanging permanent river channel (tests change selectivity)
            subtype = (i // 8) % 3
            if subtype == 2:
                # Permanent narrow river stays dark in BOTH pre and post -> Normal (no new flood)
                river = np.abs(xx - 0.5 + 0.1 * tex) < 0.18
                dark_co = rng.uniform(-19.5, -15.5) if is_cband else rng.uniform(-23.5, -19.5)
                dark_cr = rng.uniform(-26.0, -22.5) if is_cband else rng.uniform(-30.0, -26.5)
                pre_co_c[river] = dark_co + 0.8 * tex[river]
                pre_cr_c[river] = dark_cr + 0.8 * tex[river]
                post_co_c[river] = dark_co + rng.uniform(-0.4, 0.4) + 0.8 * tex[river]
                post_cr_c[river] = dark_cr + rng.uniform(-0.4, 0.4) + 0.8 * tex[river]
            else:
                post_co_c = pre_co_c + rng.uniform(-0.8, 0.9)
                post_cr_c = pre_cr_c + rng.uniform(-0.7, 0.8)
        elif cls == 1:  # Flood inundation (open flood, partial channel surge, or flooded veg double-bounce)
            subtype = (i // 8) % 3
            if subtype == 0:
                frac = xx + 0.15 * tex < rng.uniform(0.45, 0.95)
                if is_cband:
                    flood_co = rng.uniform(-18.5, -14.2) + 1.1 * tex
                    flood_cr = rng.uniform(-25.0, -20.5) + 1.0 * tex
                else:
                    flood_co = rng.uniform(-25.5, -20.5) + 1.1 * tex
                    flood_cr = rng.uniform(-32.0, -27.5) + 1.0 * tex
                post_co_c = np.where(frac, flood_co, pre_co_c + rng.uniform(-0.5, 1.0))
                post_cr_c = np.where(frac, flood_cr, pre_cr_c + rng.uniform(-0.5, 0.8))
            elif subtype == 1:
                # Flooded vegetation / wind-roughened basin: strong +co-pol surge with flat/negative cross-pol
                if is_cband:
                    if (i // 24) % 2 == 1:
                        # Partially inundated low-backscatter river basin warming via capillary waves / emergent stems
                        pre_co_c = rng.uniform(-17.5, -10.5) + 1.1 * tex
                        pre_cr_c = rng.uniform(-20.5, -16.0) + 0.9 * tex
                    post_co_c = pre_co_c + rng.uniform(2.2, 5.2) + 0.8 * tex
                    post_cr_c = pre_cr_c + rng.uniform(-1.6, 0.3) + 0.4 * tex
                else:
                    post_co_c = rng.uniform(-5.5, -1.5) + 1.3 * tex
                    post_cr_c = rng.uniform(-24.5, -20.5) + 1.0 * tex
            else:
                split = xx + 0.15 * tex > 0.48
                if is_cband:
                    post_co_c = np.where(split, pre_co_c + rng.uniform(2.5, 4.8), rng.uniform(-18.2, -14.2)) + 1.0 * tex
                    post_cr_c = np.where(split, pre_cr_c - 0.6, rng.uniform(-24.8, -20.2)) + 1.0 * tex
                else:
                    post_co_c = np.where(split, rng.uniform(-4.8, -2.0), rng.uniform(-24.5, -21.0)) + 1.1 * tex
                    post_cr_c = np.where(split, rng.uniform(-23.5, -21.0), rng.uniform(-31.0, -27.8)) + 1.0 * tex
        elif cls == 2:  # Wildfire burn scar (canopy volume cross-pol collapse > co-pol change)
            burn_mask = (xx + 0.2 * tex) > rng.uniform(-0.1, 0.50)
            if is_cband:
                b_co = pre_co_c - rng.uniform(0.4, 1.5) + 0.8 * tex
                b_cr = pre_cr_c - rng.uniform(2.0, 4.6) + 0.9 * tex
                post_co_c = np.where(burn_mask, b_co, pre_co_c - 0.4)
                post_cr_c = np.where(burn_mask, b_cr, pre_cr_c - 0.6)
            else:
                b_co = rng.uniform(-13.8, -9.8) + 2.0 * tex
                b_cr = rng.uniform(-27.0, -22.5) + 1.2 * tex
                post_co_c = np.where(burn_mask, b_co, pre_co_c - 0.8)
                post_cr_c = np.where(burn_mask, b_cr, pre_cr_c - 1.5)
        else:  # Crop / soil stress (monsoon growth, harvest dry-down, or mixed smallholder parcel mosaic)
            row_sig = 1.0 * np.sin(2.0 * np.pi * rng.uniform(4.0, 8.0) * xx + rng.uniform(0, 3.14))
            if is_cband:
                subtype = (i // 8) % 3
                if subtype == 0:  # Monsoon vegetative growth / moist soil (BOTH VV and VH rise strongly together)
                    d_cr_pos = rng.uniform(0.9, 3.4)
                    d_co_pos = d_cr_pos + rng.uniform(-0.5, 1.8)
                    post_co_c = pre_co_c + d_co_pos + row_sig * 0.6 + 0.6 * tex
                    post_cr_c = pre_cr_c + d_cr_pos + 0.6 * tex
                elif subtype == 1:  # Harvest / dry-down (co-pol drops >= cross-pol drop, non-burn)
                    post_co_c = pre_co_c - rng.uniform(1.8, 3.6) + row_sig * 0.7 + 0.6 * tex
                    post_cr_c = pre_cr_c - rng.uniform(0.3, 1.3) + 0.6 * tex
                else:  # Smallholder parcel mosaic (harvest + irrigation side-by-side; near-zero mean, high |d| variance)
                    px = np.floor(xx * rng.uniform(3.0, 5.0))
                    py = np.floor(yy * rng.uniform(3.0, 5.0))
                    checker = ((px + py) % 2) == 0
                    mag = rng.uniform(2.2, 3.8)
                    post_co_c = pre_co_c + np.where(checker, mag, -mag * 0.85) + row_sig * 0.5 + 0.5 * tex
                    post_cr_c = pre_cr_c + np.where(checker, mag * 0.8, -mag * 0.6) + 0.5 * tex
            else:
                post_co_c = rng.uniform(-17.0, -14.0) + row_sig + 0.8 * tex
                post_cr_c = rng.uniform(-22.5, -19.0) + 0.8 * tex

        pre_co_db = _apply_gamma_speckle(pre_co_c, patch_looks, rng)
        pre_cr_db = _apply_gamma_speckle(pre_cr_c, patch_looks, rng)
        post_co_db = _apply_gamma_speckle(post_co_c, patch_looks, rng)
        post_cr_db = _apply_gamma_speckle(post_cr_c, patch_looks, rng)

        pre_patches[i, 0] = db_to_norm(pre_co_db, co_range[0], co_range[1])
        pre_patches[i, 1] = db_to_norm(pre_cr_db, cr_range[0], cr_range[1])
        post_patches[i, 0] = db_to_norm(post_co_db, co_range[0], co_range[1])
        post_patches[i, 1] = db_to_norm(post_cr_db, cr_range[0], cr_range[1])

    return pre_patches, post_patches, labels


def generate_training_patch_batch(
    num_samples: int = 200,
    look_count: int = 4,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backwards-compatible single-tensor batch generator (returns post_patches, labels)."""
    _, post_patches, labels = generate_bitemporal_training_batch(
        num_samples=num_samples, look_count=look_count, seed=seed, include_cband=False
    )
    return post_patches, labels


if __name__ == "__main__":
    scene = generate_realistic_sar_scene("nagpur_flood", look_count=4)
    print(
        f"Generated {scene['scenario'].scenario_id}: "
        f"post_patches={scene['post_patches'].shape}, labels_dist={np.bincount(scene['tile_labels'], minlength=4)}"
    )
