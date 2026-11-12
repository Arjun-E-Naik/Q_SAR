"""Fetch and cache real 384x384 EPSG:4326 Sentinel-1 RTC GeoTIFF crops via Planetary Computer STAC.

Downloads zero 5GB full granules; instead performs HTTP range-read COG mosaicking &
reprojection directly onto each scenario's exact 384x384 north-up geographic grid
(same relative orbit for pre- vs post-event pairs) and stores compressed float32 GeoTIFFs
under `backend/data/real_sar/{scenario_id}/` for offline/low-latency execution.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds as transform_from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds

REAL_SAR_DIR = Path(__file__).resolve().parent / "real_sar"
STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"

REAL_SCENE_SPECS: Dict[str, Dict[str, Any]] = {
    "nagpur_flood": {
        "bbox": (79.48, 20.80, 79.72, 21.04),
        "pre_window": "2026-05-15T00:00:00Z/2026-05-30T23:59:59Z",
        "post_window": "2026-08-18T00:00:00Z/2026-08-26T23:59:59Z",
        "orbit": 165,
        "event_note": "Wainganga-Gosikhurd 2026 Monsoon Flood (Orbit 165 descending)",
    },
    "california_wildfire": {
        "bbox": (-121.84, 39.88, -121.60, 40.12),
        "pre_window": "2024-07-01T00:00:00Z/2024-07-12T23:59:59Z",
        "post_window": "2024-08-18T00:00:00Z/2024-08-28T23:59:59Z",
        "orbit": 137,
        "event_note": "Butte-Tehama Park Fire Canopy Burn Scar (Orbit 137)",
    },
    "vidarbha_crop_stress": {
        "bbox": (78.40, 20.68, 78.64, 20.92),
        "pre_window": "2026-06-20T00:00:00Z/2026-07-05T23:59:59Z",
        "post_window": "2026-10-15T00:00:00Z/2026-10-25T23:59:59Z",
        "orbit": 165,
        "event_note": "Wardha-Yavatmal 2026 Kharif Agrarian Cycle (Orbit 165)",
    },
}


def _get_sas_token() -> str:
    req = urllib.request.Request(SAS_TOKEN_URL, headers={"User-Agent": "Q-SAR-Sentinel/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return str(json.loads(resp.read().decode("utf-8"))["token"])


def _stac_search(
    bbox: Tuple[float, float, float, float], dt_range: str, orbit: Optional[int] = None, limit: int = 40
) -> List[Dict[str, Any]]:
    payload = {
        "collections": ["sentinel-1-rtc"],
        "bbox": [float(v) for v in bbox],
        "datetime": dt_range,
        "limit": limit,
    }
    req = urllib.request.Request(
        STAC_SEARCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "Q-SAR-Sentinel/2.0"},
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        feats = json.loads(resp.read().decode("utf-8")).get("features", [])
    if orbit is not None:
        matched = [f for f in feats if f.get("properties", {}).get("sat:relative_orbit") == orbit]
        return matched if matched else feats
    return feats


def _reproject_mosaic_to_epsg4326(
    items: List[Dict[str, Any]],
    asset_key: str,
    bbox: Tuple[float, float, float, float],
    sas_token: str,
    size: int = 384,
) -> Tuple[np.ndarray, List[str]]:
    dst_transform = transform_from_bounds(*bbox, width=size, height=size)
    mosaic = np.full((size, size), np.nan, dtype=np.float32)
    used_ids: List[str] = []

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_HTTP_TIMEOUT="15",
        GDAL_HTTP_CONNECTTIMEOUT="10",
        GDAL_HTTP_MAX_RETRY="1",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        VSI_CACHE="TRUE",
        CPL_VSIL_CURL_CHUNK_SIZE="262144",
    ):
        for item in items[:4]:
            href = item["assets"][asset_key]["href"]
            signed_url = f"{href}?{sas_token}"
            buf = np.full((size, size), np.nan, dtype=np.float32)
            with rasterio.open(signed_url) as src:
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326", src.crs, *bbox, densify_pts=8
                )
                if (
                    right <= src.bounds.left
                    or left >= src.bounds.right
                    or top <= src.bounds.bottom
                    or bottom >= src.bounds.top
                ):
                    continue
                c_left = max(left, src.bounds.left)
                c_bottom = max(bottom, src.bounds.bottom)
                c_right = min(right, src.bounds.right)
                c_top = min(top, src.bounds.top)
                raw_win = window_from_bounds(
                    c_left, c_bottom, c_right, c_top, transform=src.transform
                ).round_offsets().round_lengths()
                col_off = max(0, min(src.width - 1, int(raw_win.col_off) - 2))
                row_off = max(0, min(src.height - 1, int(raw_win.row_off) - 2))
                win_w = min(src.width - col_off, int(raw_win.width) + 4)
                win_h = min(src.height - row_off, int(raw_win.height) + 4)
                if win_w < 4 or win_h < 4:
                    continue
                win = Window(col_off, row_off, win_w, win_h)
                out_w = max(4, min(win_w, 512))
                out_h = max(4, min(win_h, 512))
                arr = src.read(
                    1,
                    window=win,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
                win_tf = src.window_transform(win) * Affine.scale(
                    win_w / float(out_w), win_h / float(out_h)
                )
                reproject(
                    source=arr,
                    src_transform=win_tf,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    destination=buf,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
            valid = np.isfinite(buf) & (buf > 0.0)
            if np.any(valid):
                used_ids.append(str(item["id"]))
                mosaic = np.where(np.isfinite(mosaic), mosaic, np.where(valid, buf, np.nan))
            if float(np.mean(np.isfinite(mosaic))) > 0.99:
                break

    # Fill any tiny single-pixel border gaps cleanly with local median
    if not np.all(np.isfinite(mosaic)):
        fill_val = float(np.nanmedian(mosaic)) if np.any(np.isfinite(mosaic)) else 0.05
        mosaic = np.nan_to_num(mosaic, nan=fill_val)
    return np.maximum(mosaic, 1e-6).astype(np.float32), used_ids


def _write_geotiff(path: Path, arr: np.ndarray, bbox: Tuple[float, float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    transform = transform_from_bounds(*bbox, width=w, height=h)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
        nodata=-9999.0,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


def _parse_iso_date(raw: Optional[str], fallback: date) -> date:
    if not raw:
        return fallback
    try:
        return date.fromisoformat(str(raw).strip()[:10])
    except ValueError:
        return fallback


def _group_by_orbit_and_date(
    items: List[Dict[str, Any]],
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for it in items:
        assets = it.get("assets") or {}
        if "vv" not in assets or "vh" not in assets:
            continue
        props = it.get("properties") or {}
        orbit = int(props.get("sat:relative_orbit") or -1)
        dt_str = str(props.get("datetime", ""))[:10]
        if orbit < 0 or len(dt_str) != 10:
            continue
        grouped.setdefault(orbit, {}).setdefault(dt_str, []).append(it)
    return grouped


def fetch_custom_sar_scene(
    scenario_id: str,
    bbox: Tuple[float, float, float, float],
    pre_date: Optional[str] = None,
    post_date: Optional[str] = None,
    location_name: str = "",
) -> Dict[str, Any]:
    """Fetch & cache real 384x384 pre/post Sentinel-1 RTC GeoTIFFs for any global bbox."""
    out_dir = REAL_SAR_DIR / scenario_id
    meta_path = out_dir / "metadata.json"
    required = ("pre_vv.tif", "pre_vh.tif", "post_vv.tif", "post_vh.tif")
    if meta_path.exists() and all((out_dir / f).exists() for f in required):
        return json.loads(meta_path.read_text(encoding="utf-8"))

    today = datetime.now(timezone.utc).date()
    post_target = min(_parse_iso_date(post_date, today), today)
    pre_target = _parse_iso_date(pre_date, post_target - timedelta(days=60))
    if pre_target >= post_target:
        pre_target = post_target - timedelta(days=45)

    post_start = (post_target - timedelta(days=36)).isoformat() + "T00:00:00Z"
    post_end = min(today, post_target + timedelta(days=12)).isoformat() + "T23:59:59Z"
    pre_start = (pre_target - timedelta(days=30)).isoformat() + "T00:00:00Z"
    pre_end = min(post_target - timedelta(days=3), pre_target + timedelta(days=25)).isoformat() + "T23:59:59Z"

    post_items = _stac_search(bbox, f"{post_start}/{post_end}", orbit=None, limit=48)
    pre_items = _stac_search(bbox, f"{pre_start}/{pre_end}", orbit=None, limit=48)

    post_grouped = _group_by_orbit_and_date(post_items)
    pre_grouped = _group_by_orbit_and_date(pre_items)
    common_orbits = sorted(set(post_grouped.keys()) & set(pre_grouped.keys()))

    # Automatically widen pre window up to 150 days if no same-orbit match in initial window
    if not common_orbits and post_grouped:
        wide_pre_start = (pre_target - timedelta(days=140)).isoformat() + "T00:00:00Z"
        pre_items = _stac_search(bbox, f"{wide_pre_start}/{pre_end}", orbit=None, limit=64)
        pre_grouped = _group_by_orbit_and_date(pre_items)
        common_orbits = sorted(set(post_grouped.keys()) & set(pre_grouped.keys()))

    if not post_grouped or not pre_grouped:
        raise ValueError(
            "No Sentinel-1 RTC acquisitions found covering this location and date window. "
            "Try shifting dates or selecting an inland/coastal area."
        )

    sas = _get_sas_token()

    def _pick_best_pass(
        pass_map: Dict[str, List[Dict[str, Any]]], target: date
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        return sorted(
            pass_map.items(),
            key=lambda kv: (abs((date.fromisoformat(kv[0]) - target).days), kv[0]),
        )

    chosen: Optional[Dict[str, Any]] = None
    orbit_pairs: List[Tuple[int, int, bool]] = [(orb, orb, True) for orb in common_orbits]
    if not orbit_pairs:
        for po in sorted(post_grouped.keys()):
            for pro in sorted(pre_grouped.keys()):
                orbit_pairs.append((po, pro, False))

    for p_orb, r_orb, same_orb in orbit_pairs:
        post_candidates = _pick_best_pass(post_grouped[p_orb], post_target)[:3]
        pre_candidates = _pick_best_pass(pre_grouped[r_orb], pre_target)[:3]
        for post_d, post_pass in post_candidates:
            post_vv, post_vv_ids = _reproject_mosaic_to_epsg4326(post_pass, "vv", bbox, sas)
            post_vh, _ = _reproject_mosaic_to_epsg4326(post_pass, "vh", bbox, sas)
            if not post_vv_ids:
                continue
            for pre_d, pre_pass in pre_candidates:
                pre_vv, pre_vv_ids = _reproject_mosaic_to_epsg4326(pre_pass, "vv", bbox, sas)
                pre_vh, _ = _reproject_mosaic_to_epsg4326(pre_pass, "vh", bbox, sas)
                if not pre_vv_ids:
                    continue
                chosen = {
                    "orbit": p_orb,
                    "orbit_matched": same_orb,
                    "pre_date": pre_d,
                    "post_date": post_d,
                    "pre_vv": pre_vv,
                    "pre_vh": pre_vh,
                    "post_vv": post_vv,
                    "post_vh": post_vh,
                    "pre_ids": pre_vv_ids,
                    "post_ids": post_vv_ids,
                }
                break
            if chosen is not None:
                break
        if chosen is not None:
            break

    if chosen is None:
        raise ValueError("Found Sentinel-1 STAC items, but raster pixels did not cover the ROI bbox cleanly.")

    _write_geotiff(out_dir / "pre_vv.tif", chosen["pre_vv"], bbox)
    _write_geotiff(out_dir / "pre_vh.tif", chosen["pre_vh"], bbox)
    _write_geotiff(out_dir / "post_vv.tif", chosen["post_vv"], bbox)
    _write_geotiff(out_dir / "post_vh.tif", chosen["post_vh"], bbox)

    meta = {
        "scenario_id": scenario_id,
        "stac_catalog": "Microsoft Planetary Computer STAC (sentinel-1-rtc)",
        "sensor": "Sentinel-1 IW GRDH Radiometric Terrain Corrected (C-Band 5.405 GHz, VV + VH)",
        "polarization_channels": ["VV", "VH"],
        "relative_orbit": int(chosen["orbit"]),
        "orbit_matched": bool(chosen["orbit_matched"]),
        "crs": "EPSG:4326",
        "shape": [384, 384],
        "bbox": [round(float(v), 5) for v in bbox],
        "pre_date": chosen["pre_date"],
        "post_date": chosen["post_date"],
        "pre_item_ids": chosen["pre_ids"],
        "post_item_ids": chosen["post_ids"],
        "event_note": f"On-demand Sentinel-1 RTC bi-temporal scan ({location_name or scenario_id}, Orbit {chosen['orbit']})",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def fetch_all_scenarios() -> None:
    sas = _get_sas_token()
    for sid, spec in REAL_SCENE_SPECS.items():
        bbox = tuple(spec["bbox"])
        orbit = int(spec["orbit"])
        pre_items = _stac_search(bbox, spec["pre_window"], orbit)
        post_items = _stac_search(bbox, spec["post_window"], orbit)

        # Group by date to ensure multi-frame swath mosaics belong to the same pass date
        def latest_pass(items: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
            by_date: Dict[str, List[Dict[str, Any]]] = {}
            for it in items:
                d = it["properties"]["datetime"][:10]
                by_date.setdefault(d, []).append(it)
            chosen_date = sorted(by_date.keys())[-1]
            return chosen_date, by_date[chosen_date]

        pre_date, pre_pass = latest_pass(pre_items)
        post_date, post_pass = latest_pass(post_items)

        out_dir = REAL_SAR_DIR / sid
        pre_vv, pre_vv_ids = _reproject_mosaic_to_epsg4326(pre_pass, "vv", bbox, sas)
        pre_vh, _ = _reproject_mosaic_to_epsg4326(pre_pass, "vh", bbox, sas)
        post_vv, post_vv_ids = _reproject_mosaic_to_epsg4326(post_pass, "vv", bbox, sas)
        post_vh, _ = _reproject_mosaic_to_epsg4326(post_pass, "vh", bbox, sas)

        _write_geotiff(out_dir / "pre_vv.tif", pre_vv, bbox)
        _write_geotiff(out_dir / "pre_vh.tif", pre_vh, bbox)
        _write_geotiff(out_dir / "post_vv.tif", post_vv, bbox)
        _write_geotiff(out_dir / "post_vh.tif", post_vh, bbox)

        meta = {
            "scenario_id": sid,
            "stac_catalog": "Microsoft Planetary Computer STAC (sentinel-1-rtc)",
            "sensor": "Sentinel-1 IW GRDH Radiometric Terrain Corrected (C-Band 5.405 GHz, VV + VH)",
            "polarization_channels": ["VV", "VH"],
            "relative_orbit": orbit,
            "crs": "EPSG:4326",
            "shape": [384, 384],
            "bbox": list(bbox),
            "pre_date": pre_date,
            "post_date": post_date,
            "pre_item_ids": pre_vv_ids,
            "post_item_ids": post_vv_ids,
            "event_note": spec["event_note"],
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        vv_db = 10.0 * np.log10(post_vv)
        vh_db = 10.0 * np.log10(post_vh)
        dvh = vh_db - 10.0 * np.log10(pre_vh)
        print(
            f"[RealSAR:{sid}] pre={pre_date} post={post_date} "
            f"VV_med={np.median(vv_db):.2f}dB VH_med={np.median(vh_db):.2f}dB "
            f"water%={(vv_db < -14.0).mean()*100:.1f}% burn_dVH<-3%={(dvh < -3.0).mean()*100:.1f}%"
        )


if __name__ == "__main__":
    fetch_all_scenarios()
