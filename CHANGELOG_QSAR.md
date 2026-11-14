# Changelog — Q-SAR Sentinel 2.0
NASA Space Apps Challenge 2026: Dancing with the SARs

## [2.1.0] - 2026-11-14 — Scientific Hardening, Traceable Provenance & Decision Support

### Added
- **Exact Liang-Barsky Segment Clipping for Road Hazards (`impact_analyzer.py`):**
  - Implemented `_clip_segment_to_bounds()` and `_polyline_clipped_length_km()` in `impact_analyzer.py`.
  - Roads intersecting hazard tiles are now strictly measured by the geodesic length of segments *inside* the hazard bounding box, eliminating false whole-road distance inflation.
- **Sensor-Aware Adapters & Provenance System (`sar_processor.py`):**
  - Added dedicated adapters: `NISARGCOVAdapter` (L-band HH/HV), `Sentinel1RTCAdapter` (C-band VV/VH), and `SyntheticSARAdapter` (controlled physics).
  - Explicit 4-mode provenance reporting: `live_nisar`, `live_sentinel1`, `simulation`, and `cached_demo`.
- **Exact 4-Candidate Classical Enumeration Benchmark (`qaoa_router.py`):**
  - Added `classical_enumeration` to `routing_comparison`, documenting that classical evaluation of the 4 candidate corridors requires `<0.1 ms` and serves as an exact ground-truth baseline.
- **Scientific Integrity Test Suite (`backend/tests/test_scientific_integrity.py`):**
  - Added 6 automated tests asserting polyline segment clipping, adapter isolation, heuristic reference nomenclature, and routing comparison completeness.

### Changed
- **Separation of Heuristic Agreement vs. True Accuracy:**
  - Renamed Track C real-scene evaluation metrics from `real_s1_quantum_accuracy` to `real_s1_quantum_heuristic_agreement` across `evaluate_validation.py`, `benchmark_metrics.json`, `main.py`, and `sar.ts`.
  - Updated `QuantumTelemetryPanel.tsx` headings to display "Scene Heuristic Reference Agreement" and "Real S1 Heuristic Agreement (108)" instead of misleading claims of ground-truth accuracy.
  - Added `metric_label` and `is_heuristic_agreement` flags in API responses.
- **Frontend Top HUD Provenance Badge (`MissionControlDashboard.tsx`):**
  - Replaced ambiguous pill with a persistent 4-tier indicator (`ACTUAL SENTINEL-1 RTC`, `ACTUAL NASA NISAR`, `SIMULATED L-BAND PHYSICS`, or `VERIFIED DEMO SNAPSHOT`) showing acquisition dates.
- **Refreshed Demo Cache (`demo_cache.json`):**
  - Regenerated all 6 cached scenarios (`nagpur_flood`, `california_wildfire`, `vidarbha_crop_stress` across `real_s1_rtc` and `nisar_sim`) with geometrically clipped road kilometers and dual heuristic/accuracy fields.

### Fixed
- **Unrelated Scenario Fallback Bug (`MissionControlDashboard.tsx`):**
  - Removed arbitrary `DEMO_CACHE[Object.keys(DEMO_CACHE)[0]]` fallback in `getCachedFallback()`. Missing scenario keys now return `null` and trigger an explicit unavailable state rather than displaying an unrelated disaster.
- **Scenario Switch State Drift (`MissionControlDashboard.tsx`):**
  - Guaranteed that when a scenario switch request fails, `analysis` does not retain the previous scenario's map and metrics under the new scenario's title.
