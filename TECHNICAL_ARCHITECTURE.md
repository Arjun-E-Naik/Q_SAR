# Q-SAR Sentinel 2.0 - Technical Architecture and Implementation Reference

This document details the system architecture, scientific pipeline, quantum-classical algorithms, geospatial data workflows, API contracts, and practical field usage implemented in Q-SAR Sentinel 2.0.

---

## 1. Practical Operational Problem and System Design

Emergency coordinators analyzing floodplains, wildfire scars, and agricultural stress during active disasters must resolve four bottlenecks before deploying field teams:

1. **Optical Obscuration:** Monsoon cloud decks and dense wildfire smoke block optical satellites during peak response windows. Active C-band (Sentinel-1) and L-band (NISAR) Synthetic Aperture Radar (SAR) penetrate cloud cover and darkness, but raw single-date radar intensity cannot distinguish permanent rivers or urban corner reflectors from new inundation.
2. **Speckle and False Positives:** Single-look SAR imagery exhibits multiplicative Gamma speckle noise, and non-disaster factors (crop cycles, surface roughness, soil moisture) can mimic hazard backscatter changes. Uncertain predictions must be flagged for operator triage rather than presented as confirmed detections.
3. **Infrastructure Blindness:** Pixel-level hazard classification alone does not tell coordinators which roads, bridge crossings, hospitals, shelters, or settlements intersect detected surface changes.
4. **Constrained Corridor Selection:** Shortest-path road routing ignores hazard exposure, flooded bridge closures, and model uncertainty. Dispatchers need candidate corridors parameterized by configurable risk aversion, paired with human approval gates before dispatch.

Q-SAR Sentinel 2.0 solves these in an end-to-end pipeline:

```text
User AOI (Preset Scenario OR Search City / "lat, lon")
  |
  +---> [SAR Ingestion & Alignment]  Planetary Computer STAC (Sentinel-1 RTC VV/VH)
  |     (sar_processor.py,           NASA Earthdata CMR metadata, NISAR HDF5 / nisar_sim
  |      fetch_real_sar_geotiffs.py) -> 384x384 EPSG:4326 pre/post rasters + data_quality gate
  |
  +---> [OSM Infrastructure Cache]   Overpass API (primary + lightweight arterial fallback)
        (fetch_osm_infrastructure.py) -> roads, bridges, settlements, critical facilities
  |
  v
[Bi-Temporal Hybrid Classifier]      4ch input [Post_Co, Post_Cross, tanh(2.4 * Delta)]
  (quantum_model.py)                 Dual-stream SARConvStem -> 4-qubit PennyLane QNode + MLP head
  |                                  -> class probs, Bloch entropy H_Q, abstention_flag
  v
[Geospatial Impact Engine]           Cosine-latitude km2/ha exposure, OSM bridge/facility counts
  (impact_analyzer.py)               Weighted priority_queue + plain-language explainable_brief
  |
  v
[XY-QAOA Corridor Selector]          Contracted OSM graph / 6x6 fallback -> 4 distinct corridors
  (qaoa_router.py)                   |W_4> init + p=2 XY ring mixer Ising H(alpha in [0.2, 2.0])
  |
  v
[Mission Control Dashboard]          MapLibre GL + Esri + OSM overlay, Hybrid/Classical/Diff views,
  (frontend/src/components/)         Human triage (/api/verifications), GeoJSON export, PDF SITREP
```

---

## 2. Satellite Data Ingestion and Geospatial Quality Gate

Modules: `backend/sar_processor.py`, `backend/data/fetch_real_sar_geotiffs.py`, `backend/data/generate_demo_patches.py`

### 2.1 Sensor Provenance Separation
The pipeline enforces strict provenance separation between sensor modes so C-band observations and L-band simulations are never mislabeled:
- **`real_s1_rtc`:** Radiometric Terrain Corrected Sentinel-1 C-band IW GRDH (`VV`, `VH`, 5.405 GHz) cached as four aligned `384x384` `EPSG:4326` float32 GeoTIFFs (`pre_vv.tif`, `pre_vh.tif`, `post_vv.tif`, `post_vh.tif`) plus `metadata.json`.
- **`nisar_sim`:** Physics-calibrated dual-polarization L-band (`HH`, `HV`, 1.257 GHz) bi-temporal simulation (`generate_demo_patches.py`) for controlled ablation testing across flooding, canopy burn scars, and agricultural stress.
- **Native Readers (`sar_processor.py`):** Includes `load_nisar_h5_patches` for NISAR Level-2 GCOV HDF5 datasets (`/science/LSAR/GCOV/grids/frequencyA/HH` and `HV`), `load_opera_geotiff_patches` with CRS reprojection (`_reproject_georeferenced_band`), and live NASA Earthdata CMR discovery via `query_nasa_earthdata_granules` (`OPERA_L2_RTC-S1_V1`, `NISAR_L2_GCOV_PROVISIONAL_V1`, `NISAR_UR_L2`).

### 2.2 Global On-Demand Location Provisioning
Endpoints: `GET /api/geocode` and `POST /api/locations/prepare` (`backend/main.py`)

When an operator searches any city, region, or coordinate string (`20.92, 79.60`, `29.7604N, 95.3698W`):
1. `_parse_lat_lon_query` resolves coordinate strings immediately offline; place names query OpenStreetMap Nominatim (`_nominatim_geocode`).
2. `_build_location_bbox` computes a bounded `EPSG:4326` AOI (`span_deg` default `0.24`, cosine-latitude adjusted for longitude width).
3. `fetch_custom_sar_scene` searches Planetary Computer STAC (`sentinel-1-rtc`), automatically widens pre-event search windows up to 140 days if needed to match relative orbits (`sat:relative_orbit`), and streams only cropped COG overview windows (`out_shape <= 512x512`) inside the intersecting `src.bounds` with bounded GDAL `/vsicurl/` HTTP timeouts (`15s`).
4. Concurrently (`ThreadPoolExecutor`), `ensure_osm_cache_for_bbox` queries Overpass API for roads, bridges, settlements, waterways, and critical facilities (`hospital`, `clinic`, `doctors`, `fire_station`, `police`, `shelter`, `community_centre`, `school`, `college`), falling back first to a lightweight arterial query (`motorway|trunk|primary|secondary` + core facilities) and then cleanly to an `available: False` terrain-grid marker if Overpass is unreachable.
5. `register_custom_scenario` persists dynamic scenario metadata to `backend/data/custom_scenarios.json` and clears LRU scene/OSM/topology caches so the new region loads immediately.

### 2.3 Radiometric Alignment Gate and Label Provenance
Every real scene passes `_load_cached_real_geotiff_scene` quality verification (`data_quality`):
- `crs_aligned`: verifies all 4 bands share `EPSG:4326`.
- `shape_aligned`: verifies `384x384` raster dimensions (partitioned into a `6x6` grid of `64x64` tiles).
- `transform_aligned`: verifies identical affine geotransforms across pre/post co/cross-pol rasters.
- `valid_pixel_pct` and `radiometric_in_bounds`: computes minimum valid pixel ratio and dB bounds (`[-55, 22]` dB) to surface `PASS`, `WARN`, or `UNSUITABLE` with explicit diagnostics (`unsuitable_reason`).
- **Dual Label Discipline:** Automated thresholds from native 4x4 multi-looked backscatter changes are stored strictly as `radiometric_reference_label` (`is_verified_manual_ground_truth: false`). Synthetic `L=1` speckle stress injection (`_apply_gamma_speckle`) is applied after computing the native-look reference so stress testing evaluates noise robustness without corrupting reference labels.

---

## 3. Bi-Temporal Hybrid Quantum-Classical Classifier (`backend/quantum_model.py`)

### 3.1 Input Fusion and Dual-Stream Stem
`build_bitemporal_input` constructs a 4-channel tensor `(B, 4, 64, 64)`:
- Channels 0-1: Normalized post-event backscatter `[Post_Co, Post_Cross]`.
- Channels 2-3: Smooth bounded bi-temporal change `0.5 + 0.5 * tanh(2.4 * (Post - Pre))`, preserving subtle ~2 dB C-band crop/flood shifts without saturating large -18 dB L-band drops.

`SARConvStem` extracts 48 features via two parallel branches before projecting with `Tanh() * pi` to 4 bounded qubit rotation angles in `[-pi, pi]`:
1. **Spatial Stream (`conv_blocks`):** `Conv2d(4->16, k=3)` -> `GELU` -> `AvgPool2d(2)` -> `Conv2d(16->24, k=3)` -> `GELU` -> `AvgPool2d(2)` -> `Conv2d(24->24, k=3)` -> `GELU` -> `AdaptiveAvgPool2d(1)`.
2. **Radiometric Multi-Look Stream (`multilook_blocks`):** Pre-activation `AvgPool2d(4)` (suppressing `L=1` speckle variance linearly before nonlinearity) -> `Conv2d(4->24, k=1)` -> `GELU` -> `AdaptiveAvgPool2d(1)`.

### 3.2 Variational Quantum Layer and Matched Classical Baselines
- **Quantum Circuit (`_quantum_circuit_ansatz`):** 4 qubits (`lightning.qubit` with `default.qubit` fallback), `AngleEmbedding(rotation="Y")` followed by `N_LAYERS=2` `StronglyEntanglingLayers` (weight shape `(2, 4, 3)` = 24 trainable quantum parameters), executed with `diff_method="adjoint"` inside `qml.qnn.TorchLayer`.
- **Observables (`N_Q_OBS = 8`):** Four single-qubit expectations `<Z_i>` plus four ring two-qubit correlators `<Z_i Z_{(i+1)%4}>`, mapped through a linear head (`Linear(8, 4)`) to the four surface classes: `0: Normal Terrain`, `1: Flood Inundation`, `2: Wildfire Burn Scar`, `3: Crop/Soil Stress`.
- **Compact Footprint:** `9,732` total trainable parameters (`45.0 KB` weights in `backend/models/qsar_hybrid_weights.pt`).
- **Matched Comparisons:** Every forward pass runs `classical_head` (`4 -> 8 -> 4` MLP operating on the same 4D bottleneck) and benchmarks against an independent `ClassicalSARNet` so Hybrid vs. Classical predictions and per-tile disagreements (`disagreement_flag`) are directly observable.

### 3.3 Telemetry and Epistemic Abstention
`infer_with_telemetry` computes the exact 16-dimensional basis statevector probabilities, `(B, 4, 3)` single-qubit reduced-density-matrix Bloch vectors, and normalized Shannon entropy `H_Q` in `[0, 1]`. `compute_uncertainty_and_abstention` combines softmax margin, probability entropy, and `H_Q` to flag ambiguous or boundary tiles (`abstention_flag = true`, surfaced as `Unknown / Verify`) for manual review.

---

## 4. Humanitarian Impact and Verification Workflow (`backend/impact_analyzer.py`)

- **Geodesic Area Calculation:** Computes tile-by-tile physical footprints (`geodesic_bounds_area_km2`) using exact spherical latitude cosine scaling rather than flat pixel counts, outputting `flooded_area_km2`, `burn_scar_area_km2`, and `agricultural_area_ha`.
- **Spatial Infrastructure Intersection:** Matches each `6x6` tile bounding box (`_cached_osm_spatial_index`) against cached OSM road polylines (`roads_at_risk_km`), bridges (`bridges_at_risk_count`), settlements (`estimated_exposed_population`), and critical facilities (`facilities_total`, `facilities_exposed_count`).
- **Weighted Field Priority Queue:** Scores tiles into tiers (`CRITICAL`, `HIGH`, `MEDIUM`, `MONITOR`) using deterministic additive weights across hazard severity (`Flood=42`, `Burn=38`, `Crop/Soil=18`), exposed critical facilities (`up to 30`), bridges (`up to 24`), road segments (`up to 8`), population (`up to 18`), and `abstention_flag` uncertainty (`+12`).
- **Explainable Operational Briefs:** Generates structured scene and tile interpretations (`headline`, `what_changed`, `supporting_channels`, `confidence_summary`, `next_verification_action`) reporting actual pre-to-post `dB` and RVI shifts and flagged local assets.
- **Persisted Human-in-the-Loop Verification (`GET`/`POST /api/verifications`):** Operators triage individual tiles (`confirmed_hazard`, `false_alarm`, `needs_inspection`) and set corridor approval state (`pending_human_approval`, `approved_for_recon`, `rejected`), synchronized between browser `localStorage` and `backend/data/verification_state.json` and embedded into exported `.geojson` properties and printable Situation Reports.

---

## 5. Constraint-Preserving 4-Qubit XY-QAOA Router (`backend/qaoa_router.py`)

Routing operates in two explicitly separated stages (`candidate_gen_time_ms` and `quantum_opt_time_ms`):

1. **Network Construction & Candidate Generation:**
   - `_build_osm_road_graph` constructs a connected road topology snapped to the `6x6` SAR hazard grid, contracting degree-2 same-tile road bends into simplified graph edges (`_cached_osm_topology`) while storing full vertex coordinates (`coords_fwd`) so rendered corridors follow true curved road geometry. When OSM coverage is unavailable or sparse, `build_terrain_graph` provides a seamless `6x6` diagonal-guarded terrain graph fallback.
   - Multi-objective edge weights at Risk Aversion `alpha in [0.2, 2.0]` balance geodesic distance, SAR hazard class (`Stress=2.2`, `Flood=22.0`, `Fire=30.0`), quantum uncertainty penalty (`H_Q` amplified on abstained tiles), and bridge closure penalties across flooded tiles.
   - `_extract_strategy_corridors` computes four spatially diverse candidate routes (`Rapid Direct`, `Balanced Tactical`, `High-Ground Convoy`, `Max-Safety Perimeter`) using edge-overlap diversity penalties alongside **Classical Direct** (hazard-blind), **Classical Dijkstra**, and **Classical A\*** baselines.

2. **Constraint-Preserving XY-QAOA Selector:**
   - Maps the four candidate corridors onto the 4 Hamming-weight-1 computational basis states (`|1000>`, `|0100>`, `|0010>`, `|0001>`) via an Ising Hamiltonian `H_C(alpha) = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + const`.
   - Prepares an exact `|W_4>` equal superposition state (`PauliX` + calibrated `CRY`/`CNOT` ladder) and applies `p=2` layers of cost phase separation (`RZ`, `IsingZZ`) plus a **constraint-preserving XY ring mixer** (`IsingXX` + `IsingYY` across ring pairs `(0,1), (2,3), (1,2), (3,0)`), optimized with COBYLA (`maxiter=28`).
   - Because the XY mixer preserves total Hamming weight, probability mass remains strictly within the 4 valid single-route states (`valid_onehot_prob_mass >= 0.99`, `invalid_bitstring_prob <= 0.01`), which is compared on every call against an unconstrained Hadamard + `RX` mixer diagnostic (`qaoa_x_mixer_diag_qnode`, reporting `x_mixer_invalid_leakage`).

---

## 6. Frontend Interface and Offline Accessibility (`frontend/src/`)

- **`SarViewport.tsx`:** Single persistent MapLibre GL canvas with Esri Dark Gray Base and Esri World Imagery basemaps, projected SAR layers (`Segmentation`, `Change RGB`, `Co-Pol`, `Cross-Pol`, `ASF RGB`, `Before / After` temporal swipe), cached OSM roads, bridges, critical facilities, and collision-filtered settlement labels, plus interactive tile inspection and custom Origin/Target route selection.
- **`QuantumTelemetryPanel.tsx`:** Three domain-oriented inspection tabs (`Tile Inspector`, `Corridors`, `Impact & Benchmarks`) presenting Classical vs. Hybrid probabilities, operator tile triage controls, XY-mixer constraint preservation vs. X-mixer leakage telemetry, 4-strategy selector cards, route approval sign-off, data quality diagnostics, and 3-track validation tables.
- **`MissionControlDashboard.tsx`:** Sticky Mission Control header (preset scenarios + dynamic **Search Area** popover, `Real S1-RTC` vs. `NISAR Sim`, `L=4` vs. `L=1` speckle stress, Re-Scan, `.geojson` export, printable PDF SITREP), `Map View` switcher (`Hybrid Q-SAR` | `Classical CNN` | `Disagreement Diff`), continuous `alpha` Risk Aversion slider with `Drone` / `Rescue` / `Medical` presets, KPI strip, weighted Priority Action Queue, and automatic fallback to `frontend/src/data/demo_cache.json` when the backend port is unreachable.

---

## 7. Backend API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | `GET` | Runtime health check, active PennyLane device, weight source, model size, and boot latency |
| `/api/scenarios` | `GET` | Returns preset scenarios and any prepared custom AOIs |
| `/api/geocode?q=...` | `GET` | Resolves place names (Nominatim) or `lat, lon` strings into coordinate candidates |
| `/api/locations/prepare` | `POST` | Fetches STAC Sentinel-1 pre/post GeoTIFFs + OSM cache for a searched AOI and registers scenario |
| `/api/analyze` | `POST` | Runs bi-temporal SAR inference, geodesic impact analysis, and XY-QAOA corridor routing |
| `/api/quantum-circuit` | `GET` | Returns 4-qubit circuit specification and layer parameter counts |
| `/api/scenarios/{id}/geojson` | `GET` | Exports GIS-ready `FeatureCollection` (hazard tiles, routes, origin/target markers) |
| `/api/verifications` | `GET` / `POST` | Reads or updates persisted operator tile triage and corridor approval decisions |

---

## 8. Empirical Validation Summary (`backend/evaluate_validation.py`)

Evaluation metrics stored in `backend/models/benchmark_metrics.json` (zero fine-tuning on real Sentinel-1 scenes):

- **Track A - Synthetic NISAR-Sim Holdout (`N=108`):**
  - `L=4` standard looks: Hybrid Q-SAR `86.11%` (Macro-F1 `0.8056`, ECE `0.0743`), Matched Classical Head `94.44%` (Macro-F1 `0.9199`, ECE `0.0574`), Standalone Classical CNN `92.59%` (Macro-F1 `0.9119`, ECE `0.0386`).
  - `L=1` speckle stress: Hybrid Q-SAR `80.56%` (Macro-F1 `0.7691`, ECE `0.0766`) vs. Matched Classical Head `77.78%` (Macro-F1 `0.7759`, ECE `0.1171`), Standalone Classical CNN `87.96%`.
- **Track B - Cross-Region Domain-Shift Holdout (`+/-1.5 dB` uniform bias):**
  - `L=4`: Hybrid Q-SAR `89.44%` (`0.8945` F1, `0.0684` ECE) vs. Matched Classical Head `91.39%` vs. Standalone Classical CNN `93.06%`.
  - `L=1`: Hybrid Q-SAR `86.94%` (`0.8678` F1, `0.0486` ECE) outperforms Matched Classical Head `82.50%` (`0.8203` F1, `0.0787` ECE).
- **Track C - Zero-Shot Real Sentinel-1 C-Band RTC (`N=108` tiles):**
  - Overall: Hybrid Q-SAR `68.52%` (Macro-F1 `0.6645`, ECE `0.1669`) vs. Standalone Classical CNN `62.04%` (Macro-F1 `0.6378`, ECE `0.3142`) and Matched Classical Head `49.07%` (Macro-F1 `0.4749`, ECE `0.2671`).
  - **Methodological Disclosure:** Track C reference classes derive from an automated OPERA-style radiometric baseline (`is_verified_manual_ground_truth: false`) rather than independently digitized ground-truth polygons. Operational deployment uses the tile verification workflow to confirm alerts on the ground.
