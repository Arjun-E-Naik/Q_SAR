# Q-SAR Sentinel 2.0
**Bi-Temporal SAR Disaster Assessment and Emergency Corridor Routing Platform**  
*Built for the NASA International Space Apps Challenge 2026: "Dancing with the SARs"*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-q--sar.vercel.app-0ea5e9?style=for-the-badge&logo=vercel)](https://q-sar.vercel.app)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-1e293b?style=for-the-badge&logo=github)](https://github.com/Arjun-E-Naik/Q_SAR.git)
[![NASA Space Apps](https://img.shields.io/badge/NASA-Space%20Apps%202026-1d4ed8?style=for-the-badge)](https://www.spaceappschallenge.org/)

---

### Quick Links

* **Live Web Application (Vercel):** [https://q-sar.vercel.app](https://q-sar.vercel.app)
* **GitHub Source Code:** [https://github.com/Arjun-E-Naik/Q_SAR.git](https://github.com/Arjun-E-Naik/Q_SAR.git)
* **Technical Architecture Deep Dive:** [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)
* **Complete Scientific & Technical Audit:** [QSAR_COMPLETE_TECHNICAL_AUDIT.md](QSAR_COMPLETE_TECHNICAL_AUDIT.md)
* **Release Changelog:** [CHANGELOG_QSAR.md](CHANGELOG_QSAR.md)

---

## Project Overview

Optical Earth-observation imagery is frequently obstructed by dense cloud cover during monsoon floods and heavy smoke during wildfires. Active Synthetic Aperture Radar (SAR) sensors such as Sentinel-1 (C-band, VV/VH) and NISAR (L-band, HH/HV) penetrate clouds and darkness, but disaster response teams still face major practical barriers when turning radar backscatter into field decisions:

1. Separating new disaster anomalies (inundation, flooded vegetation double-bounce, canopy burn scars, crop/soil stress) from permanent rivers, seasonal moisture, and multiplicative radar speckle noise.
2. Linking hazard detections to local roads, bridges, hospitals, shelters, and populated settlements.
3. Selecting evacuation or relief corridors that balance travel distance against hazard exposure and model uncertainty without relying on blind shortest-path routing.

Q-SAR Sentinel 2.0 addresses these gaps in a unified decision-support workflow combining bi-temporal SAR physics, a compact hybrid quantum-classical neural classifier, OpenStreetMap infrastructure analysis, human-in-the-loop verification, and constraint-preserving XY-QAOA route selection. For complete system internals, formulas, and module architecture, see [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).

## What the Application Provides

- **Dual-Sensor Bi-Temporal SAR Analysis:** Strictly separates real Radiometric Terrain Corrected Sentinel-1 C-band (`VV`, `VH`) imagery from NISAR L-band (`HH`, `HV`) physics simulations (`nisar_sim`), with live NASA Earthdata CMR metadata lookup (`OPERA_L2_RTC-S1_V1`, `NISAR_L2_GCOV_PROVISIONAL_V1`, `NISAR_UR_L2`) and raster alignment/radiometric quality checks (`data_quality`: `PASS` / `WARN` / `UNSUITABLE`).
- **Any-Location Global AOI Search (`GET /api/geocode`, `POST /api/locations/prepare`):** Search any city, river basin, or `lat, lon` coordinate pair directly from the dashboard. The backend geocodes the area (Nominatim or direct coordinate parsing), extracts cropped COG overview windows (`<= 512x512`) for matching pre/post Sentinel-1 passes from Planetary Computer STAC with bounded HTTP timeouts (`15s`), fetches local OpenStreetMap roads, bridges, settlements, and critical facilities from Overpass (with lightweight arterial retry and terrain fallback), persists dynamic locations to `custom_scenarios.json`, and runs the full analysis pipeline.
- **Hybrid Quantum-Classical Classifier (`HybridQSARNet`):** A 9,732-parameter model (45.0 KB checkpoint) combining a dual-stream `SARConvStem` (spatial `3x3` filters + pre-activation `AvgPool2d(4)` radiometric branch) with a 4-qubit PennyLane variational layer (`AngleEmbedding` + 2 `StronglyEntanglingLayers`, 8 observables). Low-confidence or high-entropy regions are flagged as `Unknown / Requires Human Verification` via `abstention_flag` instead of forcing uncertain predictions.
- **Geospatial Exposure & Verification Workflow:** Computes cosine-latitude geodesic hazard areas (km2 and hectares), tracks exposed OSM settlements, critical facilities (hospitals, clinics, fire stations, police, shelters, schools), and bridge crossings, ranks tiles via a weighted verification priority queue, separates automated `radiometric_reference_label` from `verified_ground_truth_label`, persists operator tile triage and route approvals (`/api/verifications`), and exports enriched `.geojson` or printable Situation Reports (SITREP).
- **Constraint-Preserving XY-QAOA Route Selection:** Generates four diverse candidate corridors across contracted OSM road networks (or a `6x6` tactical grid fallback when OSM connectivity is unavailable), constructs a 4-qubit Ising Hamiltonian controlled by Risk Aversion (`alpha` in `[0.2, 2.0]`), prepares an exact Hamming-weight-1 `|W_4>` state, and evolves a constraint-preserving XY ring mixer (`valid_onehot_prob_mass >= 0.99` vs. unconstrained X-mixer leakage diagnostic) benchmarked against Classical Direct, Dijkstra, and A*.

## Tools and Technology Stack

| Layer | Technologies Used |
|---|---|
| Backend API | Python 3.11+, FastAPI, Uvicorn, Pydantic |
| Quantum and ML | PyTorch, PennyLane, PennyLane-Lightning (`lightning.qubit`) |
| Geospatial and SAR | Rasterio, NumPy, SciPy, h5py, NetworkX, earthaccess, Requests |
| Satellite & GIS APIs | Microsoft Planetary Computer STAC (`sentinel-1-rtc`), NASA Earthdata CMR, OpenStreetMap Overpass API, Nominatim |
| Frontend UI | Next.js 15, React 19, TypeScript, Tailwind CSS, MapLibre GL, Lucide Icons |

## Repository Contents

```text
.
├── .gitignore
├── Agents.md.txt                         # Architectural constraints and specifications
├── TECHNICAL_ARCHITECTURE.md             # Detailed technical implementation, algorithms, API & field guide
├── README.md
├── start_local.ps1                       # Cross-environment Windows PowerShell launcher for backend + frontend
├── colab_training/
│   └── Train_QSAR_Hybrid_Cloud.ipynb     # Cloud training notebook exporting weights & benchmarks
├── backend/
│   ├── .env.example                      # Optional NASA Earthdata credentials template
│   ├── requirements.txt                  # Python dependencies (including requests)
│   ├── main.py                           # FastAPI endpoints (/api/health, /scenarios, /analyze, /geocode, /locations/prepare, /verifications, /geojson)
│   ├── sar_processor.py                  # Sensor provenance, GeoTIFF reprojection, multi-look reference, data_quality gate & CMR queries
│   ├── quantum_model.py                  # HybridQSARNet, ClassicalSARNet, telemetry, calibration & abstention logic
│   ├── impact_analyzer.py                # Geodesic exposure, OSM intersections, facilities, priority queue & briefs
│   ├── qaoa_router.py                    # Contracted OSM / grid corridor generation + 4-qubit XY-QAOA solver
│   ├── evaluate_validation.py            # 3-track validation & calibration evaluator
│   ├── data/
│   │   ├── generate_demo_patches.py      # Pre/post L-band simulator and dynamic scenario registry
│   │   ├── fetch_real_sar_geotiffs.py    # Planetary Computer STAC Sentinel-1 RTC COG window fetcher
│   │   ├── fetch_osm_infrastructure.py   # Overpass OSM road, bridge, facility & settlement fetcher with fallback
│   │   ├── osm_cache/                    # Pre-cached OSM JSONs for preset scenarios
│   │   └── real_sar/                     # Pre-cached EPSG:4326 Sentinel-1 RTC GeoTIFFs for preset scenarios
│   ├── models/
│   │   ├── qsar_hybrid_weights.pt        # Pre-trained bi-temporal model checkpoint (45.0 KB)
│   │   └── benchmark_metrics.json        # Evaluated 3-track metrics and confusion matrices
│   └── tests/
│       └── test_phase1.py                # Automated backend, XY-QAOA, location prep & verification tests
└── frontend/
    ├── package.json
    ├── next.config.ts
    ├── tsconfig.json
    └── src/
        ├── app/                          # Next.js App Router pages, print SITREP styling & layout
        ├── components/
        │   ├── MissionControlDashboard.tsx
        │   ├── QuantumTelemetryPanel.tsx
        │   └── SarViewport.tsx
        ├── data/
        │   └── demo_cache.json           # Offline fallback snapshots for all 3 preset scenarios
        └── types/
            └── sar.ts                    # Shared TypeScript API contracts
```

## Empirical Benchmarks (From `benchmark_metrics.json`)

All metrics below are computed by `backend/evaluate_validation.py` with zero real-scene fine-tuning:

- **Track A - Synthetic NISAR-Sim Holdout (`N=108`):**
  - `L=4` standard looks: Hybrid Q-SAR `86.11%` (Macro-F1 `0.8056`, ECE `0.0743`), Matched Classical Head `94.44%` (Macro-F1 `0.9199`, ECE `0.0574`), Standalone Classical CNN `92.59%` (Macro-F1 `0.9119`, ECE `0.0386`).
  - `L=1` speckle stress: Hybrid Q-SAR `80.56%` (Macro-F1 `0.7691`, ECE `0.0766`), Matched Classical Head `77.78%` (Macro-F1 `0.7759`, ECE `0.1171`), Standalone Classical CNN `87.96%` (Macro-F1 `0.8807`, ECE `0.0678`).
- **Track B - Cross-Region Domain-Shift Holdout (`+/-1.5 dB` uniform bias):**
  - `L=4`: Hybrid Q-SAR `89.44%` (`0.8945` F1, `0.0684` ECE) vs. Matched Classical Head `91.39%` (`0.9131` F1) vs. Standalone Classical CNN `93.06%` (`0.9295` F1).
  - `L=1`: Hybrid Q-SAR `86.94%` (`0.8678` F1, `0.0486` ECE) vs. Matched Classical Head `82.50%` (`0.8203` F1, `0.0787` ECE) vs. Standalone Classical CNN `88.61%` (`0.8870` F1, `0.0483` ECE).
- **Track C - Zero-Shot Real Sentinel-1 C-Band RTC (`N=108` tiles):**
  - Overall: Hybrid Q-SAR `68.52%` (Macro-F1 `0.6645`, ECE `0.1669`), Standalone Classical CNN `62.04%` (Macro-F1 `0.6378`, ECE `0.3142`), Matched Classical Head `49.07%` (Macro-F1 `0.4749`, ECE `0.2671`).
  - Note: Track C labels are generated from an automated OPERA-style dual-polarization radiometric baseline (`is_verified_manual_ground_truth: false`) rather than manually digitized polygons, so Track C measures agreement with radiometric thresholds alongside human-in-the-loop review in the UI.

## How to Run Locally

### Prerequisites
- Python 3.11+
- Node.js 20+ and npm

### 1. Backend Setup
```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

pip install -r backend/requirements.txt
```

Optional: copy `backend/.env.example` to `backend/.env` and add `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` if querying authenticated NASA Earthdata CMR downloads. The core preset scenarios and Sentinel-1 RTC planetary computer search run without credentials.

### 2. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

### 3. Start Both Servers

**Option A - Windows PowerShell helper:**
```powershell
.\start_local.ps1
```

**Option B - Manual two-terminal launch:**
```bash
# Terminal 1 (from repository root)
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend
npm run dev
```

Open `http://127.0.0.1:3000` for the Mission Control dashboard and `http://127.0.0.1:8000/docs` for the OpenAPI documentation.

## Running Tests and Checks

```bash
# Backend unit tests (SAR scenes, HybridQSARNet, XY-QAOA, location preparation, verifications)
python -m unittest backend.tests.test_phase1 -v

# Re-run 3-track benchmark evaluation
python -m backend.evaluate_validation

# Frontend typecheck and production build
cd frontend
npx tsc --noEmit
npm run build
```

## Data Attribution and License

- **Satellite Imagery:** Pre-cached and on-demand Sentinel-1 C-band IW GRDH (`VV`, `VH`) RTC crops via Microsoft Planetary Computer STAC (`sentinel-1-rtc`), paired with NASA Earthdata CMR granule discovery (`earthaccess`) and an internal L-band `nisar_sim` ablation simulator.
- **Infrastructure & Basemaps:** OpenStreetMap roads, bridges, facilities, and settlements via Overpass API; Nominatim geocoding; Esri World Dark Gray Base and Esri World Imagery rendered through MapLibre GL.
- **License:** Open-source MIT License.
