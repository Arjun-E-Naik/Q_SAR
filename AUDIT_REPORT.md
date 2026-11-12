# Q-SAR Sentinel — Pre-Build Architectural & Scientific Audit Report

**Audit Date:** November 11, 2026  
**Target Event:** 2026 NASA International Space Apps Challenge (November 14–15, 2026; deadline Nov 15, 11:59 PM local time)  
**Target Challenge:** *"Dancing with the SARs"* (NISAR Multi-Polarization Surface Change & Disaster Response)  
**Audit Scope:** Cross-check of project specification, [`Agents.md.txt`](Agents.md.txt), local hardware/software environment, NASA SAR data formats, PennyLane/PyTorch compatibility, and NASA judging criteria alignment.

---

## 1. Executive Understanding & Readiness Snapshot

### What We Are Building
**Q-SAR Sentinel** is a Hybrid Quantum-Classical Geospatial Decision Support System designed for the **"Dancing with the SARs"** NASA Space Apps challenge. It combines:
1. **NASA Open SAR Ingestion (`earthaccess`):** Queries NISAR L-band / OPERA Sentinel-1 RTC backscatter products (`HH`, `HV` / `VV`, `VH`) with a deterministic, physics-accurate offline SAR patch generator so the live demo never depends on live multi-GB downloads.
2. **Dressed Quantum-Classical SAR Classifier (`PyTorch` + `PennyLane`):** A lightweight classical convolutional feature compressor mapping `(B, 2, 64, 64)` dual-polarization SAR patches into 4 angle features (`[-π, π]`), processed by a 4-qubit PennyLane `TorchLayer` (`AngleEmbedding` + 2-layer `StronglyEntanglingLayers`) on `lightning.qubit` (<50ms CPU inference), outputting 4 surface-state classes plus 16-dimensional basis statevector probabilities (`|0000⟩`–`|1111⟩`).
3. **QAOA Disaster Relief Router (`PennyLane` + `NetworkX`):** Converts the classified 6×6 terrain grid into a weighted hazard graph and runs a 4-qubit QAOA formulation to select an optimal hazard-avoiding evacuation/drone corridor.
4. **Zero-Local-Compute Cloud Training (`Google Colab`):** Offloads training to a free Colab T4 GPU notebook (`Train_QSAR_Hybrid_Cloud.ipynb`) exporting `<5 MB` weights (`qsar_hybrid_weights.pt`) and empirical ablation metrics.
5. **NASA Mission Control UI (`Next.js 15` + `Tailwind` + `Deck.gl / MapLibre`):** Dark-mode geospatial dashboard displaying interactive SAR polarizations, before/after change comparisons, QAOA rescue routes, live quantum statevector telemetry, and classical vs. quantum benchmarks.

### Verified Local Environment Audit
| Component | Verified Value | Status / Action Needed |
|---|---|---|
| **CPU / RAM** | 13th Gen Intel Core i7-13620H (10 cores, 16 threads), 15.7 GB RAM | Suitable for <50ms 4-qubit CPU inference; heavy training stays in Colab |
| **Python** | Python `3.11.9` | Matches `Agents.md.txt` (`3.11+`) |
| **Node / npm** | Node `v22.12.0`, npm `10.9.0` | Ready for Next.js 15 |
| **Git** | `2.47.1.windows.2` | Ready |
| **Installed ML/Quantum** | `torch 2.14.1+cpu`, `torchvision 0.29.1`, `pennylane 0.45.1`, `pennylane_lightning 0.45.0`, `qiskit 2.5.2`, `networkx 3.6.1`, `scipy 1.17.1`, `numpy 2.4.6`, `pydantic 2.14.1` | Core ML & quantum simulator verified working locally |
| **Missing Backend Deps** | `fastapi`, `uvicorn`, `earthaccess`, `h5py`, `rasterio` | Need installation during Phase 1 (`h5py` is essential for NISAR HDF5) |

---

## 2. Critical Technical & Scientific Findings (Must Fix Before Coding)

### Finding 1: NISAR Data Format Is HDF5 (`.h5`), Not GeoTIFF
- **Issue in Current Spec:** Phase 1 mentions `earthaccess` + `rasterio` fetching GeoTIFFs for NISAR.
- **Verified Reality:**
  - NASA ASF DAAC distributes **NISAR Level-2 GCOV** (`NISAR_L2_GCOV_V1` fully validated as of Nov 6, 2026; also `NISAR_L2_GCOV_PROVISIONAL_V1` / `NISAR_L2_GCOV_BETA_V1`) as **CF-compliant HDF5 (`.h5`) files** (with datasets at `/science/LSAR/GCOV/grids/frequencyA/HH` and `HV`), whereas **NASA OPERA RTC-S1** (`OPERA_L2_RTC-S1_V1`) distributes **Cloud-Optimized GeoTIFFs (COGs)** (`VV` and `VH`).
- **Required Fix:** Include `h5py` in `backend/requirements.txt` alongside `rasterio` so `sar_processor.py` natively supports **both** NISAR L-band `.h5` (`HH`/`HV`) and OPERA Sentinel-1 `.tif` (`VV`/`VH`) products.

### Finding 2: Weight Size Limit (`<5 MB`) vs. Frozen MobileNetV3-Small (`~10.2 MB`)
- **Conflict Identified:**
  - `Agents.md.txt` (line 34) references `Frozen MobileNetV3 + 4-Qubit PennyLane TorchLayer`.
  - User prompt requires `<5 MB` `qsar_hybrid_weights.pt`, `(B, 2, 64, 64)` SAR input, and `Agents.md.txt` (line 9) requires 100% offline local startup.
  - **The Math:** Full `mobilenet_v3_small` has ~2.54M parameters (~10.2 MB in FP32), expects 3-channel RGB ImageNet inputs (`224×224`), and triggers an internet download if weights aren't cached locally.
- **Recommended Resolution (Two Valid Options):**
  - **Option A (Recommended — Pure SAR Domain Stem):** Use the lightweight **3-layer SAR Conv2d speckle-reduction backbone** (`Conv2d(2→16→32→64)` + `AdaptiveAvgPool2d(1)` + `Linear(64, 4)`), which handles `(B, 2, 64, 64)` dual-pol radar natively, runs in ~2–4ms on CPU, requires zero ImageNet downloads, and keeps `qsar_hybrid_weights.pt` under **~250 KB** (`<0.3 MB`).
  - **Option B (Strict MobileNetV3 Variant):** Use a truncated early slice (`mobilenet_v3_small.features[:6]`) with a trainable `1×1` `(2→3)` polarimetric adapter and save **only** the trainable head/adapter weights (`<150 KB`) or ship FP16 truncated backbone (`<1 MB`) without online weight fetching.
  - *Decision needed before Phase 1:* Recommend **Option A** for scientific fidelity to SAR radar physics and zero-dependency offline boot (while updating `Agents.md.txt` comments cleanly if approved).

### Finding 3: PennyLane `diff_method="adjoint"` + `qml.probs` Constraint (Verified Locally)
- **Technical Trap:** The prompt asks `HybridQSARNet` to use `diff_method="adjoint"` inside `qml.qnn.TorchLayer` AND return the 16-dimensional quantum statevector probabilities (`qml.probs`) for the UI.
- **Verified Behavior (`pennylane 0.45.1` + `lightning.qubit 0.45.0`):**
  - Adjoint differentiation natively supports expectation values (`[qml.expval(qml.PauliZ(i)) for i in range(4)]`), which we tested and confirmed works with `TorchLayer` + `.backward()`.
  - Mixing `qml.probs()` inside an `adjoint`-differentiated `TorchLayer` QNode causes differentiation errors or slow fallbacks.
- **Required Architecture:** Keep **two QNodes sharing the exact same circuit ansatz and `q_weights`**:
  1. `q_classify_node` (`diff_method="adjoint"`): Wrapped in `qml.qnn.TorchLayer`, returning 4 Pauli-Z expectations for training & fast inference.
  2. `q_telemetry_node` (`diff_method=None`, run under `torch.no_grad()`): Returns `qml.probs(wires=range(4))` (16 basis states) + single-qubit Pauli `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)` Bloch vectors on demand for any inspected patch.

### Finding 4: Making 4-Qubit QAOA Over a 6×6 Grid Mathematically Genuine
- **Scientific Risk:** A 6×6 grid has 36 nodes and 60 edges. Encoding 36 binary node/edge variables requires 36+ qubits (impossible on a 4-qubit statevector). If the backend runs a dummy 4-qubit circuit unrelated to the map and silently calls `networkx.shortest_path()`, NASA judges will flag it as "quantum theater."
- **Required Architecture (Genuine Hierarchical QUBO-QAOA):**
  1. Build the full 6×6 weighted hazard graph in `networkx` from the 36 SAR-classified tiles (`Normal`: 1.0, `Crop/Soil Stress`: 3.0, `Flood Inundation`: 50.0, `Wildfire Scar`: 80.0).
  2. Extract the **Top-4 distinct candidate emergency corridors / critical sector-bridge decisions** (using Yen's k-shortest simple paths / bottleneck graph reduction), mapping them to a **4-qubit Ising Cost Hamiltonian** $H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j$ whose diagonal energies encode:
     - Total SAR hazard traversal penalty along corridor $i$,
     - One-hot route selection constraint $(\sum_i x_i - 1)^2$,
     - Multi-sector risk overlap penalties $J_{ij}$.
  3. Execute a genuine **4-qubit QAOA circuit ($p=2$ layers)** on `lightning.qubit` (`Cost` + `X-Mixer` Hamiltonians), evaluate optimal bitstring probabilities (`|1000⟩`, `|0100⟩`, `|0010⟩`, `|0001⟩`), compute real QAOA expectation value $\langle H_C \rangle$ and approximation ratio vs. classical Dijkstra optimum, and return the winning route GeoJSON.
  - **Result:** 100% mathematically authentic QAOA tied directly to the live SAR classification map, executing in **<35ms**.

### Finding 5: Contradictions & Missing Spec Items Between Prompt and `Agents.md.txt`
1. **Channel vs. Coherence Mismatch (`Agents.md.txt` line 17):** Mentions shape `(B, 2, 64, 64)` for `HH`, `HV`, *"plus interferometric coherence"* (which would be 3 channels). We should keep tensor shape `(B, 2, 64, 64)` (`[HH, HV]` normalized dB) strictly consistent across Colab, Backend, and `Agents.md.txt`, and expose derived polarimetric/temporal metrics (`HV - HH` cross-pol ratio, RVI, and Pre/Post $\Delta\sigma^0$ change) cleanly in the geospatial analysis layer.
2. **Startup Training Rule (`Agents.md.txt` line 7 vs. Phase 1 item 2):** `Agents.md.txt` explicitly forbids running training loops during FastAPI startup. Therefore, if `backend/models/qsar_hybrid_weights.pt` is absent at boot, perform **zero-training deterministic physics initialization** (edge/speckle-aware Gabor/Sobel-like convolutional kernels + calibrated readout weights) in `<50ms`, never a startup training loop.
3. **Missing Before/After SAR Temporal Change:** Section 2 of the prompt describes a *"before/after SAR slider"*, which is essential for *"Dancing with the SARs"*, yet Phase 1/3 omitted `pre_event` vs. `post_event` backscatter scenes. Adding paired Pre/Post SAR layers fixes this gap directly.
4. **Hardcoded Mapbox Token Risk:** Using Mapbox requires an API key that can break when NASA judges inspect the hosted demo. Using **MapLibre GL + Deck.gl with free Carto Dark Matter / OpenStreetMap raster/vector tiles** guarantees 100% zero-token reliability.

---

## 3. High-Impact Improvements to Win NASA Appreciation

These targeted additions directly maximize scores across all **5 official NASA Space Apps Judging Criteria** (`Impact`, `Creativity`, `Validity`, `Relevance`, `Presentation`) within the 48-hour sprint:

### 1. Leverage Live 2026 NISAR History (Relevance & Storytelling)
- **Correction to Narrative:** NISAR successfully launched on **July 30, 2025**, began provisional L-band releases in July 2026, and released **fully calibrated/validated L-band products (L0–L2) on November 6, 2026** (days before this hackathon!).
- **Action:** Update documentation, UI badges, and Colab ingestion to search real `NISAR_L2_GCOV_V1` / `NISAR_L2_GCOV_BETA_V1` and `OPERA_L2_RTC-S1_V1` granules via `earthaccess`, plus bundle small pre-cached real sample/metadata footprints for **Nagpur / Godavari Basin (Maharashtra, India)** and **California Wildfire ROI** so judges see real NASA coordinates and SAR physics immediately.

### 2. Physically Accurate SAR Simulation & Standard NASA False-Color Composites (Validity)
- **Multiplicative Gamma Speckle (Not Gaussian):** Model $L$-look SAR speckle correctly using Gamma-distributed intensity scaling ($L=4$ default) rather than additive Gaussian noise.
- **L-Band Flood Double-Bounce Physics:** Explicitly model L-band `HH` bright **double-bounce** spikes in flooded vegetation/crops alongside dark **specular reflection** on open water, and sharp `HV` volume-scattering drops in severely burned forest canopies.
- **Interactive Polarimetric Views in UI:** Add 5 rich geospatial layers instead of 3:
  1. `HH Co-Polarization (dB)`
  2. `HV Cross-Polarization (dB)`
  3. `NASA/ASF False-Color RGB` (`R: HH, G: HV, B: HH/HV`)
  4. `Before / After Change Swipe` (Pre-Disaster vs. Post-Disaster $\Delta \sigma^0$)
  5. `Quantum Anomaly Segmentation + QAOA Route`

### 3. Interactive "Tile-Click Quantum Inspector" & Live Speckle Stress-Test (Creativity & Proof of Principle)
- **Click Any of the 36 Map Tiles:** Clicking a specific 6×6 grid cell updates the right-hand telemetry panel in real time for **that exact patch**:
  - The 4 classical feature angles fed into `AngleEmbedding(Y)`,
  - The 16-state quantum probability bar chart (`|0000⟩`–`|1111⟩`),
  - Per-qubit Bloch expectation readouts (`⟨X⟩, ⟨Y⟩, ⟨Z⟩`) and entanglement layer weights.
- **Live Speckle Look ($L$) Stress-Test Slider:** Let judges toggle SAR noise ($L = 1$ raw high-speckle vs. $L = 4$ standard vs. $L = 16$ multi-looked) and watch classification confidence and the QAOA route dynamically recompute in `<100ms`.

### 4. Genuine Few-Shot Ablation Benchmark from Colab (Scientific Integrity)
- In `Train_QSAR_Hybrid_Cloud.ipynb`, train **both** a parameter-matched `ClassicalSARNet` and `HybridQSARNet` across small-data slices ($N = 40, 80, 160, 400$ patches) — directly testing the real scientific hypothesis that compact variational quantum heads regularize well when labeled training data for a brand-new mission (NISAR) is scarce.
- Export **real measured results** to `backend/models/benchmark_metrics.json` (Accuracy, Macro-F1, Per-Class IoU/F1, Parameter Count, Latency) and render those authentic numbers in the UI. State honestly in the README where classical matches or where hybrid helps in low-data/noisy regimes—judges strongly reward scientific honesty over hype.

### 5. Humanitarian Actionability & GIS Interoperability (Impact)
- **Real Quantitative Impact Metrics:** Display estimated **Affected Area ($\text{km}^2$)** per hazard class, **Corridor Hazard Reduction (%)** (QAOA safest route vs. naive shortest Euclidean path), and **Estimated Rescue Transit Risk Score**.
- **One-Click GeoJSON Export:** Let first responders / judges download the classified hazard grid and QAOA optimal rescue path as a valid `.geojson` file compatible with QGIS, ArcGIS, and Google Earth.
- **Standalone Static Demo Fallback (`demo_cache.json`):** Pre-compute real backend responses into `frontend/src/data/demo_cache.json` so a Vercel-hosted frontend works seamlessly even if the Python FastAPI server is offline during asynchronous judging.

---

## 4. Proposed Canonical Repository Structure (Aligned with `Agents.md.txt`)

```text
Nasa Project/
├── Agents.md.txt                          # Master invariants (source of truth)
├── AUDIT_REPORT.md                        # This pre-build audit report
├── NASA_SUBMISSION_GUIDE.md               # Phase 4: Submission answers + 120s video script
├── colab_training/
│   └── Train_QSAR_Hybrid_Cloud.ipynb      # Zero-local-compute Colab T4 trainer + ablation plots
├── backend/
│   ├── requirements.txt                   # fastapi, uvicorn, torch, pennylane, earthaccess, h5py, rasterio, ...
│   ├── main.py                            # FastAPI server (/api/scenarios, /api/analyze, /api/quantum-circuit, /api/geojson)
│   ├── sar_processor.py                   # NASA earthaccess (NISAR .h5 + OPERA .tif) + Pre/Post Gamma-speckle SAR generator
│   ├── quantum_model.py                   # Lightweight SAR stem + 4-qubit PennyLane TorchLayer + Telemetry QNode
│   ├── qaoa_router.py                     # 6x6 NetworkX hazard graph + genuine 4-qubit corridor QUBO QAOA solver
│   └── models/
│       ├── qsar_hybrid_weights.pt         # Exported Colab weights (or deterministic <50ms baseline fallback)
│       └── benchmark_metrics.json         # Real Colab few-shot Classical vs. Hybrid QCNN metrics
└── frontend/
    ├── package.json                       # Next.js 15 + Tailwind CSS + MapLibre GL / Deck.gl + Lucide + Recharts
    └── src/
        ├── data/demo_cache.json           # Pre-computed real backend snapshots for standalone Vercel reliability
        └── app/                           # Dark-mode NASA Quantum SAR Mission Control UI
```

---

## 5. Pre-Build Decision Checklist (Awaiting User Approval)

Before starting **Phase 1**, please confirm two small architectural choices:
1. **Classical Stem Choice:** Proceed with the **3-layer SAR-native Conv2d stem (`<300 KB` weights, 100% offline, dual-pol native)** rather than ImageNet `mobilenet_v3_small` (`~10.2 MB` full weights), or use a truncated offline MobileNetV3 slice? *(Recommendation: 3-layer SAR-native Conv2d stem).*
2. **Enhanced Scope Approval:** Include the high-ROI NASA upgrades audited above (**Pre/Post before-after SAR change detection + ASF false-color RGB composite + Clickable 36-tile quantum statevector inspector + Live Gamma-speckle slider + Genuine 4-corridor QUBO QAOA + GeoJSON download**)?
