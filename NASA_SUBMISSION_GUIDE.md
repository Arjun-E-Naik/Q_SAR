# Q-SAR Sentinel 2.0 — 2026 NASA Space Apps Submission Guide

This guide contains ready-to-use copy for the **spaceappschallenge.org** project submission form (closing **Sunday, November 15, 2026, at 11:59 PM local time**), a deployment checklist, and an updated **120-second demo video script** aligned with Q-SAR Sentinel 2.0.

---

## 1. Space Apps Project Submission Form Answers

### Project Name
**Q-SAR Sentinel 2.0: Bi-Temporal Hybrid Quantum-Classical SAR Intelligence & OSM Disaster Routing Engine**

### Selected Challenge
**Dancing with the SARs** *(2026 NASA International Space Apps Challenge)*

### High-Level Summary (~95 words)
> **Q-SAR Sentinel 2.0** fuses pre-event and post-event Sentinel-1 (`VV`, `VH`) and NISAR-simulated (`HH`, `HV`) SAR imagery to map floods, emergent flooded vegetation, wildfire burn scars, and agricultural stress in under 205 ms. A compact `9,732`-parameter (`45.0 KB`) dual-stream hybrid CNN + 4-qubit PennyLane classifier outputs calibrated hazard classes with **Unknown/Verify** abstention flags and plain-language radar explanations. Intersecting hazards against cached **OpenStreetMap** roads, bridges, and settlements produces geodesic impact metrics ($\text{km}^2$, hectares, exposed population) and dynamically routes response convoys via a multi-objective **4-qubit QAOA solver** ($\alpha \in [0.2, 2.0]$).

### Detailed Project Description (Mapped to NASA's 5 Judging Criteria)

#### 1. Relevance & Open NASA Data Integration
* Addresses **"Dancing with the SARs"** using **true multi-temporal radar change detection** (`Post` + smooth `tanh(2.4 · (Post − Pre))` co-pol & cross-pol deltas) across three high-impact operational scenes:
  1. **Wainganga–Nagpur Monsoon Flood (`nagpur_flood`):** Cached real **Sentinel-1 C-Band IW RTC** (`VV`, `VH`) GeoTIFFs separating permanent river channels from newly inundated floodplains and emerged flooded-vegetation **double-bounce** backscatter.
  2. **Sierra Foothills Wildfire Burn Scar (`california_wildfire`):** Cached real **Sentinel-1 C-Band IW RTC** (`VV`, `VH`) GeoTIFFs isolating severe cross-polarized volume scattering collapse (`ΔVH`) from water.
  3. **Vidarbha Agricultural & Soil Stress (`vidarbha_crop_stress`):** Cached real **Sentinel-1 C-Band IW RTC** GeoTIFFs tracking smallholder parcel mosaic moisture shifts across rural road corridors.
* Explicitly separates **C-Band `VV/VH` Sentinel-1** provenance from **L-Band `HH/HV` NISAR** ablation simulations (`nisar_sim`), queries live **NASA Earthdata CMR** metadata (`OPERA_L2_RTC-S1_V1`, `NISAR_L2_GCOV_PROVISIONAL_V1`, `NISAR_UR_L2`) via `earthaccess`, and provides an OPERA-style 4×4 multi-looked radiometric reference explicitly flagged as non-manual reference (`is_verified_manual_ground_truth: false`).

#### 2. Scientific Validity (Rigorous Hybrid Architecture & Honest 3-Track Validation)
* **Bi-Temporal Dual-Stream Hybrid Classifier:** Combines a spatial `3×3` convolutional branch and an `AvgPool2d(4)` radiometric multi-look branch (`SARConvStem`, 48 features → 4 angles) with a 4-qubit `AngleEmbedding(Y)` + 2-layer `StronglyEntanglingLayers` variational circuit measuring **8 quantum observables** (`4× ⟨Zᵢ⟩` + `4× ⟨ZᵢZᵢ₊₁⟩`). Low-confidence or high-entropy boundary tiles trigger an **Unknown / Abstention** flag (`abstention_flag`) rather than silent overconfident errors.
* **Three-Track Empirical Benchmark (`evaluate_validation.py` -> `benchmark_metrics.json`, zero real-scene training leakage):**
  * **Track A (Synthetic NISAR-Sim Holdout, `N=108`):** `L=4`: Hybrid Q-SAR `86.11%` (`F1: 0.8056`, `ECE: 0.0743`) vs. Standalone Classical CNN `92.59%`. Under `L=1` speckle stress, Hybrid Q-SAR `80.56%` outperforms Matched Classical Head `77.78%`.
  * **Track B (Cross-Region Domain-Shift Holdout):** `L=4` `89.44%`; `L=1` `86.94%` (lowest calibration ECE `0.0486` vs. `0.0787` matched classical head).
  * **Track C (Zero-Shot Real Sentinel-1 C-Band RTC, `N=108`):** **Hybrid Q-SAR achieves `68.52%` overall accuracy (`F1: 0.6645`, `ECE: 0.1669`)** vs. **Standalone Classical CNN `62.04%` (`F1: 0.6378`, `ECE: 0.3142`)** and **Matched Classical Head `49.07%`**, showing stronger cross-domain calibration on real orbital SAR without fine-tuning on real GeoTIFFs.
* **Contracted OSM Road Network + Dynamic QAOA Routing:** Evaluates 4 genuinely distinct candidate corridors across connected OpenStreetMap road topologies (contracted inside static same-tile bends while preserving full road polyline geometry, with automatic `6×6` tactical grid fallback when road coverage is sparse). Adjusting **Risk Aversion ($\alpha \in [0.2, 2.0]$)** transitions the Ising ground state across strategies (`Rapid Direct` → `Balanced Tactical` → `High-Ground Convoy` → `Max-Safety Perimeter`), benchmarked side-by-side against **Classical Direct**, **Classical Dijkstra**, and **Classical A\***.

#### 3. Impact & Humanitarian Utility
* Computes exact cosine-latitude geodesic **flooded area ($\text{km}^2$)**, **burn scar ($\text{km}^2$)**, **agricultural stress area (hectares)**, cached **OSM roads at risk (km)**, **bridges flagged**, **estimated exposed population**, critical logistics chokepoint tiles, plain-language scene/tile briefs, and one-click **`.geojson`** GIS export.

#### 4. Outcome-Focused Transparency & Decision UI
* Replaces intimidating circuit jargon with emergency-responder comparisons:
  * **Map Classification Switcher:** Toggle live between **`Hybrid Q-SAR`**, **`Classical CNN`**, and **`Disagreement Diff`**.
  * **Plain-Language Tile Inspector:** Explains *what changed radiometrically*, pre→post decibel shifts, OSM road/bridge intersections, and Classical vs. Quantum confidence on any clicked sector.
  * **Before / After Temporal Swipe** + **OSM Infrastructure Overlay (`ON/OFF`)** + **Speckle Stress Test (`L1`–`L16`)**.

#### 5. Accessibility
* Zero-install Next.js 15 + MapLibre GL frontend backed by an updated offline snapshot (`demo_cache.json`, ~4.6 MB) across all 3 scenarios and both `Real S1 RTC` / `NISAR Sim` sources.

### Tools, Libraries & Data Used
* **Satellite & Infrastructure Data:** Sentinel-1 C-band RTC (Microsoft Planetary Computer STAC), NASA Earthdata CMR (`earthaccess`, `OPERA_L2_RTC-S1_V1`, `NISAR_L2_GCOV_PROVISIONAL_V1`), OpenStreetMap Overpass API, Esri World Dark Gray / World Imagery.
* **AI & Quantum Stack:** PyTorch, PennyLane (`lightning.qubit`), SciPy, NetworkX, FastAPI, Uvicorn, h5py, Rasterio.
* **Frontend:** Next.js 15, TypeScript, Tailwind CSS, MapLibre GL, Lucide Icons.

---

## 2. 120-Second Demo Video Script

*Recording setup: Open `http://127.0.0.1:3000` full-screen (with FastAPI running on `:8000`) at 1080p.*

* **[0:00 – 0:20] Problem & Bi-Temporal Real SAR**
  * *Screen:* Load **Nagpur Flood** on **`Real S1 RTC`**, click **`Before / After`**, and drag the temporal swipe across the Wainganga river basin.
  * *Voiceover:* "When monsoon clouds or wildfire smoke blind optical satellites, active radar sees right through. Meet **Q-SAR Sentinel 2.0**, built for *Dancing with the SARs*. Using real bi-temporal Sentinel-1 C-band imagery over Nagpur, the Before/After swipe reveals permanent river channels versus newly flooded floodplains and emergent flooded vegetation."

* **[0:20 – 0:48] Hybrid Q-SAR vs. Classical CNN & Plain-Language Tile Inspector**
  * *Screen:* Toggle the map switcher between **`Hybrid Q-SAR`**, **`Classical CNN`**, and **`Disagreement Diff`**. Click a highlighted disagreement/flooded tile to open the **Tile Inspector** tab.
  * *Voiceover:* "Instead of hiding behind black-box math, responders can flip directly between our 45-kilobyte Hybrid Q-SAR model and a classical CNN baseline—or highlight where they disagree. Clicking any tile gives a plain-language briefing on pre-to-post decibel shifts, local OSM bridges, and Unknown/Verify uncertainty flags—improving zero-shot real-SAR F1 from 0.638 to 0.665 across all three scenarios."

* **[0:48 – 1:18] Humanitarian Exposure & Dynamic Risk-Aversion OSM Routing**
  * *Screen:* Point to the bottom geodesic impact cards (`km² flooded`, `ha agriculture`, `OSM Bridges / Roads`, `Exposed Pop`), confirm **OSM (`ON`)** overlay, open the **Corridors** tab, and slide **Risk Aversion $\alpha$** from `0.25` (`Rapid Direct`) through `1.00` to `1.85` (`High-Ground Convoy` / `Max-Safety Perimeter`). Click **`GeoJSON`**.
  * *Voiceover:* "Beneath the radar map, we intersect hazards against real OpenStreetMap roads, bridges, and settlements. Now watch what happens on the road network as a dispatcher drags the Risk Aversion slider from urgent drone response toward cautious convoy dispatch: the multi-objective QAOA solver physically shifts routes around flooded chokepoints, compared directly against Classical Direct, Dijkstra, and A*—and exports as GeoJSON in one click."

* **[1:18 – 1:42] California Burn Scars, Vidarbha Crops & Honest 3-Track Benchmarks**
  * *Screen:* Switch briefly to **California Wildfire** (`Disagreement Diff` / `Hybrid Q-SAR`), open **`Impact & Benchmarks`** tab showing the 3-track table and `L=1` speckle stress.
  * *Voiceover:* "The exact same dual-polarization engine extends to Sierra Foothills wildfire canopy burn scars and Vidarbha agricultural stress, backed by an honest three-track evaluation separating synthetic NISAR ablations from zero-shot real Sentinel-1 transfer."

* **[1:42 – 2:00] Live NASA CMR & Closing**
  * *Screen:* Click **`NASA CMR`** (`OPERA_L2_RTC-S1_V1` / `NISAR` granule metadata footer) and **`NISAR Sim`**.
  * *Voiceover:* "Connected directly to NASA Earthdata via `earthaccess` and deployable offline or on Cloud Run in under 205 milliseconds per scan, Q-SAR Sentinel 2.0 turns raw radar into actionable humanitarian relief."

---

## 3. Deployment & Submission Checklist (Before Nov 15, 11:59 PM)

1. **Git Push Clean Repo:** Confirm `backend/.env` is excluded by `.gitignore` and push `backend/`, `frontend/`, `README.md`, and `NASA_SUBMISSION_GUIDE.md`.
2. **Deploy Frontend (Vercel):** Root `frontend`, set `NEXT_PUBLIC_API_URL` if backend is hosted (works completely standalone via `frontend/src/data/demo_cache.json`).
3. **Deploy Backend (Optional Live API on Render/Cloud Run):** Python `3.11`, `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
4. **Verify Before Submitting:** Test all 3 scenarios, `Real S1 RTC` / `NISAR Sim`, `Hybrid Q-SAR | Classical CNN | Disagreement Diff` switcher, `Risk Aversion` slider, OSM overlay, and `.geojson` export.
