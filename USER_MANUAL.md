# Q-SAR Sentinel 2.0: Field Operator Manual & User Guide

**Bi-Temporal SAR Disaster Assessment & Emergency Corridor Routing**  
*Document Version:* 2.0  
*Live Application:* [https://q-sar.vercel.app](https://q-sar.vercel.app)  
*Source Repository:* [https://github.com/Arjun-E-Naik/Q_SAR.git](https://github.com/Arjun-E-Naik/Q_SAR.git)

---

## Table of Contents
1. [System Purpose & Operational Scope](#1-system-purpose--operational-scope)
2. [Real-World Case Study: The Wainganga Monsoon Flood](#2-real-world-case-study-the-wainganga-monsoon-flood)
3. [Interface Architecture & Navigation Map](#3-interface-architecture--navigation-map)
4. [Header Controls & Data Ingestion](#4-header-controls--data-ingestion)
5. [Tactical Toolbar & Risk Configuration](#5-tactical-toolbar--risk-configuration)
6. [Interactive Map Viewport & SAR Visualization](#6-interactive-map-viewport--sar-visualization)
7. [Tile Inspector & Quantum Telemetry](#7-tile-inspector--quantum-telemetry)
8. [Corridor Routing Engine & XY-QAOA Solver](#8-corridor-routing-engine--xy-qaoa-solver)
9. [Humanitarian Impact & Verification Priority Queue](#9-humanitarian-impact--verification-priority-queue)
10. [Field Hand-off: SITREP & GeoJSON Export](#10-field-hand-off-sitrep--geojson-export)
11. [Troubleshooting & Offline Resilience](#11-troubleshooting--offline-resilience)

---

## 1. System Purpose & Operational Scope

During extreme weather events such as tropical monsoons, hurricanes, and severe wildfires, conventional optical satellite imaging (e.g., Sentinel-2, Landsat) is blinded by persistent cloud decks and heavy smoke plumes. 

**Synthetic Aperture Radar (SAR)** bypasses this limitation. By actively illuminating the Earth's surface with microwave pulses (C-band $\approx 5.6\text{ cm}$ on Sentinel-1; L-band $\approx 24\text{ cm}$ on NISAR), SAR penetrates clouds, torrential rain, and smoke day or night.

However, interpreting raw SAR backscatter in an emergency operations center presents three severe challenges:
1. **Radar Speckle & Misinterpretation:** Multiplicative speckle noise makes flooded roads resemble wet soil or flat tarmac.
2. **Disconnected Infrastructure:** Standard hazard rasters do not calculate exact road passability or tell dispatchers which bridges are compromised.
3. **Blind Shortest-Path Routing:** Traditional GPS routing directs emergency vehicles onto the shortest path, which during a disaster is often submerged or surrounded by active hazards.

**Q-SAR Sentinel 2.0** provides an end-to-end operational decision-support tool. It ingests bi-temporal dual-polarization SAR observations, processes them through a calibrated hybrid quantum-classical neural network (`HybridQSARNet`), clips road geometries to exact hazard boundaries, optimizes evacuation corridors using constraint-preserving XY-QAOA, and generates field-verifiable situation reports (SITREPs).

---

## 2. Real-World Case Study: The Wainganga Monsoon Flood

To understand how every button and view functions, consider an actual disaster response scenario handled by this system:

### The Scenario
* **Location:** Bhandara and Nagpur Districts, Maharashtra, India.
* **Event:** Unprecedented 48-hour monsoon downpours force the opening of 16 spillway gates at the upstream Gosikhurd Dam on the Wainganga River.
* **Problem:** Dense cloud cover prevents aerial drones and optical satellites from viewing the inundation. Water levels are rising rapidly around villages, cut-off farmlands, and the regional road network. The District Disaster Management Authority (DDMA) must dispatch medical supply convoys from the primary staging base to an isolated clinic on the eastern bank of the river.

### The Operational Walkthrough

```mermaid
flowchart TD
    A["Alert Received: Gosikhurd Dam Discharge"] --> B["1. Open Q-SAR Sentinel & Select 'Wainganga Flood'"]
    B --> C["2. Confirm C-Band Sentinel-1 RTC Provenance & Dates"]
    C --> D["3. Check Operational SAR Intelligence Brief for Headline"]
    D --> E["4. Open Priority Queue: Note Submerged Bridges & Facilities"]
    E --> F["5. Switch Layers: Co-Pol (VV) to Change (Δσ⁰) & ASF RGB"]
    F --> G["6. Inspect High-Entropy Tiles (Flagged Abstentions)"]
    G --> H["7. Set Risk Aversion α = 1.70 for Medical Convoy"]
    H --> I["8. Compare QAOA Corridor vs Direct Naive Route"]
    I --> J["9. Sign Off Route Approval in Incident Command Panel"]
    J --> K["10. Print Official SITREP & Push GeoJSON to Field GPS/ATAK"]
```

#### Step 1: Ingestion & Verification
The operator selects **Wainganga Flood** on the header bar. The system loads pre-flood ($2024\text{-}08\text{-}15$) and post-flood ($2024\text{-}08\text{-}27$) Radiometric Terrain Corrected (RTC) Sentinel-1 observations. The header pill turns green, confirming `ACTUAL SENTINEL-1 RTC · C-Band (VV+VH)` data is active.

#### Step 2: Immediate Situation Awareness
The **Operational SAR Intelligence Brief** summarizes the scene:
> *"Extensive specular inundation detected along the Wainganga river corridor. Co-pol backscatter drop exceeding -7 dB indicates open water expansion over agricultural lowlands."*

The KPI board immediately calculates:
* **Flooded Area:** $42.6\text{ km}^2$ (separating new flood expansion from the permanent riverbed).
* **Roads at Risk:** $38\text{ km}$ of OpenStreetMap roads intersecting flooded boundaries.
* **Bridges at Risk:** $4$ critical river crossings flagged for structural inspection.
* **Potentially Cut-off Population:** $\approx 18,400$ residents across intersecting rural settlements.

#### Step 3: Resolving Radar Ambiguities (The Tile Inspector)
Tile `T14` covers agricultural land near a village. The classical CNN predicts "Normal Terrain," but the hybrid quantum classifier predicts "Flood Inundation." The system flags `Disagreement` and sets an `Abstention` flag because the Shannon entropy is $0.89$ (exceeding the $0.85$ safety threshold).
The operator clicks `T14`:
* $\sigma^0_{\text{pre}}$ was $-11.2\text{ dB}$, and $\sigma^0_{\text{post}}$ dropped to $-18.7\text{ dB}$ ($\Delta\sigma^0 = -7.5\text{ dB}$).
* The Radar Vegetation Index (RVI) dropped by $45\%$.
* The double-bounce metric indicates flooded vegetation.
The operator clicks **`Approve Class`** to confirm the flood detection and adds the note: *"Standing water in rice paddies confirmed via double-bounce; road access cut off."*

#### Step 4: Medical Corridor Dispatch
The default shortest path (Naive Corridor 0) is $18.4\text{ km}$, but it crosses directly over two inundated bridges. 
The operator slides **Risk Aversion ($\alpha$)** to `1.70` (**Medical Preset**). The **XY-QAOA routing engine** evaluates alternative routes on the contracted road graph. 
The system selects **Corridor 1 (QAOA Selected)**:
* Route length: $24.8\text{ km}$ ($+6.4\text{ km}$ detour).
* Hazard exposure: $0.0\text{ km}$ inside flooded tiles.
* Bridges compromised: $0$.
The incident commander changes the route status to **`Approved for Recon`**.

#### Step 5: Field Deliverables
The operator clicks **`SITREP`** to generate an official 2-page briefing for ground commanders and downloads the **`GeoJSON`** file containing the clipped road lines and evacuation corridors for ATAK/QGIS devices.

---

## 3. Interface Architecture & Navigation Map

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ [LOGO] Q-SAR SENTINEL 2.0  [PROVENANCE]  [SCENARIO TABS]  [SEARCH] [S1/NISAR] [L=4/L=1] [SITREP] │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [CLASSIFICATION: Hybrid | Classical | Disagree]   [RISK α: ──●── 1.00]  [Drone|Rescue|Med] [CMR] │
├─────────────────────────────────────────────────┬────────────────────────────────────────────────┤
│                                                 │ [RADAR & QUANTUM TELEMETRY PANEL]              │
│  CENTRAL MAP VIEWPORT                           │  Tabs: [Tile Inspector | Corridors | Validate] │
│                                                 │                                                │
│  - Satellite / Dark Gray Basemap                │  - Quantum Probabilities (4 classes)           │
│  - SAR Layers (VV, VH, Change dB, RGB, Swipe)   │  - Shannon Entropy & Purity                    │
│  - Tile Hazard Outlines (36-tile grid)          │  - Dual-pol dB Values & Double-Bounce          │
│  - Clipped Road Hazards & Compromised Bridges   │  - Human-in-the-Loop Triage Buttons            │
│  - 4 Candidate Evacuation Corridors             │  - XY-QAOA Corridor Selector & Benchmarks      │
│                                                 │  - Incident Commander Approval Control         │
├─────────────────────────────────────────────────┴────────────────────────────────────────────────┤
│ HUMANITARIAN KPI STRIP: Flooded km² | Burn km² | Agri ha | Roads at Risk | Pop Exposed | Route   │
├─────────────────────────────────────────────────┬────────────────────────────────────────────────┤
│ OPERATIONAL INTELLIGENCE BRIEF                  │ PRIORITY ACTION QUEUE & INFRASTRUCTURE CHOKE   │
│ - Headline & What Changed                       │ - Ranked Tiles (Severity + Bridge + Hospitals) │
│ - Polarimetric Evidence & Confidence            │ - Quick-Triage Buttons (Verify / Flag)         │
│ - Recommended Field Action                      │ - Persistent Operator State Counter            │
└─────────────────────────────────────────────────┴────────────────────────────────────────────────┘
```

---

## 4. Header Controls & Data Ingestion

Every button in the top navigation bar manages satellite data ingestion, scenarios, and exports:

| Control Name | Location | Function & Purpose | Operational Significance |
|---|---|---|---|
| **Satellite Logo** | Top Left | System identifier and reset anchor. | Clicking refreshes the primary dashboard state. |
| **Provenance Pill** | Top Left | Displays active sensor tier (Actual Sentinel-1, Actual NISAR, Physics Sim, or Demo Snapshot). | **Critical for judges & commanders:** Prevents confusing simulated data with real orbital observations. |
| **Scenario Tabs** | Center Left | Switches between **Wainganga Flood**, **Park Fire Scar**, and **Vidarbha Crop Stress**. | Instantly loads verified satellite observations for preset disaster regions. |
| **Custom Scenario Pills** | Center Left | Displays newly searched locations with a globe icon. | Keeps recently provisioned areas accessible with one click. |
| **`Search Area` Button** | Center | Opens the dynamic global area-of-interest (AOI) search popover. | Allows evaluating any location on Earth beyond presets. |
| **Search Input Field** | Search Popover | Text input for place names (e.g., *"Houston"*) or coordinates (`29.76, -95.37`). | Geocoded via OpenStreetMap Nominatim with instant candidate dropdown. |
| **Pre/Post Date Pickers** | Search Popover | Optional input fields to restrict satellite pass search windows. | Allows targeting specific disaster timeframes; leaves blank to auto-fetch the latest matching pair. |
| **`Scan` Button** | Search Popover | Fetches Sentinel-1 RTC Cloud-Optimized GeoTIFFs from Planetary Computer and OSM vector roads. | Sets up a complete live disaster assessment for a new location in under 15 seconds. |
| **`Real S1-RTC` / `NISAR Sim`** | Center Right | Toggles between authentic Sentinel-1 C-band imagery and NISAR L-band physics simulations. | Allows comparing C-band surface sensitivity against L-band canopy penetration. |
| **`L=4 Std` / `L=1 Stress`** | Center Right | Toggles multi-look averaging (4 looks) vs raw single-look speckle stress (1 look). | Tests how resilient the quantum network is when receiving noisy, unfiltered satellite data. |
| **`Re-Scan` (Refresh)** | Center Right | Forces a fresh execution of the entire backend analysis pipeline. | Recomputes quantum states, road clippings, and QAOA routes after parameter changes. |
| **`GeoJSON` Button** | Right | Downloads enriched RFC 7946 GeoJSON vector data. | Directly imports hazard tiles, clipped roads, and candidate routes into QGIS, ArcGIS, or ATAK. |
| **`SITREP` Button** | Right | Launches the browser print/PDF preview formatted as an emergency situation report. | Creates a 2-page executive brief for field commanders without opening GIS software. |

---

## 5. Tactical Toolbar & Risk Configuration

Located directly under the header, this toolbar configures how the map highlights hazards and how routes balance risk:

### 1. Map View Classification Modes
* **`Hybrid Q-SAR`:** Displays hazard tiles as classified by the 4-qubit quantum-classical neural network.
* **`Classical CNN`:** Displays hazard tiles as classified by the classical convolutional baseline.
* **`Disagreement Diff` (Amber):** Hides agreeing tiles and highlights *only* the tiles where quantum and classical models disagree.
  * *Why this matters:* When algorithms disagree, it indicates subtle radar physics (e.g., wet soil vs shallow water). Operators can focus triage efforts specifically on these borderline areas.

### 2. Risk Aversion Slider ($\alpha \in [0.2, 2.0]$)
The routing engine optimizes a cost function combining path distance and hazard penalties:
$$\text{Cost}(\text{Route}) = \text{Distance (km)} + \alpha \times \sum (\text{Hazard Penalty})$$

* **Continuous Slider:** Allows dragging $\alpha$ in steps of $0.05$.
* **`Drone` Preset ($\alpha = 0.35$):** Emphasizes short distances for battery-constrained aerial drones that can fly over flooded roads.
* **`Rescue` Preset ($\alpha = 1.00$):** Balanced safety profile for high-clearance tactical vehicles that can handle minor waterlogging.
* **`Medical` Preset ($\alpha = 1.70$):** High-safety profile for standard civilian ambulances and supply trucks; forces wide detours to guarantee zero hazard exposure.

### 3. `NASA CMR` Toggle Button
Queries the **NASA Earthdata Common Metadata Repository** API for available `NISAR_L2_GCOV_PROVISIONAL_V1` and `OPERA_L2_RTC-S1_V1` granules covering the active bounding box, showing the matching granule count.

---

## 6. Interactive Map Viewport & SAR Visualization

The central map is built on MapLibre GL with high-resolution satellite basemaps and vector layers.

### Map Controls (Top Left of Map)
1. **Basemap Toggle (`Satellite` / `Dark`):**
   * **`Satellite`:** Maxar / Esri World Imagery showing actual terrain, towns, and vegetation.
   * **`Dark`:** High-contrast dark gray canvas that maximizes visibility of radar difference colors and route lines.
2. **Layer Mode Selector:**
   * **`Hazard Seg` (Segmentation):** Color-coded hazard overlays (Blue = Flood, Orange = Fire, Yellow = Crop Stress).
   * **`Change Δσ⁰`:** Decibel difference map ($\sigma^0_{\text{post}} - \sigma^0_{\text{pre}}$). Blue/purple indicates significant backscatter loss (inundation or canopy loss); red/yellow indicates backscatter gain (double-bounce).
   * **`Co-Pol`:** Raw Co-polarization backscatter (`VV` or `HH`). Flat water acts as a specular mirror, reflecting microwave pulses away and appearing pitch black.
   * **`Cross-Pol`:** Raw Cross-polarization backscatter (`VH` or `HV`). Reflects volumetric scattering from leaves and tree branches; sharp drops indicate defoliation or burn scars.
   * **`ASF RGB`:** Alaska Satellite Facility false-color composite:
     * $\text{Red} = \text{Co-Pol backscatter}$
     * $\text{Green} = \text{Cross-Pol backscatter}$
     * $\text{Blue} = \text{Co-Pol / Cross-Pol Ratio}$
   * **`Swipe`:** Activates a split-screen slider on the map. Dragging the center handle left or right compares the pre-disaster baseline directly against the post-disaster radar layer.
3. **Tile Display Mode (`Hazards` / `All` / `Off`):**
   * **`Hazards` (Default):** Shows only the tiles identified as hazardous, keeping unaffected areas clean.
   * **`All`:** Shows the full $6 \times 6$ grid (36 tiles) across the 24 km scene.
   * **`Off`:** Hides tile outlines completely to inspect underlying satellite basemaps and roads.
4. **Interactive Routing Waypoint Pickers (`A`, `B`, `Reset`):**
   * **`Inspect`:** Default map click mode. Clicking any tile selects it and populates the Tile Inspector.
   * **`Origin (A)` (Crosshair):** Click anywhere on the road network to define a custom starting base.
   * **`Target (B)` (Flag):** Click anywhere on the map to define a custom evacuation destination.
   * **`Reset Route` (Undo):** Reverts start and end nodes to the scenario defaults.
5. **Fullscreen Expand (`⛶`):**
   * Expands the map across the entire screen width, ideal for multi-monitor command center displays.

---

## 7. Tile Inspector & Quantum Telemetry

When an operator clicks any tile on the map, the right-hand panel displays its radar physics and quantum circuit state:

```
┌────────────────────────────────────────────────────────┐
│ TILE INSPECTOR: T14 [2,2] · 20.76°N, 79.62°E           │
├────────────────────────────────────────────────────────┤
│ Quantum Prediction: Flood Inundation (94.2%)           │
│ Classical Baseline: Normal Terrain (51.8%)             │
│ [!] MODELS DISAGREE — OPERATOR VERIFICATION REQUIRED   │
├────────────────────────────────────────────────────────┤
│ QUANTUM CIRCUIT TELEMETRY:                             │
│ - Shannon Entropy H(p): 0.89 (Threshold: 0.85)         │
│ - State Purity γ: 0.82                                 │
│ - Abstention: YES (Flagged as Uncertain)               │
│ - 4-Qubit Variational Layer: AngleEmbedding + SEL      │
├────────────────────────────────────────────────────────┤
│ RADAR POLARIMETRY:                                     │
│ - Pre-Event σ⁰: -11.2 dB  ·  Post-Event σ⁰: -18.7 dB   │
│ - Backscatter Change Δσ⁰: -7.5 dB                      │
│ - Radar Vegetation Index (RVI): 0.28 (Drop: 45%)       │
│ - Double-Bounce Signature: DETECTED                    │
├────────────────────────────────────────────────────────┤
│ HUMAN-IN-THE-LOOP TRIAGE:                              │
│ [ Approve Flood ]  [ Flag Uncertain ]  [ False Alarm ] │
│ Note: "Standing water in paddies cut off road access." │
└────────────────────────────────────────────────────────┘
```

### Key Quantum & Radar Metrics Explained
* **Shannon Entropy ($H(p)$):** Measures disorder across the 4 class output probabilities. When $H(p) > 0.85$, the quantum network abstains from making an autonomous decision, requiring human verification.
* **Quantum Purity ($\gamma$):** Evaluates mixedness of the density matrix ($\gamma = \text{Tr}(\rho^2)$). Higher purity indicates sharp quantum state alignment.
* **Double-Bounce Flag:** Indicates corner reflection where radar pulses bounce off water and then off standing trees or crops back to the satellite. This prevents mistaking flooded forests for dry land.

---

## 8. Corridor Routing Engine & XY-QAOA Solver

The **Corridors** tab visualizes emergency transit routes across the OpenStreetMap road network:

### The 4 Candidate Corridors
1. **Corridor 0 (Direct / Shortest):** The standard shortest-path route computed without hazard awareness. In flooded scenarios, it almost always cuts through hazardous tiles.
2. **Corridor 1 (QAOA Selected):** The optimal route selected by the Quantum Approximate Optimization Algorithm. Balances distance against hazard clearance based on the active $\alpha$ setting.
3. **Corridor 2 (Balanced Compromise):** A secondary alternative corridor using intermediate arterial roads.
4. **Corridor 3 (Maximum Detour):** A wide detour utilizing perimeter highways, prioritizing zero hazard exposure above all else.

### The XY-QAOA Advantage & Classical Baseline
* **Constraint-Preserving XY Mixer:** Standard QAOA uses single-qubit Pauli-X mixers that leak probability mass into invalid multi-corridor states. Q-SAR Sentinel uses an **XY ring mixer** that confines evolution strictly to the Hamming-weight-1 subspace ($\sum x_i = 1$) with $\ge 99\%$ valid probability mass.
* **Transparent Classical Benchmark:** For complete scientific honesty, the system benchmarks the quantum solver against **Classical Direct Enumeration** ($<0.1\text{ ms}$), showing exact execution times alongside Dijkstra and A*.

### Incident Commander Approval
Ground convoys should never be dispatched purely on automated AI output. The panel provides an **Approval Control**:
* `Pending Human Approval` (Default amber badge)
* `Approved for Reconnaissance` (Green badge; unlocks field dispatch)
* `Rejected / Unsafe` (Rose badge; marks corridor closed)

---

## 9. Humanitarian Impact & Verification Priority Queue

The bottom section provides situational intelligence for disaster coordination:

### Humanitarian KPI Strip
* **Flooded Area:** Square kilometers of detected standing water, split into *new flood expansion* and *pre-existing permanent water*.
* **Burn Scar Area:** Square kilometers of destroyed forest canopy.
* **Agricultural Loss:** Cropland affected, displayed in hectares.
* **Roads at Risk:** Kilometers of road inside hazard boundaries, computed via **Liang–Barsky geometric clipping** rather than whole-road summation.
* **Compromised Bridges:** Total bridge crossings located inside active hazard tiles.
* **Potentially Exposed Population:** Estimated population residing within intersecting settlement areas.

### Priority Action Queue (Top Chokepoints)
Ranks all 36 tiles by a weighted emergency score:
$$\text{Score} = (\text{Hazard Severity} \times 40) + (\text{Facilities} \times 25) + (\text{Bridges} \times 20) + (\text{Model Uncertainty} \times 15)$$

Clicking any tile in the priority queue immediately centers the map and opens its telemetry in the Tile Inspector for one-click verification.

---

## 10. Field Hand-off: SITREP & GeoJSON Export

### 1. The Printable SITREP (Situation Report)
Clicking **`SITREP`** in the header formats the entire dashboard into a clean, two-page printable document designed for field commanders:
* Header with Incident Name, Coordinates, and Sensor Metadata.
* Key Impact Statistics (inundated km², compromised bridges, cut-off hospitals).
* Selected Corridor Logistics (length, transit time, hazard clearance).
* Priority Chokepoint Action Items.
* Formal Sign-off Box for Incident Commander signature and authorization timestamp.

### 2. Enriched GeoJSON Export
Clicking **`GeoJSON`** exports an RFC 7946 GeoJSON file containing:
* 36 SAR tile bounding boxes with polarimetric decibel values.
* Clipped road segments with passability flags.
* OpenStreetMap critical facilities (hospitals, shelters, police stations).
* All 4 candidate corridor polylines with approval status and cautions.

---

## 11. Troubleshooting & Offline Resilience

* **Zero-Downtime Demo Mode (Vercel):**  
  When running on cloud hosting without a dedicated GPU backend, the application automatically loads **`demo_cache.json`**. All three preset scenarios (**Wainganga Flood**, **Park Fire Scar**, and **Vidarbha Crop Stress**) are fully interactive with genuine satellite backscatter, road clipping, and routing. The header badge displays `VERIFIED DEMO SNAPSHOT`.
* **Starting the Full Local Backend (for Live Global Search):**  
  To search any custom coordinate on Earth, start the local backend using the Windows PowerShell helper:
  ```powershell
  .\start_local.ps1
  ```
  Or run manually:
  ```bash
  # Terminal 1: Backend
  uvicorn backend.main:app --host 127.0.0.1 --port 8000

  # Terminal 2: Frontend
  cd frontend
  npm run dev
  ```
  Once running, the header badge will switch to `API Live`, and you can search and analyze any global city or river basin on Earth.