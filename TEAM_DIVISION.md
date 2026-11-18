# PathSnap: Team Workload Division & Module Ownership
**PES University — UE24CS352A: Machine Learning Hackathon**

**Team Number:** 10  
**Project:** PathSnap: Distributed Map-Matching & Sequential Route Reconstruction System  
**Repository:** [https://github.com/avulabalaji35-rgb/ml-hackathon](https://github.com/avulabalaji35-rgb/ml-hackathon)

---

## 1. Executive Workload Division: Backend vs Frontend Tracks

The team is structured into two collaborative engineering tracks:
* **BACKEND ENGINEERING TRACK:**
  - `PES2UG24AM035` — **AVULA BALAJI** *(Backend Lead: Spatial Pipeline, Graphs & API Server)*
  - `PES2UG24AM056` — **DURUGESH** *(Backend Lead: Machine Learning, Bayesian Optimization & HMM Engine)*
* **FRONTEND & EVALUATION TRACK:**
  - `PES2UG25AM810` — **YASHAS V** *(Frontend Lead: UI Dashboard Architecture & System Documentation)*
  - `PES2UG24AM051` — **DEEPTI M AHUJA** *(Frontend Lead: Interactive Map Visualization & Exploratory Analysis)*

```
                     End-to-End System Integration Flow
┌───────────────────────────────────────────────┐      ┌───────────────────────────────────────────────┐
│              BACKEND TRACK                    │      │              FRONTEND TRACK                   │
│                                               │      │                                               │
│  [Member 035: AVULA BALAJI]                   │      │  [Member 810: YASHAS V]                       │
│  • Haversine Distance Engine                  │      │  • Evaluator UI Layout & Styling (ui/)        │
│  • Batch STRtree Candidate Generator (R=35m)  │      │  • Dynamic Remote LAN API Connection Client   │
│  • 14-D Spatial/Kinematic Feature Extractor   │ ───► │  • Real-Time KPI Cards (Acc, Cont, Latency)   │
│  • FastAPI High-Throughput Server (main.py)   │      │  • Publication-Grade Technical Reports (PDF)  │
│                                               │      │                                               │
│  [Member 056: DURUGESH]                       │      │  [Member 051: DEEPTI M AHUJA]                 │
│  • 5-Family ML Benchmark (GroupKFold)         │      │  • Dual-Layer Leaflet Route Visualizer        │
│  • Optuna Bayesian Hyperparameter Tuning      │      │  • Multi-Layer Folium Interactive Maps        │
│  • Map-Matching HMM & Viterbi Decoder         │      │  • Esri WorldStreetMap (Zero 403 Tile Blocks) │
│  • Route Continuity Engine (77.1% -> 99.8%)   │      │  • Exploratory Data Analysis (EDA) Notebook   │
└───────────────────────────────────────────────┘      └───────────────────────────────────────────────┘
```

---

## 2. Detailed Technical Breakdown & Module Ownership

### 🛠️ BACKEND TRACK

#### Member 035: AVULA BALAJI — Backend Lead (Spatial Pipeline, Graphs & API Server)
* **Core Responsibilities**:
  1. **Geodesic Distance Engine (`src/data/loader.py`)**: Implemented spherical Haversine distance calculations ($R = 6,371,000$ m) to prevent Euclidean degree distortions.
  2. **Spatial Indexing & Candidate Generation (`src/features/extractor.py`)**: Built batch 2D `STRtree` spatial indexing over directed road graphs. Swept candidate search radii from 10m to 100m, establishing $R = 35$m as the optimal threshold (>98.5% recall, ~8 candidates per fix, reducing candidate search time from 30s to <2s per trip).
  3. **14-Dimensional Feature Extractor**: Engineered perpendicular distance, projection offset ratio, heading divergence, inter-fix velocity, speed-limit divergence, arc length, oneway constraint, node proximity, and wrap-around invariant bearing sine/cosine.
  4. **High-Throughput FastAPI Server (`src/api/server.py`, `main.py`)**: Built asynchronous endpoints (`/health`, `/api/trips`, `/api/evaluate/{trip_id}`, `/api/map/{trip_id}`), handling CORS and server-side evaluation.
  5. **Data Datasheet & Schemas (`DATASET.md`)**: Ingested and cataloged all 90 trips (80 Train, 10 Student Test; 227,577 GPS fixes) and formatted sample data.
* **Viva Speaking Point**:
  > *"I engineered the spatial foundation and backend server infrastructure. Raw GPS data contains severe multipath noise and drift. By building a vectorized STRtree spatial index, I reduced candidate search from 30 seconds to under 2 seconds per trip, capturing over 98.5% of true road segments at a 35m radius. I extracted 14 geometric, angular, and kinematic features that feed our ML models, and developed the FastAPI backend server that executes trajectory evaluations in sub-70ms without leaking ground truth."*

---

#### Member 056: DURUGESH — Backend Lead (Machine Learning, Bayesian Optimization & HMM Engine)
* **Core Responsibilities**:
  1. **Multi-Model Benchmark Suite (`scripts/train_compare_models.py`)**: Benchmarked 5 distinct algorithm families (Logistic Regression, Decision Trees, Random Forest, HistGradientBoosting, MLP Neural Network).
  2. **Leakage-Free Cross-Validation**: Implemented 5-Fold `GroupKFold` grouped strictly by `trip_id` across 28,779 training candidates, eliminating spatial-temporal leakage.
  3. **Bayesian Hyperparameter Optimization (`scripts/tune_model_bayes.py`)**: Executed 25 Optuna TPE trials on HistGradientBoosting. Pruned tree depth and optimized regularizers, cutting inference latency by **60.4%** (from 80.55 ms to 31.91 ms per trip) while maintaining peak Macro F1 (0.9356) and lowest log loss (0.0983).
  4. **Sequential Hidden Markov Model & Viterbi Decoder (`src/hmm/decoder.py`, `scripts/run_hmm_evaluation.py`)**: Formulated the map-matching HMM combining calibrated classifier emission probabilities with Dijkstra shortest-path topological transition penalties and exponential temporal gap scaling $\exp(-\Delta t / \tau)$.
  5. **Continuity Boost**: Raised route topological continuity from **77.15%** (classifier-only) to **99.80%**, and Arc Precision/Recall to 94.75%.
* **Viva Speaking Point**:
  > *"I owned the machine learning, hyperparameter tuning, and sequential route reconstruction engine. To ensure zero data leakage, I enforced 5-Fold GroupKFold grouped strictly by trip_id across five algorithm families. HistGradientBoosting delivered the lowest log loss of 0.0983. Using 25 trials of Optuna Bayesian optimization, I slashed inference latency by 60.4% down to 31.91 ms per trip. I then formulated the HMM and negative log-space Viterbi decoder. Pointwise classifiers break route continuity down to 77.15%, but my HMM decoder raised continuity to 99.80%, ensuring 100% physically driveable routes."*

---

### 🎨 FRONTEND & EVALUATION TRACK

#### Member 810: YASHAS V — Frontend Lead (UI Dashboard Architecture & System Documentation)
* **Core Responsibilities**:
  1. **Evaluator Dashboard UI Architecture (`ui/index.html`, `ui/style.css`)**: Built the responsive, modern UI dashboard using clean semantic HTML5 and CSS3 (custom cards, status badges, flex/grid layouts).
  2. **Dynamic Client Controller (`ui/app.js`)**: Implemented dynamic remote API origin resolution. When loaded on Laptop 2 via `http://<IP>:8000/ui/index.html`, it automatically connects to Laptop 1's backend with zero manual configuration.
  3. **Real-Time KPI Metric Cards**: Designed and wired the live metrics display (Top-1 Accuracy, Route Topological Continuity %, Latency, Reconstructed Arcs vs Ground Truth).
  4. **Two-Laptop Client-Server Integration & Security**: Verified zero data leakage—confirmed that Laptop 2 only receives computed KPI metrics and polyline coordinates, while raw ground truth remains strictly quarantined on Laptop 1.
  5. **Publication-Grade Technical Report Compilation (`scripts/generate_two_page_report.py`, `scripts/generate_technical_report.py`)**: Authored programmatic ReportLab PDF generators compiling the comprehensive report and the exact 2-page executive summary report.
* **Viva Speaking Point**:
  > *"I designed the evaluator dashboard UI architecture, client-server communication, and formal technical reports. I built the responsive Web UI so that Laptop 2 acts strictly as an evaluator client. When loaded across the local network, our JavaScript client automatically detects Laptop 1's IP address, communicates over asynchronous REST APIs, and renders real-time KPI metrics in under 70 milliseconds. I verified that no ground-truth files leave Laptop 1, preserving complete data integrity. I also developed the ReportLab engines that compiled our 2-page executive summary and full technical report."*

---

#### Member 051: DEEPTI M AHUJA — Frontend Lead (Interactive Map Visualization & Exploratory Analysis)
* **Core Responsibilities**:
  1. **Interactive Leaflet Route Visualizer (`ui/app.js`)**: Built the dual-layer map interface rendering raw noisy GPS observation markers (red), ground truth path (blue), and HMM predicted reconstructed route (green polyline).
  2. **Basemap 403 Access-Block Resolution**: Replaced volunteer OpenStreetMap tile URLs with high-availability Esri / ArcGIS WorldStreetMap tiles, eliminating HTTP 403 Forbidden blocks under `file:///` and localhost protocols.
  3. **Multi-Layer Folium Interactive Maps (`src/visualization/folium_maps.py`, `reports/maps/`)**: Built interactive HTML maps with layer toggles, vehicle telemetry popups, and road network arc overlays for training and test trips.
  4. **Exploratory Data Analysis (EDA) Jupyter Notebook (`notebooks/01_exploratory_data_analysis.ipynb`)**: Executed the complete publication-grade EDA notebook (5.5 MB) analyzing 227,577 GPS fixes, sampling intervals (94.32% at 1s), extreme velocity spikes (>140 km/h up to 648 km/h), and road graph degree distributions.
  5. **Client-Side Telemetry & Diagnostic Plots**: Produced all six diagnostic figures (`reports/figures/eda_*.png`) and integrated telemetry inspector tooltips into the map view.
* **Viva Speaking Point**:
  > *"I developed the interactive mapping and exploratory data analysis visualization. When displaying trajectories, opening maps locally can trigger HTTP 403 tile errors; I resolved this by integrating ArcGIS WorldStreetMap tiles, guaranteeing smooth rendering without API keys. I built the Leaflet dual-layer visualizer that allows the evaluator to visually inspect raw GPS points, ground truth, and the reconstructed path side-by-side. I also executed the complete EDA notebook across all 90 trips, uncovering extreme GPS jumps of up to 648 km/h and generating the publication plots for our deliverables."*

---

## 3. Cross-Track Hand-off Matrix

| Producing Track | Hand-off Artifact / API | Consuming Track | Interface Functionality |
| :---: | :---: | :---: | :--- |
| **Backend (035)** | `data/train_features.csv` | **Backend (056)** | 14-D spatial features ingested by GroupKFold model training. |
| **Backend (056)** | `models/tuned_model.joblib` & HMM | **Backend (035)** | Tuned model and Viterbi decoder wrapped inside FastAPI `/api/evaluate/{trip_id}`. |
| **Backend (035, 056)** | REST JSON API & `/api/map/{trip_id}` | **Frontend (810, 051)** | Laptop 1 streams evaluation metrics, polylines, and Folium maps to Laptop 2. |
| **Frontend (810)** | Dashboard UI & Origin Handler | **Frontend (051)** | UI shell embeds interactive Leaflet and Folium map canvases. |
| **Frontend (810, 051)** | Two-Page Report & Visuals | **Evaluators / PESU** | 2-Page Executive Summary PDF (`reports/PathSnap_Two_Page_Report.pdf`). |
