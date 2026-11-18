# PathSnap: Distributed Map-Matching & Route Reconstruction
**PES University — UE24CS352A: Machine Learning Hackathon**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Conda Env](https://img.shields.io/badge/conda%20env-ml__hackathon-green.svg)](https://docs.conda.io/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Leaflet](https://img.shields.io/badge/frontend-Leaflet%20UI-199900.svg)](https://leafletjs.com/)
[![Optuna](https://img.shields.io/badge/tuning-Optuna%20TPE-4169E1.svg)](https://optuna.org/)
[![ReportLab](https://img.shields.io/badge/report-PDF%20Compiled-red.svg)](reports/PathSnap_Technical_Report.pdf)

---

## 1. Executive Summary & Overview
GPS telemetry inherently suffers from severe urban multipath noise, sensor jitter, signal dropouts, and low sampling intervals. When plotted directly, vehicles appear to drift onto parallel side streets, leap across city blocks, or cut across non-navigable buildings.

**PathSnap** is a high-performance, distributed machine learning and sequential probabilistic inference system designed to match noisy GPS trajectories to an underlying directed road network and reconstruct topologically continuous routes.

This repository provides the complete, production-grade implementation of all hackathon requirements:
1. **Exploratory Data Analysis (EDA)** across 90 trips (80 Train, 10 Student Test; 227,577 GPS fixes) in a fully executed Jupyter Notebook (`notebooks/01_exploratory_data_analysis.ipynb`).
2. **Spatial Candidate Generation & Feature Extraction** ($R=35$m candidate search, batch Shapely 2.0 STRtree, 14 geometric, topological, velocity, and angular features).
3. **Multi-Model Benchmark Comparison** across 5 algorithm families evaluated via 5-fold `GroupKFold` grouped strictly by `trip_id`.
4. **Bayesian Hyperparameter Optimization** (Optuna TPE, 25 trials) yielding a 60.4% inference latency reduction.
5. **Hidden Markov Model (HMM) & Viterbi Decoder** incorporating shortest-path road network transition penalties and exponential temporal gap scaling, lifting route continuity from **77.15% to 99.80%**.
6. **Two-Laptop Distributed Deployment Architecture** with a high-throughput FastAPI backend and an evaluator-facing Leaflet UI dashboard.
7. **Strict Ground-Truth Isolation Guarantee** ensuring evaluator test data is physically sequestered on Laptop 1 and never exposed to the client or training pipelines.
8. **Comprehensive Technical Report** compiled directly into `reports/PathSnap_Technical_Report.pdf`.

---

## 2. System Architecture

```
                          Two-Laptop Deployment Topology
========================================================================================

  [ LAPTOP 2: Evaluator Client ]                         [ LAPTOP 1: PathSnap Server ]
  ┌───────────────────────────────┐                      ┌───────────────────────────────┐
  │  Browser Web UI (HTML/CSS/JS) │                      │   Uvicorn / FastAPI Backend   │
  │  Leaflet.js Multi-Layer Map   │                      │   Port: 8000                  │
  │                               │                      │                               │
  │  1. Configure Server IP       │─── HTTP GET /trips ─►│  Trip Enumerator & Loader     │
  │  2. Select Test Trip          │                      │                               │
  │  3. Trigger Evaluation Run    │─── POST /evaluate ──►│  1. Spatial Candidate Gen     │
  │                               │    {trip_id}         │     (STRtree, R=35m)          │
  │                               │                      │  2. Feature Extractor (14-D)  │
  │  4. Display KPI Summary Cards │                      │  3. Tuned HistGradientBoost   │
  │     • Accuracy, Precision     │◄── JSON Results ─────│  4. NetworkX Viterbi HMM      │
  │     • Recall, Continuity %    │    {metrics, path,   │  5. Ground Truth Evaluator    │
  │     • Execution Latency       │     gps_points}      │     (Strictly Isolated!)      │
  │                               │                      │                               │
  │  5. Interactive Map View      │                      │  Protected Storage:           │
  │     • Red dashed: Noisy GPS   │                      │  data/evaluator_test/         │
  │     • Green: Reconstructed    │                      │  (Never leaves Laptop 1)      │
  │     • Blue: Ground Truth      │                      └───────────────────────────────┘
  └───────────────────────────────┘
```

---

## 3. Benchmark Results & Key Findings

### 3.1 Multi-Model Benchmark (5-Fold GroupKFold by Trip)

| Model Family | Algorithm | 5-Fold CV Macro F1 | Top-1 Candidate Acc | Log Loss | Per-Trip Infer Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Linear / Margin | Logistic Regression | $0.9157 \pm 0.0207$ | 91.53% | 0.1518 | 8.27 ms |
| Tree-based (Single) | Decision Tree | $0.9249 \pm 0.0202$ | 91.97% | 0.2469 | 4.06 ms |
| Bagging Ensemble | Random Forest | $0.9316 \pm 0.0199$ | 92.51% | 0.1002 | 252.67 ms |
| **Boosting Ensemble** | **HistGradientBoosting** | $\mathbf{0.9355 \pm 0.0198}$ | $\mathbf{92.61\%}$ | $\mathbf{0.0983}$ | **52.16 ms** |
| Deep Learning | MLP Neural Network | $0.9091 \pm 0.0260$ | 92.80% | 0.1061 | 29.05 ms |

**Winner**: **HistGradientBoosting** achieved the highest cross-validation stability, lowest multi-candidate log loss, and exceptional inference efficiency.

### 3.2 Bayesian Optimization (Optuna TPE)
- **Tuned Hyperparameters**: `learning_rate = 0.1513`, `max_iter = 180`, `max_depth = 6`, `min_samples_leaf = 35`, `l2_regularization = 0.8050`.
- **Latency Reduction**: From **80.55 ms** to **31.91 ms** per trip (**60.4% speedup**).
- **Validation Macro F1**: Maintained peak generalization performance of **0.9356**.

### 3.3 HMM Viterbi Sequential Route Reconstruction
Point-wise classifier emissions alone fail to ensure physical driveability. Incorporating topological graph shortest-path transition probabilities via Viterbi decoding dramatically improves path viability:

| Evaluation Metric | Classifier-Only (Top-1) | HMM + Viterbi Decoder | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Topological Continuity** | 77.15% | **99.80%** | **+22.65% absolute (+29.4% relative)** |
| **Arc Precision** | 92.61% | **94.75%** | +2.14% absolute |
| **Arc Recall** | 92.61% | **94.75%** | +2.14% absolute |
| **Overall Macro F1** | 0.9355 | **0.9475** | +0.0120 |
| **Average End-to-End Latency**| 31.91 ms | **68.42 ms** | Real-time capable (< 100 ms) |

---

## 4. Repository Structure

```
ml-hackathon/
├── DATASET.md                       # Comprehensive dataset datasheet & specification
├── environment.yml                  # Conda environment definition (ml_hackathon)
├── requirements.txt                 # Pip dependencies
├── main.py                          # Unified CLI entry point & server runner
├── README.md                        # Complete project documentation
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Fully executed 5.5MB EDA notebook
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py                # FastAPI server (health, trips, evaluate, map)
│   ├── data/
│   │   ├── loader.py                # Geodesic calculation, parser, loader
│   │   └── dataset.py               # PathSnapDataset high-level API
│   ├── features/
│   │   ├── __init__.py
│   │   └── extractor.py             # STRtree spatial candidate & feature extraction
│   ├── hmm/
│   │   ├── __init__.py
│   │   └── decoder.py               # MapMatchingHMM & Viterbi sequential decoder
│   └── visualization/
│       └── folium_maps.py           # Multi-layer Folium interactive map builder
├── ui/                              # Evaluator client dashboard
│   ├── index.html                   # HTML5 responsive UI layout
│   ├── style.css                    # Professional dark/light modern UI styling
│   └── app.js                       # Client controller, API client, Leaflet renderer
├── models/                          # Serialized trained model binaries
│   ├── tuned_model.joblib           # Optuna-tuned HistGradientBoosting model
│   ├── histgradientboosting_model.joblib
│   ├── random_forest_model.joblib
│   ├── decision_tree_model.joblib
│   ├── logistic_regression_model.joblib
│   └── mlp_neural_network_model.joblib
├── scripts/
│   ├── compute_eda_stats.py         # Dataset inventory and telemetry metrics
│   ├── analyze_candidates.py        # Spatial candidate radius trade-off analysis
│   ├── generate_eda_visualizations.py # EDA figures and interactive Folium maps
│   ├── extract_features.py          # Batch feature extraction for train/test
│   ├── train_compare_models.py      # Multi-model benchmarking (5 families)
│   ├── tune_model_bayes.py          # Optuna Bayesian hyperparameter optimization
│   ├── run_hmm_evaluation.py        # HMM Viterbi trajectory evaluation
│   ├── build_and_execute_notebook.py# Programmatic notebook generation & execution
│   └── generate_technical_report.py # ReportLab PDF technical report generator
└── reports/
    ├── PathSnap_Technical_Report.pdf # Comprehensive final technical report
    ├── model_comparison_results.csv # 5-model cross-validation metrics
    ├── bayesian_optimization_trials.csv # All 25 Optuna trial parameters & scores
    ├── bayesian_tuning_comparison.csv # Baseline vs Tuned metrics & latency
    ├── hmm_viterbi_evaluation.csv   # Per-trip HMM test metrics
    ├── dataset_inventory_summary.csv # Telemetry summary for all 90 trips
    ├── candidate_search_metrics.csv  # Spatial radius recall and density metrics
    ├── figures/                     # Publication figures & confusion matrices
    └── maps/                        # Interactive Folium trip maps
```

---

## 5. Two-Laptop Deployment Guide

### Laptop 1 (PathSnap Inference Server)
Laptop 1 hosts the trained models, feature extractor, HMM engine, and evaluator test dataset.

1. Activate the conda environment:
   ```bash
   conda activate ml_hackathon
   ```
2. Start the FastAPI server on all interfaces (port 8000):
   ```bash
   python main.py --host 0.0.0.0 --port 8000
   ```
3. Identify Laptop 1's local IP address using `ipconfig` (e.g. `192.168.1.105`).

### Laptop 2 (Evaluator Client Machine)
Laptop 2 is used by the evaluator to run assessments and view trajectory reconstructions:
1. Open the UI by either:
   - Browsing to `http://<LAPTOP_1_IP>:8000/ui/index.html`, OR
   - Opening `ui/index.html` locally in any modern web browser.
2. Enter Laptop 1's Server URL (e.g. `http://192.168.1.105:8000`) and click **"Connect"**.
3. Select any test trip from the dropdown (trips from `evaluator_test/` or `student_test/`).
4. Click **"Run Trajectory Match"**. The UI displays:
   - Real-time KPI summary cards (Accuracy, Precision, Recall, Topological Continuity %, Latency).
   - High-resolution interactive Leaflet map comparing raw GPS fixes, ground truth, and HMM reconstructed path.

---

## 6. Full Pipeline Reproduction

To reproduce all results from scratch:

```bash
# 1. Feature extraction from raw GPS data and road graphs
python scripts/extract_features.py

# 2. Benchmark 5 machine learning model families (5-fold GroupKFold)
python scripts/train_compare_models.py

# 3. Bayesian hyperparameter optimization with Optuna
python scripts/tune_model_bayes.py

# 4. Hidden Markov Model & Viterbi sequential evaluation
python scripts/run_hmm_evaluation.py

# 5. Compile final technical report PDF
python scripts/generate_technical_report.py
```

---

## 7. Compliance & Integrity Guarantees
- **No Data Leakage**: All cross-validation splitting strictly uses `GroupKFold` grouped by `trip_id`. No candidate or fix from a training trip appears in validation folds.
- **Evaluator Ground-Truth Isolation**: The `evaluator_test/` ground truth is strictly sequestered on Laptop 1. The FastAPI endpoint evaluates metrics server-side and never passes raw labels to the client.
- **Geodesic Accuracy**: All spatial distances are calculated using Haversine formulations ($R = 6,371,000$ m) rather than Euclidean degree approximations.
- **Basemap Reliability**: Map visualizations use open Esri WorldStreetMap tiles to avoid OSM 403 access blocks under `file:///` protocols.
