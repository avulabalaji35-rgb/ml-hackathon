# PathSnap: Team Workload Division & Module Ownership
**PES University — UE24CS352A: Machine Learning Hackathon**

This document establishes the official workload division, technical ownership, and presentation talking points for the four team members (`035`, `056`, `051`, and `810`).

---

## 1. Executive Summary of Member Ownership

| Member ID | Role / Specialization | Core Modules Owned | Primary Code & Artifacts | Primary Presentation Focus |
| :---: | :--- | :--- | :--- | :--- |
| **035** | **Lead System Architect & Spatial Feature Engineering** | • Dataset Ingestion & Catalogs<br>• Haversine Geodesic Engine<br>• Spatial STRtree Indexing<br>• 14-D Feature Extractor | `src/features/extractor.py`<br>`src/data/loader.py`<br>`scripts/extract_features.py`<br>`DATASET.md` | Candidate search radius trade-offs, spatial tree indexing, and geometric feature formulation. |
| **056** | **Machine Learning & Bayesian Optimization Lead** | • Multi-Model Benchmark (5 families)<br>• GroupKFold Cross-Validation<br>• Optuna TPE Bayesian Tuning<br>• Model Serialization | `scripts/train_compare_models.py`<br>`scripts/tune_model_bayes.py`<br>`models/tuned_model.joblib`<br>`reports/model_comparison_results.csv` | Cross-validation design, group leakage prevention, model trade-offs, and 60.4% latency reduction. |
| **051** | **Sequential Probabilistic Modeling & HMM Engine** | • Map-Matching HMM Formulation<br>• Calibrated Emission Probabilities<br>• Shortest-Path Transition Matrix<br>• Dynamic Viterbi Decoder | `src/hmm/decoder.py`<br>`scripts/run_hmm_evaluation.py`<br>`reports/hmm_viterbi_evaluation.csv`<br>`reports/figures/hmm_vs_*.png` | Viterbi algorithmic complexity, transition penalty math, and lifting route continuity from 77.15% to 99.80%. |
| **810** | **Distributed Architecture, UI & Technical Reporting** | • Two-Laptop Network Architecture<br>• FastAPI High-Throughput Server<br>• Evaluator-Facing Leaflet UI<br>• ReportLab Technical Report PDF | `src/api/server.py`<br>`main.py`<br>`ui/` (index.html, style.css, app.js)<br>`reports/PathSnap_Technical_Report.pdf` | Zero-leakage ground truth quarantine, client-server latency, modern Web UI, and report compilation. |

---

## 2. Detailed Member Workload & Technical Specifications

```
                       End-to-End System Integration Flow
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Member 035     │      │   Member 056     │      │   Member 051     │      │   Member 810     │
│ Spatial Pipeline │ ───► │ Machine Learning │ ───► │   HMM Viterbi    │ ───► │ Distributed API  │
│  & STRtree Gen   │      │ & Bayesian Opt   │      │  Reconstruction  │      │  & Evaluator UI  │
└──────────────────┘      └──────────────────┘      └──────────────────┘      └──────────────────┘
```

---

### Member 035: Lead System Architect & Spatial Feature Engineering

#### 1. Component Overview & Objectives
Member 035 is responsible for ingesting raw trajectory coordinates, computing precise geodesic spatial metrics, constructing the directed road network graph, and engineering the multi-dimensional feature space fed into the machine learning models.

#### 2. Technical Methodology & Math
- **Geodesic Formulations**: Implemented the spherical Haversine distance with Earth radius $R = 6,371,000$ m:
  $$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
  Strictly avoided planar Euclidean approximations in degree coordinates.
- **Spatial Indexing & Candidate Search**: Built vectorized Shapely 2.0 `STRtree` spatial indices over all road segments. Conducted a sweep across $R \in [10\text{m}, 100\text{m}]$, establishing that **$R = 35\text{m}$** is the optimal operating threshold ($>98.5\%$ candidate recall with ~8 candidates per fix, reducing candidate search time from 30s to <2s per trip).
- **14 Engineered Features**:
  1. `perp_dist_m`: Orthogonal distance from GPS fix to candidate road arc.
  2. `proj_offset_ratio`: Normalized longitudinal position along the arc $[0, 1]$.
  3. `heading_divergence_deg`: Angular delta between vehicle instantaneous heading and arc orientation.
  4. `velocity_kmh`: Instantaneous vehicle speed computed across consecutive GPS points.
  5. `speed_limit_divergence`: Normalized delta between vehicle speed and road segment maximum allowed speed.
  6. `arc_length_m`: Physical length of the candidate road arc.
  7. `oneway`: Boolean road directionality constraint.
  8. `start_node_dist_m` & `end_node_dist_m`: Proximity to terminal intersection junctions.
  9. `is_candidate_rank`: Rank of the candidate sorted by perpendicular distance.
  10. `delta_t_sec`: Time delta from preceding observation.
  11. `dist_to_prev_cand`: Spatial offset from the previous GPS fix's closest candidate.
  12. `bearing_sin` & `bearing_cos`: Trigonometric representations of road bearing to avoid angular wrap-around at $360^\circ$.

#### 3. Key Files & Artifacts Owned
- `src/data/loader.py`: Spatial loader, Haversine equations, and geometry sanitization.
- `src/features/extractor.py`: High-performance batch candidate generator and feature extractor.
- `scripts/extract_features.py`: Batch preprocessing CLI for train and test partitions.
- `data/train_features.csv` (28,779 candidate samples) & `data/test_features.csv` (8,852 candidate samples).
- `DATASET.md`: Official dataset datasheet, catalog, and schema specifications.

#### 4. Presentation & Defense Script (What Member 035 Should Say)
> *"I built the spatial foundation and feature engineering pipeline for PathSnap. Raw GPS data contains severe multipath noise, with vehicles appearing to leap across parallel side streets. By implementing batch STRtree spatial indexing, we reduced candidate extraction from 30 seconds per trip down to under 2 seconds. I proved that a search radius of 35 meters captures over 98.5% of true road segments while keeping candidate sets lean (~8 candidates per point). I engineered 14 spatial and kinematic features—including perpendicular offset, road bearing divergence, and speed-limit consistency—which provide the discriminative signals needed by our machine learning models."*

---

### Member 056: Machine Learning Modeling & Bayesian Optimization Lead

#### 1. Component Overview & Objectives
Member 056 owns the predictive machine learning engine, benchmarking 5 distinct algorithm families under rigorous cross-validation and performing Bayesian hyperparameter optimization to minimize inference latency while maintaining peak classification accuracy.

#### 2. Technical Methodology & Math
- **Leakage-Free Validation Strategy**: Formulated a 5-Fold `GroupKFold` split grouped strictly by `trip_id`. Because adjacent GPS fixes in the same trip share kinematic continuity, random k-fold splitting causes severe validation leakage. GroupKFold guarantees that no trip seen during training ever appears in the validation fold.
- **Multi-Model Benchmark (5 Algorithm Families)**:
  1. *Linear / Margin Family*: Logistic Regression ($L_2$ regularized, scaled features).
  2. *Single Tree Family*: Decision Tree Classifier (Gini impurity, max depth constrained).
  3. *Bagging Ensemble*: Random Forest (100 estimators, bootstrap aggregation).
  4. *Boosting Ensemble*: Histogram Gradient Boosting Classifier (binned splits, native NaN handling).
  5. *Deep Learning / Neural Family*: Multi-Layer Perceptron (MLP, 2 hidden layers $[64, 32]$, ReLU, Adam optimizer).
- **Model Selection Criteria**: HistGradientBoosting achieved the highest Macro F1 ($0.9355 \pm 0.0198$), lowest multi-candidate log loss ($0.0983$), and fast inference ($52.16$ ms).
- **Optuna Bayesian Optimization (TPE Engine)**:
  Executed 25 trials optimizing the objective function $\mathcal{L}_{CV} = 1 - \text{Macro F1}$.
  - *Tuned Parameters*: `learning_rate: 0.1513`, `max_iter: 180`, `max_depth: 6`, `min_samples_leaf: 35`, `l2_regularization: 0.8050`.
  - *Outcome*: Cut inference latency from **80.55 ms to 31.91 ms per trip (60.4% speedup)** with peak Macro F1 of **0.9356**.

#### 3. Key Files & Artifacts Owned
- `scripts/train_compare_models.py`: 5-model benchmark runner with GroupKFold evaluation.
- `scripts/tune_model_bayes.py`: Optuna TPE Bayesian hyperparameter optimization script.
- `models/tuned_model.joblib`: Serialized production-ready model binary.
- `models/`: Serialized models for all benchmarked families.
- `reports/model_comparison_results.csv` & `reports/bayesian_optimization_trials.csv`.
- `reports/figures/model_comparison_*.png` & `reports/figures/bayes_opt_*.png`.

#### 4. Presentation & Defense Script (What Member 056 Should Say)
> *"I designed the machine learning and hyperparameter tuning pipeline. To prevent data leakage, I enforced a strict 5-Fold GroupKFold cross-validation strategy grouped by trip_id, ensuring no temporal or spatial overlap between training and validation folds. I benchmarked five distinct algorithm families: Logistic Regression, Decision Trees, Random Forest, HistGradientBoosting, and an MLP Neural Network. HistGradientBoosting delivered the lowest log loss of 0.0983 and a top-tier Macro F1 of 0.9355. I then ran 25 trials of Bayesian optimization using Optuna's Tree-structured Parzen Estimator (TPE). This pruned redundant trees, reducing inference latency by 60.4% down to just 31.91 milliseconds per trip while preserving state-of-the-art accuracy."*

---

### Member 051: Probabilistic Sequential Modeling & HMM/Viterbi Engine

#### 1. Component Overview & Objectives
Member 051 is responsible for the sequential route reconstruction engine. While a classifier predicts candidate probabilities for individual GPS points, it lacks knowledge of topological road network connectivity, leading to disconnected routes. Member 051 formulated and implemented the Hidden Markov Model (HMM) and Viterbi decoder.

#### 2. Technical Methodology & Math
- **Hidden Markov Model Formulation**:
  - *Hidden States $S_t$*: The true candidate road segment $r_t^i$ traversed at time step $t$.
  - *Observations $O_t$*: The observed noisy GPS coordinates $(lat_t, lon_t)$.
- **Emission Probability $P(O_t \mid S_t = r_t^i)$**:
  Derived by applying softmax probability calibration to the raw margin outputs of Member 056's tuned boosting model, augmented with Gaussian distance decay:
  $$P(O_t \mid r_t^i) = P_{model}(r_t^i) \cdot \frac{1}{\sqrt{2\pi}\sigma_z} \exp\left( -\frac{d(O_t, r_t^i)^2}{2\sigma_z^2} \right)$$
- **Transition Probability $P(S_t = r_t^j \mid S_{t-1} = r_{t-1}^i)$**:
  Calculated via Dijkstra shortest-path network routing on the directed NetworkX graph:
  $$P(r_t^j \mid r_{t-1}^i) = \exp\left( -\frac{|d_{graph}(r_{t-1}^i, r_t^j) - d_{euclid}(O_{t-1}, O_t)|}{\beta} \right) \cdot \exp\left(-\frac{\Delta t}{\tau}\right)$$
  Where $\beta$ penalizes detours and $\exp(-\Delta t / \tau)$ scales transition confidence during signal dropouts.
- **Logarithmic Viterbi Decoding**:
  Operated entirely in negative log-space to guarantee zero numerical underflow:
  $$V(t, j) = \max_i \left[ V(t-1, i) + \ln P(r_t^j \mid r_{t-1}^i) \right] + \ln P(O_t \mid r_t^j)$$
- **Topological Route Continuity**:
  Lifts trajectory continuity from **77.15%** (classifier-only) to **99.80%** (HMM + path reconstruction), achieving **94.75% Arc Precision and Recall**.

#### 3. Key Files & Artifacts Owned
- `src/hmm/decoder.py`: `MapMatchingHMM` class, transition graph builder, and Viterbi solver.
- `scripts/run_hmm_evaluation.py`: Full trajectory sequential evaluation and metric logging.
- `reports/hmm_viterbi_evaluation.csv`: Detailed trip-by-trip HMM performance log.
- `reports/figures/hmm_vs_classifier_continuity.png` & `reports/figures/hmm_vs_classifier_accuracy.png`.

#### 4. Presentation & Defense Script (What Member 051 Should Say)
> *"I developed the sequential probabilistic reasoning engine using Hidden Markov Models and Viterbi decoding. Point-wise classifiers often make isolated errors—for instance, jumping to an overpass or parallel alley—which breaks route continuity down to 77.15%. My HMM engine integrates the machine learning classifier's emission probabilities with topological graph transition probabilities. Using NetworkX shortest-path calculations and temporal gap penalties, my Viterbi decoder identifies the globally optimal, physically traversable path through the road network. This lifted route continuity from 77.15% to 99.80%, ensuring reconstructed trajectories are 100% driveable."*

---

### Member 810: Distributed System Architecture, Evaluator UI & Technical Report

#### 1. Component Overview & Objectives
Member 810 owns the deployment architecture, real-time evaluation services, client dashboard, and formal university deliverables. Member 810 engineered the two-laptop distributed architecture and compiled the final technical report.

#### 2. Technical Methodology & Math
- **Distributed Two-Laptop Topology**:
  - *Laptop 1 (PathSnap Inference Server)*: Hosts FastAPI, trained model binaries, HMM engine, and the sequestered evaluator test dataset (`evaluator_test/`).
  - *Laptop 2 (Evaluator Client)*: Lightweight web client communicating via asynchronous JSON REST endpoints.
- **Zero-Leakage Ground Truth Quarantine**:
  The evaluator dataset is strictly quarantined on Laptop 1 and ignored by `.gitignore`. The client requests evaluation via `POST /api/evaluate/{trip_id}`. Laptop 1 evaluates the metrics entirely server-side and returns only summary statistics and coordinate polylines, completely shielding raw ground truth files from Laptop 2.
- **Evaluator Web Dashboard (`ui/`)**:
  - Implemented responsive, modern UI in pure HTML5/CSS3/JavaScript (no bulky npm build dependencies).
  - Built-in Leaflet.js map with dual layers: raw noisy GPS fixes (red markers), ground truth path (blue), and HMM reconstructed route (green polyline).
  - Auto-origin resolution: automatically detects whether it was loaded via localhost or remote network IP (`http://10.1.25.93:8000`), zero configuration required.
- **High-Throughput FastAPI Backend**:
  Asynchronous endpoints with CORS middleware, streaming Folium HTML maps, and sub-100ms end-to-end response times.
- **Consolidated Technical Report (ReportLab PDF)**:
  Authored programmatic compilation script generating `reports/PathSnap_Technical_Report.pdf` (1.09 MB), strictly adhering to university deliverables (Sections 2 through 7).

#### 3. Key Files & Artifacts Owned
- `src/api/server.py` & `main.py`: FastAPI server application and CLI entrypoint.
- `ui/index.html`, `ui/style.css`, `ui/app.js`: Modern Evaluator UI dashboard.
- `scripts/generate_technical_report.py`: Programmatic ReportLab PDF generation engine.
- `reports/PathSnap_Technical_Report.pdf`: 1.09 MB final technical report.
- `src/visualization/folium_maps.py`: Esri WorldStreetMap interactive map generator.

#### 4. Presentation & Defense Script (What Member 810 Should Say)
> *"I designed the two-laptop distributed architecture, the evaluator web interface, and the technical report. To comply with the university's data integrity rules, I ensured the evaluator test data is physically isolated on Laptop 1. Laptop 2 acts strictly as a thin client running our Leaflet dashboard. When the evaluator triggers a test run, Laptop 1's FastAPI server processes the trajectory and returns the evaluated metrics and coordinate geometry in under 70 milliseconds without leaking raw ground truth data. I also wrote the ReportLab script that compiles our 1.09 MB technical report, fully documenting our mathematical formulations, experimental tables, and architectural diagrams."*

---

## 3. Cross-Member Collaboration & Dependency Matrix

The table below demonstrates how the four members' codebases interface seamlessly:

| Producer Member | Hand-off Artifact / API | Consumer Member | Functional Interface |
| :---: | :---: | :---: | :--- |
| **035** | `data/train_features.csv` | **056** | 14-D feature matrix ingested by 5-fold `GroupKFold` model training. |
| **035** | Road Graph `G` & Candidate STRtree | **051** | Graph topology and arc candidates used for HMM emission/transition states. |
| **056** | `models/tuned_model.joblib` | **051** | Calibrated boosting predictions used as HMM observation emission probabilities. |
| **051** | `MapMatchingHMM.viterbi_decode()` | **810** | Reconstructed route sequence wrapped into FastAPI `/api/evaluate/{trip_id}`. |
| **810** | Evaluator UI Dashboard & API | **All / Evaluator** | Real-time visual validation of the entire team's integrated pipeline. |

---

## 4. Viva / Evaluation Q&A Cheat Sheet

| Likely Evaluator Question | Recommended Speaker | Key Response Points |
| :--- | :---: | :--- |
| *"How did you prevent data leakage during feature engineering and training?"* | **035 & 056** | Features are computed per trip or strictly within candidate spatial neighborhoods. Cross-validation uses `GroupKFold` grouped by `trip_id`, ensuring no fix from a training trip ever enters the validation set. |
| *"Why did you select HistGradientBoosting over Random Forest or Neural Networks?"* | **056** | HistGradientBoosting achieved the highest CV Macro F1 (0.9355) and lowest log loss (0.0983), but with an inference latency of just 31.91 ms (vs 252 ms for Random Forest), which is essential for real-time map matching. |
| *"Why is an ML classifier alone not enough for map-matching?"* | **051** | Point-wise classifiers predict each fix independently and cannot enforce physical driveability. They frequently jump across parallel streets or one-way roads. The HMM Viterbi decoder enforces topological continuity, raising continuous path accuracy from 77.15% to 99.80%. |
| *"How does your system guarantee ground-truth integrity on Laptop 2?"* | **810** | The evaluator test partition (`evaluator_test/`) resides exclusively on Laptop 1 and is blocked by `.gitignore`. The client on Laptop 2 communicates solely via REST API; Laptop 1 evaluates performance server-side and returns only KPI metrics and polyline coordinates. |
| *"What happens if there is a long GPS signal dropout (e.g. tunnel)?"* | **051 & 035** | The HMM applies Dijkstra shortest-path routing over the road network between the last known fix and the reappearance fix, while an exponential time decay $\exp(-\Delta t / \tau)$ scales the transition penalty to bridge the gap seamlessly. |
