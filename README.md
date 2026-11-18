# PathSnap: Distributed Map-Matching & Route Reconstruction
**PES University — UE24CS352A: Machine Learning Hackathon**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Conda Env](https://img.shields.io/badge/conda%20env-ml__hackathon-green.svg)](https://docs.conda.io/)
[![Folium](https://img.shields.io/badge/visualization-Folium%20Maps-orange.svg)](https://python-visualization.github.io/folium/)

---

## 1. Project Overview
GPS tracking systems inherently produce noisy, uncertain location measurements due to urban canyons, multipath reflections, signal dropouts, and low sampling frequencies. As a consequence, vehicles frequently appear to drift onto parallel streets, jump across city blocks, or show disconnected positions.

**PathSnap** is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to an underlying directed road network.

This repository implements the complete Phase 1 deliverables:
- Comprehensive exploratory data analysis across **80 Training** and **10 Student-Testing** trips (227,577 total GPS observations).
- Geodesic spatial analysis and distance calculations using Haversine formulations.
- Road network topological analysis (NetworkX directed graphs).
- Ground-truth route continuity verification (99.93% continuity).
- Spatial candidate search radius optimization ($R \in [10\text{m}, 100\text{m}]$).
- Interactive multi-layer Folium maps with layer controls and telemetry inspection.
- Saved, fully executed Jupyter Notebook (`notebooks/01_exploratory_data_analysis.ipynb`).

---

## 2. Repository Structure
```
ml-hackathon/
├── DATASET.md                       # Comprehensive dataset specifications and datasheet
├── environment.yml                  # Conda environment definition (ml_hackathon)
├── requirements.txt                 # Pip dependency specifications
├── .gitignore                       # Ignored dataset binaries and build caches
├── README.md                        # Project documentation and reproduction guide
├── data/                            # Dataset catalog, manifest, and sample data
│   ├── README.md                    # Data setup and format guide
│   ├── dataset_catalog.csv          # Metadata for all 90 trips
│   ├── manifest.json                # JSON schema and partition manifests
│   └── sample_data/                 # Self-contained sample trip (00000005)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Fully executed publication-grade EDA notebook
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Data loading, validation, and Haversine calculations
│   │   └── dataset.py               # PathSnapDataset high-level API
│   └── visualization/
│       ├── __init__.py
│       └── folium_maps.py           # Multi-layer Folium interactive map builder
├── scripts/
│   ├── compute_eda_stats.py         # Full dataset inventory and trajectory metrics
│   ├── analyze_candidates.py        # Spatial candidate search radius & miss rate analysis
│   ├── generate_eda_visualizations.py # Matplotlib/Seaborn figures & HTML maps
│   └── build_and_execute_notebook.py# Programmatic notebook builder and execution engine
└── reports/
    ├── dataset_inventory_summary.csv # Metrics for all 90 trips
    ├── candidate_search_metrics.csv  # Candidate density and miss rates per radius
    ├── figures/                      # High-res publication plots (PNG)
    │   ├── eda_01_sampling_interval_dist.png
    │   ├── eda_02_speed_distribution.png
    │   ├── eda_03_road_network_degrees.png
    │   ├── eda_04_arc_length_distribution.png
    │   ├── eda_05_trajectory_distance_duration.png
    │   └── eda_06_candidate_miss_tradeoff.png
    └── maps/                         # Interactive HTML Folium maps
        ├── trip_00000001_interactive_map.html
        └── trip_00000000_interactive_map.html
```

---

## 3. Environment Setup

### Using Conda (Recommended)
Create and activate the dedicated `ml_hackathon` environment:
```bash
conda env create -f environment.yml
conda activate ml_hackathon
```

Or manually create via Conda:
```bash
conda create -n ml_hackathon python=3.10 -y
conda activate ml_hackathon
pip install -r requirements.txt
```

Register the IPython kernel for Jupyter:
```bash
python -m ipykernel install --user --name ml_hackathon --display-name "Python 3.10 (ml_hackathon)"
```

---

## 4. Key EDA Findings & Dataset Characteristics

| Metric / Dimension | Observation | Engineering Decision for Subsequent Phases |
| :--- | :--- | :--- |
| **Dataset Scale** | 90 trips total (80 Train, 10 Student Test), 227,577 GPS observations, 0 missing values. | All trips are complete and valid. No imputation needed for missing fields. |
| **Sampling Rate** | Nominally 1 Hz; **94.32%** of points are exactly 1.0s apart. | Retain nominal step assumptions, but handle dropout gaps explicitly. |
| **GPS Dropouts** | ~60% of trips contain dropouts ($\Delta t > 2$s), with maximum gaps up to 300s+. | Transition probability in HMM must account for elapsed time $\Delta t$ using shortest-path routing. |
| **GPS Multipath Noise** | Speeds $> 140$ km/h detected (instantaneous jumps up to 648 km/h). | Implement velocity threshold filtering to mitigate severe satellite teleportation spikes. |
| **Road Network Topology** | Mean arc length $\approx 45$m. 30.8% through-segments, 24.1% intersections ($\ge 3$ branches). | Directed graphs require directional heading alignment in emission probabilities. |
| **Ground-Truth Route Continuity** | **99.93%** average route continuity ($v_k = u_{k+1}$). | The vehicle physically traverses connected road sequences; topological adjacency is essential. |
| **Candidate Search Radius ($R$)** | At $R=10$m: 7.73% miss rate. At $R=35$m-$50$m: $< 1.5\%$ miss rate with 8-13 candidates. | Fix candidate search radius to **$35\text{m}$**, providing $>98.5\%$ candidate recall with fast inference. |

---

## 5. Visualizations & Maps

Interactive maps are stored in `reports/maps/`:
- Open `reports/maps/trip_00000001_interactive_map.html` in any web browser to explore raw GPS points (with timestamp/coordinate popups), the underlying road network, and the ground-truth traversed path.
- Static diagnostic plots are saved in `reports/figures/`.

---

## 6. How to Reproduce

Run the end-to-end pipeline to compute stats, generate figures/maps, and re-execute the notebook:
```bash
# 1. Compute summary stats
python scripts/compute_eda_stats.py

# 2. Evaluate spatial candidate search trade-offs
python scripts/analyze_candidates.py

# 3. Generate diagnostic figures and Folium maps
python scripts/generate_eda_visualizations.py

# 4. Build and execute Jupyter Notebook
python scripts/build_and_execute_notebook.py
```
