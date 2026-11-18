# PathSnap Dataset Specification & Datasheet
**PES University — UE24CS352A: Machine Learning Hackathon**  
*Challenge: Distributed Map-Matching & Route Reconstruction*

---

## 1. Dataset Overview

The **PathSnap** dataset contains high-resolution GPS trajectory observations synchronized with local road-network subgraphs. Trajectories record vehicular travel across varied urban, suburban, and arterial road networks.

Due to urban obstacles, buildings, electromagnetic interference, and receiver inaccuracies, raw GPS signals exhibit:
- **Spatial Measurement Noise:** Lateral displacement from the true road centerline (Gaussian and non-Gaussian deviations).
- **Satellite Multipath & Teleportation Jumps:** Spurious instantaneous velocity spikes exceeding physical vehicular capabilities (>140 km/h).
- **GPS Signal Dropouts:** Temporal gaps ($\Delta t > 1.0$s) caused by tunnels, underpasses, urban canyons, or device polling delays.

The objective is to match these noisy, sequential GPS observations onto a directed road network and accurately reconstruct the complete ground-truth traversed route.

---

## 2. Dataset Partitions & Leakage Prevention Rules

The complete dataset is partitioned into three mutually exclusive subsets:

| Partition | Number of Trips | Purpose | Ground Truth Route Supplied |
| :--- | :---: | :--- | :---: |
| **Training** | **80 trips** | Development, exploratory analysis, candidate generation, feature engineering, model training, and group-aware cross-validation. | Yes (`.route`) |
| **Student Testing** | **10 trips** | Development-stage model comparison, sequence decoding evaluation, and hyperparameter validation. | Yes (`.route`) |
| **Evaluator Testing** | **10 trips** | Final blind evaluation on Laptop 2 via the FastAPI server running on Laptop 1. | Held-out on Server |

### Strict Integrity Rules (from Hackathon Guidelines)
- **Zero Cross-Partition Contamination:** No GPS observations or graph elements from the same trip may appear across multiple partitions.
- **Strict Anti-Leakage Protocol:** Evaluator testing data must never be used for feature selection, model fitting, algorithm selection, or threshold tuning.
- **Model Training Scope:** Preprocessing transformations and candidate generation statistics must be fit using permitted training data only.

---

## 3. Raw File Formats & Schemas

Each trip is stored in an independent folder named with an 8-digit zero-padded ID (e.g., `00000001`, `00000005`, `00000040`). Each trip folder contains exactly four raw whitespace-separated text files:

### 3.1. Trajectory Observations: `<trip_id>.track`
Whitespace-separated, headerless file containing chronological GPS fixes:

| Column Index | Field | Data Type | Physical Meaning | Example Value |
| :---: | :--- | :---: | :--- | :---: |
| `0` | `longitude` | `float64` | Angular longitude coordinate (WGS84, EPSG:4326) | `8.595203` |
| `1` | `latitude` | `float64` | Angular latitude coordinate (WGS84, EPSG:4326) | `49.844053` |
| `2` | `timestamp` | `float64` | Elapsed time since journey start (seconds) | `1.000000` |

- **Nominal Sampling Rate:** 1.0 Hz (1 observation per second). Across the entire dataset, **94.32%** of successive fixes have $\Delta t = 1.00$s.

### 3.2. Road Network Nodes: `<trip_id>.nodes`
Whitespace-separated, headerless file containing vertices of the local road network:

| Column Index | Field | Data Type | Physical Meaning | Example Value |
| :---: | :--- | :---: | :--- | :---: |
| `0` | `longitude` | `float64` | Vertex longitude coordinate | `8.581679` |
| `1` | `latitude` | `float64` | Vertex latitude coordinate | `49.857880` |

- **Node Indexing:** The **0-based line number (row index)** serves as the canonical `node_id`.

### 3.3. Directed Road Segments: `<trip_id>.arcs`
Whitespace-separated, headerless file defining directed connections between nodes:

| Column Index | Field | Data Type | Physical Meaning | Example Value |
| :---: | :--- | :---: | :--- | :---: |
| `0` | `from_node_id` | `int64` | Source node ID (references row index of `.nodes`) | `0` |
| `1` | `to_node_id` | `int64` | Target node ID (references row index of `.nodes`) | `281` |

- **Arc Indexing:** The **0-based line number (row index)** serves as the canonical `arc_id`.
- **Directionality:** Arcs represent directed road segments. Two-way streets appear as two separate reciprocal arcs ($u \to v$ and $v \to u$).

### 3.4. Ground-Truth Traversal Route: `<trip_id>.route`
Single-column integer file containing the chronological sequence of traversed road segments:

| Column Index | Field | Data Type | Physical Meaning | Example Value |
| :---: | :--- | :---: | :--- | :---: |
| `0` | `arc_id` | `int64` | Ground-truth road segment traversed by vehicle | `2641` |

- **Sequence Meaning:** The ordered sequence of physical road segments traversed by the vehicle.
- **Topological Adjacency:** Across the training partition, **99.93%** of consecutive arcs satisfy physical connectivity ($v_k = u_{k+1}$).

---

## 4. Coordinate Reference System & Metric Transformations

Because geographic coordinates $(\lambda, \phi)$ are angular measurements on an oblate spheroid, **ordinary Euclidean distance in degrees is invalid**:
$$\Delta_{\text{deg}} = \sqrt{(\Delta \lambda)^2 + (\Delta \phi)^2} \quad \text{[INCORRECT for metric operations]}$$

### 4.1. Geodesic Haversine Formulation
To calculate real-world physical distances in meters between coordinate pairs $(\phi_1, \lambda_1)$ and $(\phi_2, \lambda_2)$, we use the Haversine equation with mean Earth radius $R = 6,371,000 \text{ m}$:
$$a = \sin^2\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2 - \lambda_1}{2}\right)$$
$$d = 2 R \arcsin(\sqrt{a})$$

### 4.2. Local Metric Projection for Point-to-Segment Queries
For high-speed candidate arc queries, each trip is projected locally relative to its centroid $(\phi_0, \lambda_0)$:
$$x = R \cdot (\lambda - \lambda_0) \cdot \cos(\phi_0) \cdot \frac{\pi}{180}$$
$$y = R \cdot (\phi - \phi_0) \cdot \frac{\pi}{180}$$

The perpendicular distance from GPS point $P = (x_p, y_p)$ to arc segment $AB = [(x_a, y_a), (x_b, y_b)]$ is computed by clamping the projection parameter $t \in [0, 1]$:
$$t = \max\left(0, \min\left(1, \frac{(P - A) \cdot (B - A)}{\|B - A\|^2}\right)\right)$$
$$\text{dist}(P, AB) = \|P - (A + t(B - A))\|$$

---

## 5. Dataset Summary Statistics (Empirical Findings)

The table below summarizes dataset characteristics obtained from our exhaustive EDA across all 90 trips:

| Dimension / Metric | Measured Value | Standard Deviation / Notes |
| :--- | :---: | :--- |
| **Total Trips Analyzed** | **90** | 80 Training + 10 Student-Testing |
| **Total GPS Observations** | **227,577** | Average 2,528.6 observations per trip |
| **Average Trip Duration** | **5,274.2 s** | $\approx 87.9$ minutes per trip |
| **Nominal 1 Hz Adherence** | **94.32%** | Fraction of intervals where $\Delta t = 1.00$s |
| **Trips with Dropout Gaps ($\Delta t > 2$s)** | **54 / 90 (60.0%)** | Maximum gap recorded: 312.0 s |
| **Average Trajectory Distance** | **36.39 km** | Cumulative Haversine path length |
| **Average Speed** | **54.78 km/h** | Plausible urban/arterial driving speeds |
| **Maximum Recorded Speed (Outlier)** | **648.50 km/h** | Satellite multipath teleportation jump |
| **Total Missing / NaN Values** | **0** | Zero corrupt or empty records |
| **Average Road Network Nodes per Trip** | **78,394** | Min: 898 (Trip 00000005), Max: 184,210 |
| **Average Road Network Arcs per Trip** | **147,812** | Min: 1,579 (Trip 00000005), Max: 346,899 |
| **Median Arc Length** | **45.2 m** | Inter-quartile range: 22.4m – 84.1m |
| **Ground-Truth Route Continuity** | **99.93%** | Consecutive traversed arcs connect topologically |

---

## 6. Spatial Candidate Generation Analysis

Using an R-tree spatial index (`shapely.strtree.STRtree`), we analyzed the trade-off between search radius $R$, average candidate count ($k$), and **Candidate Miss Rate** (percentage of GPS observations where the true road segment is absent from the candidate set):

| Search Radius $R$ (meters) | Average Candidates ($k$) | Candidate Miss Rate (%) | Modeling Evaluation |
| :---: | :---: | :---: | :--- |
| **10 m** | 2.98 | 7.73% | High miss rate; unsuitable for urban GPS scatter. |
| **25 m** | 6.07 | 1.89% | Good balance; miss rate drops by 75%. |
| **35 m (Recommended)** | **8.42** | **1.45%** | **Optimal trade-off; $>98.5\%$ recall with lightweight trellis.** |
| **50 m** | 13.35 | 1.24% | High recall, but candidate density increases by 60%. |
| **75 m** | 22.73 | 1.04% | Marginal recall gain with significant computational overhead. |
| **100 m** | 34.52 | 0.93% | Excess candidate noise; increases Viterbi complexity ($O(K^2 T)$). |

---

## 7. Programmatic Loading & Usage

### 7.1. Quick Start using `PathSnapDataset`
```python
from src.data.dataset import PathSnapDataset

# 1. Initialize dataset for training partition
dataset = PathSnapDataset(base_dir=".", partition="train")
print(f"Total training trips: {len(dataset)}")

# 2. Access a single trip
trip = dataset[0] # Returns TripData instance
print(f"Loaded Trip ID: {trip.trip_id}")
print(f"  GPS Fixes: {len(trip.track):,}")
print(f"  Road Nodes: {len(trip.nodes):,}")
print(f"  Road Arcs: {len(trip.arcs):,}")
print(f"  Route Length: {len(trip.route):,} arcs")

# 3. Access NetworkX directed graph
G = trip.graph
print(f"  Graph edges: {G.number_of_edges():,}")
```

### 7.2. Inspecting Sample Trip Data
A complete self-contained sample trip (`00000005`, 64 KB) is included in `data/sample_data/00000005/` for instant execution without requiring full dataset downloads:
```python
sample_dataset = PathSnapDataset(base_dir=".", partition="sample")
sample_trip = sample_dataset[0]
print("Sample Trip:", sample_trip.trip_id, "Track points:", len(sample_trip.track))
```

---

## 8. Catalog and Manifest Files
- **Tabular Catalog:** [`data/dataset_catalog.csv`](data/dataset_catalog.csv) contains per-trip metrics (duration, distance, speeds, bounds, node/arc counts).
- **JSON Manifest:** [`data/manifest.json`](data/manifest.json) contains machine-readable partition manifests and schema definitions.
