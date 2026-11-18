# PathSnap Dataset Directory

This directory houses the dataset catalog, metadata manifest, and sample data for the **PathSnap: Distributed Map-Matching & Route Reconstruction** challenge.

---

## 1. Directory Layout
```
data/
├── README.md              # Dataset directory guide
├── dataset_catalog.csv    # Full metadata catalog for all 90 trips
├── manifest.json          # Machine-readable JSON metadata & specifications
└── sample_data/           # Working sample data for immediate tests
    └── 00000005/          # Self-contained sample trip (train partition)
        ├── 00000005.track # GPS trajectory observations (1,045 fixes)
        ├── 00000005.nodes # Road network nodes (898 vertices)
        ├── 00000005.arcs  # Directed road segments (1,579 edges)
        └── 00000005.route # Traversed reference ground-truth route (105 arcs)
```

---

## 2. Using Full Hackathon Data
The complete dataset (80 training trips and 10 student-testing trips) should be placed in either of the following paths:
- `data/train/` and `data/student_test/`
- Or the original `Student folder/Student folder/train/` and `student_test/`

The built-in loader (`src.data.loader.discover_trips` and `src.data.dataset.PathSnapDataset`) automatically detects both locations seamlessly.

---

## 3. Data File Formats
1. **`.track`**: Longitude, Latitude, Timestamp in seconds (whitespace-separated, no headers). Nominally 1 Hz.
2. **`.nodes`**: Longitude, Latitude (whitespace-separated, no headers). The 0-based row index represents the `node_id`.
3. **`.arcs`**: `from_node_id`, `to_node_id` (whitespace-separated, no headers). The 0-based row index represents the `arc_id`.
4. **`.route`**: `arc_id` sequence (1 integer column, no headers).

For comprehensive schema, statistics, and modeling guidelines, see [DATASET.md](../DATASET.md).
