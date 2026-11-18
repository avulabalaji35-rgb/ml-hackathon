"""
Script to analyze spatial proximity and candidate generation characteristics:
- Perpendicular distance from GPS observations to ground-truth road segments
- Candidate count distribution across radii: 10m, 25m, 50m, 75m, 100m
- Candidate Miss Rate (true arc not in candidate set) across radii
"""

import os
import sys
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree

sys.path.append(os.path.abspath("."))
from src.data.loader import discover_trips, load_trip, point_to_segment_distance, haversine_distance


def evaluate_candidate_generation(trip, radii=[10, 25, 50, 75, 100]):
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    if route is None or len(route) == 0:
        return None

    # Sample GPS observations (e.g. 200 points for speed if trajectory is large)
    step = max(1, len(track) // 250)
    sampled_indices = range(0, len(track), step)

    # Build spatial index using local metric coordinates
    # Anchor at trip center
    center_lat = track['latitude'].mean()
    center_lon = track['longitude'].mean()
    cos_lat = np.cos(np.radians(center_lat))
    R = 6371000.0

    def to_xy(lon, lat):
        x = np.radians(lon - center_lon) * R * cos_lat
        y = np.radians(lat - center_lat) * R
        return x, y

    # Precompute line strings for all arcs
    lines = []
    arc_ids = []
    
    # Filter arcs roughly within bounding box of track (+ 200m buffer)
    lat_min, lat_max = track['latitude'].min() - 0.003, track['latitude'].max() + 0.003
    lon_min, lon_max = track['longitude'].min() - 0.003, track['longitude'].max() + 0.003
    
    for arc_id, row in arcs.iterrows():
        u = row['from_node_id']
        v = row['to_node_id']
        if u in nodes.index and v in nodes.index:
            u_lat, u_lon = nodes.loc[u, 'latitude'], nodes.loc[u, 'longitude']
            v_lat, v_lon = nodes.loc[v, 'latitude'], nodes.loc[v, 'longitude']
            if (lat_min <= u_lat <= lat_max or lat_min <= v_lat <= lat_max) and \
               (lon_min <= u_lon <= lon_max or lon_min <= v_lon <= lon_max):
                ux, uy = to_xy(u_lon, u_lat)
                vx, vy = to_xy(v_lon, v_lat)
                if ux != vx or uy != vy:
                    lines.append(LineString([(ux, uy), (vx, vy)]))
                    arc_ids.append(arc_id)

    if not lines:
        return None

    tree = STRtree(lines)
    arc_ids = np.array(arc_ids)

    # For ground truth matching evaluation:
    # A GPS point t roughly corresponds to the closest ground truth route arc
    # Let's compute distance to the true route arcs
    route_arc_ids = set(route['arc_id'].values)

    results_per_radius = {r: {'count': [], 'contains_gt': []} for r in radii}
    min_dist_to_any_arc = []
    min_dist_to_gt_arc = []

    for idx in sampled_indices:
        px, py = to_xy(track.loc[idx, 'longitude'], track.loc[idx, 'latitude'])
        pt = Point(px, py)

        # Query candidates for each radius
        for r in radii:
            buffer_circle = pt.buffer(r)
            candidates_idx = tree.query(buffer_circle)
            cand_arcs = set(arc_ids[candidates_idx])
            results_per_radius[r]['count'].append(len(cand_arcs))
            # True arc in candidate set?
            has_gt = len(cand_arcs.intersection(route_arc_ids)) > 0
            results_per_radius[r]['contains_gt'].append(has_gt)

    summary = {'trip_id': trip.trip_id}
    for r in radii:
        summary[f'avg_cands_{r}m'] = np.mean(results_per_radius[r]['count'])
        summary[f'miss_rate_{r}m'] = 100.0 * (1.0 - np.mean(results_per_radius[r]['contains_gt']))

    return summary


def main():
    trips_info = discover_trips(".")
    train_dir = trips_info['train_path']
    # Sample 10 train trips for comprehensive candidate analysis
    sample_trips = trips_info['train_trips'][:10]
    
    print(f"Evaluating spatial candidate metrics on {len(sample_trips)} sample training trips...")
    records = []
    for t_id in sample_trips:
        trip = load_trip(os.path.join(train_dir, t_id), partition="train")
        res = evaluate_candidate_generation(trip)
        if res:
            records.append(res)

    df_cand = pd.DataFrame(records)
    cand_csv = "reports/candidate_search_metrics.csv"
    df_cand.to_csv(cand_csv, index=False)
    print(f"Candidate search metrics saved to {cand_csv}")
    print("\n=== Average Candidate Statistics Across Search Radii ===")
    for r in [10, 25, 50, 75, 100]:
        print(f"Radius {r:3d}m: Avg Candidates = {df_cand[f'avg_cands_{r}m'].mean():.2f}, Candidate Miss Rate = {df_cand[f'miss_rate_{r}m'].mean():.2f}%")


if __name__ == '__main__':
    main()
