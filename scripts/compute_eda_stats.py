"""
Script to compute comprehensive dataset-wide EDA statistics across
all 80 Training and 10 Student Testing trips in the PathSnap challenge.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("."))
from src.data.loader import discover_trips, load_trip, haversine_distance


def analyze_single_trip(trip):
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    n_pts = len(track)
    duration = track['timestamp'].iloc[-1] - track['timestamp'].iloc[0]

    # Time delta
    dt = np.diff(track['timestamp'].values)
    dt_1s_pct = 100.0 * np.sum(np.isclose(dt, 1.0, atol=0.01)) / len(dt) if len(dt) > 0 else 0.0
    dt_max = np.max(dt) if len(dt) > 0 else 0.0
    gaps_gt_2s = int(np.sum(dt > 2.0))

    # GPS Distances & Speeds
    lats = track['latitude'].values
    lons = track['longitude'].values
    dists = haversine_distance(lats[:-1], lons[:-1], lats[1:], lons[1:])
    total_dist_m = float(np.sum(dists))
    
    # Speeds (avoid div by zero)
    speeds_mps = np.where(dt > 0, dists / dt, 0.0)
    speeds_kmh = speeds_mps * 3.6
    mean_speed_kmh = float(np.mean(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0
    median_speed_kmh = float(np.median(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0
    p95_speed_kmh = float(np.percentile(speeds_kmh, 95)) if len(speeds_kmh) > 0 else 0.0
    max_speed_kmh = float(np.max(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0
    speed_outliers = int(np.sum(speeds_kmh > 150.0))  # Plausibility threshold

    # Missing values
    missing_track = int(track.isna().sum().sum())
    missing_nodes = int(nodes.isna().sum().sum())
    missing_arcs = int(arcs.isna().sum().sum())
    missing_route = int(route.isna().sum().sum()) if route is not None else 0

    # Route Continuity
    route_len = len(route) if route is not None else 0
    route_continuity_pct = 0.0
    if route is not None and len(route) > 1:
        route_arc_ids = route['arc_id'].values
        # Check valid arc ids
        valid_mask = np.isin(route_arc_ids, arcs.index)
        valid_arcs = route_arc_ids[valid_mask]
        if len(valid_arcs) > 1:
            from_nodes = arcs.loc[valid_arcs, 'from_node_id'].values
            to_nodes = arcs.loc[valid_arcs, 'to_node_id'].values
            # Continuous if to_node[i] == from_node[i+1]
            matches = (to_nodes[:-1] == from_nodes[1:])
            route_continuity_pct = 100.0 * np.sum(matches) / len(matches)

    # Road network size
    n_nodes = len(nodes)
    n_arcs = len(arcs)

    return {
        'trip_id': trip.trip_id,
        'partition': trip.partition,
        'num_points': n_pts,
        'duration_sec': duration,
        'dt_1s_pct': dt_1s_pct,
        'dt_max_sec': dt_max,
        'gaps_gt_2s': gaps_gt_2s,
        'total_gps_dist_km': total_dist_m / 1000.0,
        'mean_speed_kmh': mean_speed_kmh,
        'median_speed_kmh': median_speed_kmh,
        'p95_speed_kmh': p95_speed_kmh,
        'max_speed_kmh': max_speed_kmh,
        'speed_outliers_gt_150': speed_outliers,
        'num_nodes': n_nodes,
        'num_arcs': n_arcs,
        'num_route_arcs': route_len,
        'route_continuity_pct': route_continuity_pct,
        'min_lat': lats.min(),
        'max_lat': lats.max(),
        'min_lon': lons.min(),
        'max_lon': lons.max(),
        'center_lat': lats.mean(),
        'center_lon': lons.mean(),
        'missing_values': missing_track + missing_nodes + missing_arcs + missing_route
    }


def main():
    trips_info = discover_trips(".")
    train_dir = trips_info['train_path']
    test_dir = trips_info['test_path']

    results = []
    print(f"Analyzing {len(trips_info['train_trips'])} train trips...")
    for t_id in trips_info['train_trips']:
        t_path = os.path.join(train_dir, t_id)
        trip = load_trip(t_path, partition="train")
        results.append(analyze_single_trip(trip))

    print(f"Analyzing {len(trips_info['test_trips'])} student_test trips...")
    for t_id in trips_info['test_trips']:
        t_path = os.path.join(test_dir, t_id)
        trip = load_trip(t_path, partition="student_test")
        results.append(analyze_single_trip(trip))

    df = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    csv_path = "reports/dataset_inventory_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary saved to {csv_path}")

    # Display high-level metrics
    print("\n=== Dataset Aggregate Summary ===")
    print(f"Total Trips: {len(df)}")
    print(f"  Train Trips: {len(df[df['partition'] == 'train'])}")
    print(f"  Test Trips:  {len(df[df['partition'] == 'student_test'])}")
    print(f"Total GPS Observations: {df['num_points'].sum():,}")
    print(f"Average Duration per Trip: {df['duration_sec'].mean():.1f} s (~{df['duration_sec'].mean()/60:.1f} mins)")
    print(f"Nominal 1Hz sampling adherence: {df['dt_1s_pct'].mean():.2f}%")
    print(f"Average Total GPS Trajectory Distance: {df['total_gps_dist_km'].mean():.2f} km")
    print(f"Average Speed: {df['mean_speed_kmh'].mean():.2f} km/h (Max recorded: {df['max_speed_kmh'].max():.2f} km/h)")
    print(f"Total Missing Values Across Entire Dataset: {df['missing_values'].sum()}")
    print(f"Average Ground Truth Route Continuity: {df[df['partition']=='train']['route_continuity_pct'].mean():.2f}%")


if __name__ == '__main__':
    main()
