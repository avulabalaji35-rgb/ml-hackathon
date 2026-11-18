"""
Script to generate all required EDA plots and interactive Folium maps
for the PathSnap hackathon.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath("."))
from src.data.loader import discover_trips, load_trip, haversine_distance
from src.visualization.folium_maps import save_trip_map

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 14

FIG_DIR = "reports/figures"
MAP_DIR = "reports/maps"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)


def plot_sampling_intervals(df_trips, sample_trips_data):
    """Figure 1: Sampling Interval (dt) Distribution & Anomalies"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregated dt across 5 sample trips
    all_dts = []
    for trip in sample_trips_data:
        t = trip.track['timestamp'].values
        all_dts.extend(np.diff(t))
    all_dts = np.array(all_dts)

    # Left: Histogram near 1s
    dts_normal = all_dts[all_dts <= 5.0]
    pct_1s = 100.0 * np.sum(np.isclose(all_dts, 1.0, atol=0.01)) / len(all_dts)
    sns.histplot(dts_normal, bins=40, color="#1f77b4", ax=axes[0], kde=False, edgecolor='black')
    axes[0].set_title(f"GPS Sampling Delta ($\Delta t \leq 5s$) — {pct_1s:.1f}% Exactly 1.0s")
    axes[0].set_xlabel("Time Delta between consecutive points (seconds)")
    axes[0].set_ylabel("Count of Observations")
    axes[0].axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Nominal 1.0s (1 Hz)')
    axes[0].legend()

    # Right: Distribution of Max Gaps across all trips
    sns.boxplot(y=df_trips['dt_max_sec'], color="#aec7e8", ax=axes[1])
    axes[1].set_title("Maximum GPS Dropout Gap per Trip ($\Delta t_{max}$)")
    axes[1].set_ylabel("Max Gap Duration (seconds)")
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_01_sampling_interval_dist.png"), dpi=300)
    plt.close()
    print("Saved eda_01_sampling_interval_dist.png")


def plot_speed_distributions(sample_trips_data):
    """Figure 2: Speeds and GPS Jump Anomalies"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_speeds_kmh = []
    for trip in sample_trips_data:
        t = trip.track
        dt = np.diff(t['timestamp'].values)
        dists = haversine_distance(
            t['latitude'].values[:-1], t['longitude'].values[:-1],
            t['latitude'].values[1:], t['longitude'].values[1:]
        )
        speeds = np.where(dt > 0, dists / dt, 0.0) * 3.6
        all_speeds_kmh.extend(speeds)
    all_speeds_kmh = np.array(all_speeds_kmh)

    # Plausible driving speeds
    plausible_speeds = all_speeds_kmh[all_speeds_kmh <= 140.0]
    sns.histplot(plausible_speeds, bins=45, color="#2ca02c", ax=axes[0], kde=True, edgecolor='black')
    axes[0].set_title("Vehicle Speed Distribution (Plausible Driving: <= 140 km/h)")
    axes[0].set_xlabel("Speed (km/h)")
    axes[0].set_ylabel("Observation Count")
    axes[0].axvline(np.median(plausible_speeds), color='black', linestyle=':', label=f'Median: {np.median(plausible_speeds):.1f} km/h')
    axes[0].legend()

    # Outlier / Anomaly identification
    outliers = all_speeds_kmh[all_speeds_kmh > 140.0]
    axes[1].bar(['Plausible (<= 140 km/h)', 'GPS Teleport Jumps (> 140 km/h)'],
                [len(plausible_speeds), len(outliers)], color=['#2ca02c', '#d62728'], edgecolor='black')
    axes[1].set_title(f"GPS Anomaly Detection ({len(outliers)} jumps detected out of {len(all_speeds_kmh)} points)")
    axes[1].set_ylabel("Number of Point Transitions")
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_02_speed_distribution.png"), dpi=300)
    plt.close()
    print("Saved eda_02_speed_distribution.png")


def plot_road_network_topology(sample_trip):
    """Figure 3: Road Network Graph Degree Distribution"""
    G = sample_trip.graph
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    in_counts = pd.Series(in_degrees).value_counts().sort_index()
    out_counts = pd.Series(out_degrees).value_counts().sort_index()

    df_deg = pd.DataFrame({'In-Degree': in_counts, 'Out-Degree': out_counts}).fillna(0)
    df_deg.plot(kind='bar', ax=axes[0], color=['#1f77b4', '#ff7f0e'], width=0.8, edgecolor='black')
    axes[0].set_title(f"Node Degree Distribution (Trip {sample_trip.trip_id} Road Network)")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Number of Nodes")
    axes[0].set_yscale('log')
    axes[0].set_xlim(-0.5, 6.5)

    # Graph connectivity breakdown pie
    dead_ends = sum(1 for d in in_degrees if d <= 1)
    intersections = sum(1 for d in in_degrees if d >= 3)
    normal_segments = len(in_degrees) - dead_ends - intersections
    
    axes[1].pie([dead_ends, normal_segments, intersections],
                labels=['Dead-ends/Terminals (deg <= 1)', 'Through-segments (deg 2)', 'Intersections (deg >= 3)'],
                autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'],
                startangle=140, explode=(0.05, 0, 0.05))
    axes[1].set_title("Road Network Intersection Topology Composition")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_03_road_network_degrees.png"), dpi=300)
    plt.close()
    print("Saved eda_03_road_network_degrees.png")


def plot_arc_length_distribution(sample_trip):
    """Figure 4: Arc Physical Length Distribution"""
    arcs = sample_trip.arcs
    nodes = sample_trip.nodes

    u_lats = nodes.loc[arcs['from_node_id'], 'latitude'].values
    u_lons = nodes.loc[arcs['from_node_id'], 'longitude'].values
    v_lats = nodes.loc[arcs['to_node_id'], 'latitude'].values
    v_lons = nodes.loc[arcs['to_node_id'], 'longitude'].values

    lengths_m = haversine_distance(u_lats, u_lons, v_lats, v_lons)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(lengths_m[lengths_m <= 300], bins=50, color="#9467bd", ax=ax, kde=True, edgecolor='black')
    ax.set_title(f"Road Segment (Arc) Physical Lengths (Median: {np.median(lengths_m):.1f}m, Max: {np.max(lengths_m):.1f}m)")
    ax.set_xlabel("Arc Length (meters)")
    ax.set_ylabel("Arc Count")
    ax.axvline(np.median(lengths_m), color='red', linestyle='--', label=f'Median: {np.median(lengths_m):.1f}m')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_04_arc_length_distribution.png"), dpi=300)
    plt.close()
    print("Saved eda_04_arc_length_distribution.png")


def plot_trip_distance_vs_duration(df_trips):
    """Figure 5: Trajectory Duration vs Distance"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.scatterplot(
        data=df_trips,
        x='duration_sec',
        y='total_gps_dist_km',
        hue='partition',
        style='partition',
        s=70,
        palette={'train': '#1f77b4', 'student_test': '#d62728'},
        ax=ax
    )
    ax.set_title("Trajectory Scale: Total Duration vs Trajectory Distance")
    ax.set_xlabel("Trip Duration (seconds)")
    ax.set_ylabel("Total GPS Distance (km)")
    ax.legend(title="Partition")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_05_trajectory_distance_duration.png"), dpi=300)
    plt.close()
    print("Saved eda_05_trajectory_distance_duration.png")


def plot_candidate_miss_tradeoff():
    """Figure 6: Spatial Candidate Search Radius vs Candidate Count and Miss Rate"""
    cand_csv = "reports/candidate_search_metrics.csv"
    if not os.path.exists(cand_csv):
        return

    df_cand = pd.read_csv(cand_csv)
    radii = [10, 25, 50, 75, 100]
    avg_cands = [df_cand[f'avg_cands_{r}m'].mean() for r in radii]
    miss_rates = [df_cand[f'miss_rate_{r}m'].mean() for r in radii]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color = '#1f77b4'
    ax1.set_xlabel('Spatial Candidate Search Radius $R$ (meters)')
    ax1.set_ylabel('Average Candidate Count ($k$)', color=color)
    line1 = ax1.plot(radii, avg_cands, marker='o', linewidth=2.5, color=color, label='Avg Candidate Count')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Candidate Miss Rate (%)', color=color)
    line2 = ax2.plot(radii, miss_rates, marker='s', linewidth=2.5, linestyle='--', color=color, label='Candidate Miss Rate (%)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlight optimal trade-off zone (25m - 50m)
    ax1.axvspan(25, 50, color='green', alpha=0.15, label='Recommended Search Zone (25m-50m)')

    lines = line1 + line2
    labels = [l.get_label() for l in lines] + ['Recommended Search Zone (25m-50m)']
    ax1.legend(lines + [plt.Rectangle((0,0),1,1, color='green', alpha=0.15)], labels, loc='center right')

    plt.title("Map-Matching Candidate Search: Computation vs Miss-Rate Trade-off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_06_candidate_miss_tradeoff.png"), dpi=300)
    plt.close()
    print("Saved eda_06_candidate_miss_tradeoff.png")


def generate_interactive_maps(sample_train_trip, sample_test_trip):
    """Generates standalone interactive Folium HTML maps."""
    map1_path = os.path.join(MAP_DIR, f"trip_{sample_train_trip.trip_id}_interactive_map.html")
    save_trip_map(sample_train_trip, map1_path, max_network_arcs=2500)
    print(f"Generated Folium map: {map1_path}")

    map2_path = os.path.join(MAP_DIR, f"trip_{sample_test_trip.trip_id}_interactive_map.html")
    save_trip_map(sample_test_trip, map2_path, max_network_arcs=2500)
    print(f"Generated Folium map: {map2_path}")


def main():
    trips_info = discover_trips(".")
    train_dir = trips_info['train_path']
    test_dir = trips_info['test_path']
    
    df_trips = pd.read_csv("reports/dataset_inventory_summary.csv")

    # Load 5 sample training trips for granular distributions
    sample_train_trips = []
    for t_id in trips_info['train_trips'][:5]:
        sample_train_trips.append(load_trip(os.path.join(train_dir, t_id), partition="train"))

    sample_test_trip = load_trip(os.path.join(test_dir, trips_info['test_trips'][0]), partition="student_test")

    print("Generating EDA Figures...")
    plot_sampling_intervals(df_trips, sample_train_trips)
    plot_speed_distributions(sample_train_trips)
    plot_road_network_topology(sample_train_trips[0])
    plot_arc_length_distribution(sample_train_trips[0])
    plot_trip_distance_vs_duration(df_trips)
    plot_candidate_miss_tradeoff()

    print("Generating Interactive Folium Maps...")
    generate_interactive_maps(sample_train_trips[0], sample_test_trip)

    print("All visualizations generated successfully.")


if __name__ == '__main__':
    main()
