"""
PathSnap Dataset Loader and Utilities
Handles loading, validation, and geographic distance calculations for:
- .track (GPS observations: longitude, latitude, timestamp)
- .nodes (Road network nodes: longitude, latitude)
- .arcs  (Road network connections: from_node_id, to_node_id)
- .route (Ground truth traversed arc_ids)
"""

import os
import math
import numpy as np
import pandas as pd
import networkx as nx

EARTH_RADIUS_METERS = 6371000.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes geodesic distance between two points in meters using Haversine formula.
    Can accept float numbers or numpy arrays.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_METERS * c


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """
    Calculates minimum distance from point P to line segment AB in meters.
    Assumes local equirectangular projection centered at py, px.
    """
    # Convert lat/lon to local meters relative to P
    cos_lat = math.cos(math.radians(py))
    def to_xy(lon, lat):
        x = math.radians(lon - px) * EARTH_RADIUS_METERS * cos_lat
        y = math.radians(lat - py) * EARTH_RADIUS_METERS
        return x, y

    p_x, p_y = 0.0, 0.0
    a_x, a_y = to_xy(ax, ay)
    b_x, b_y = to_xy(bx, by)

    dx = b_x - a_x
    dy = b_y - a_y
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        return math.hypot(p_x - a_x, p_y - a_y)

    # Project point P onto AB clamped to [0, 1]
    t = max(0.0, min(1.0, ((p_x - a_x) * dx + (p_y - a_y) * dy) / seg_len_sq))
    proj_x = a_x + t * dx
    proj_y = a_y + t * dy

    return math.hypot(p_x - proj_x, p_y - proj_y)


class TripData:
    """Represents all data associated with a single trip."""
    def __init__(self, trip_id, partition, track_df, nodes_df, arcs_df, route_df=None):
        self.trip_id = trip_id
        self.partition = partition
        self.track = track_df
        self.nodes = nodes_df
        self.arcs = arcs_df
        self.route = route_df
        self._graph = None

    @property
    def graph(self):
        """Constructs and caches a NetworkX directed graph for the road network."""
        if self._graph is None:
            G = nx.DiGraph()
            for node_id, row in self.nodes.iterrows():
                G.add_node(node_id, lon=row['longitude'], lat=row['latitude'])
            for arc_id, row in self.arcs.iterrows():
                G.add_edge(row['from_node_id'], row['to_node_id'], arc_id=arc_id)
            self._graph = G
        return self._graph

    def get_arc_endpoints(self, arc_id):
        """Returns ((from_lon, from_lat), (to_lon, to_lat)) for a given arc_id."""
        if arc_id not in self.arcs.index:
            raise KeyError(f"Arc ID {arc_id} not found in trip {self.trip_id}")
        u = self.arcs.loc[arc_id, 'from_node_id']
        v = self.arcs.loc[arc_id, 'to_node_id']
        u_row = self.nodes.loc[u]
        v_row = self.nodes.loc[v]
        return ((u_row['longitude'], u_row['latitude']), (v_row['longitude'], v_row['latitude']))


def load_trip(trip_dir, partition="train", trip_id=None):
    """Loads a single trip from a directory."""
    if trip_id is None:
        trip_id = os.path.basename(trip_dir.rstrip("/\\"))

    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    # Load track
    track_df = pd.read_csv(
        track_path,
        sep=r'\s+',
        header=None,
        names=['longitude', 'latitude', 'timestamp'],
        dtype={'longitude': float, 'latitude': float, 'timestamp': float}
    )

    # Load nodes
    nodes_df = pd.read_csv(
        nodes_path,
        sep=r'\s+',
        header=None,
        names=['longitude', 'latitude'],
        dtype={'longitude': float, 'latitude': float}
    )
    nodes_df.index.name = 'node_id'

    # Load arcs
    arcs_df = pd.read_csv(
        arcs_path,
        sep=r'\s+',
        header=None,
        names=['from_node_id', 'to_node_id'],
        dtype={'from_node_id': int, 'to_node_id': int}
    )
    arcs_df.index.name = 'arc_id'

    # Load route if available
    route_df = None
    if os.path.exists(route_path):
        route_df = pd.read_csv(
            route_path,
            header=None,
            names=['arc_id'],
            dtype={'arc_id': int}
        )

    return TripData(trip_id, partition, track_df, nodes_df, arcs_df, route_df)


def discover_trips(base_dir):
    """
    Discovers all trips in standard directory structure:
    Looks inside Student folder/Student folder/train and student_test,
    or data/train and data/student_test.
    """
    candidate_train_paths = [
        os.path.join(base_dir, "Student folder", "Student folder", "train"),
        os.path.join(base_dir, "data", "train"),
        os.path.join(base_dir, "train")
    ]
    candidate_test_paths = [
        os.path.join(base_dir, "Student folder", "Student folder", "student_test"),
        os.path.join(base_dir, "data", "student_test"),
        os.path.join(base_dir, "student_test")
    ]

    train_path = next((p for p in candidate_train_paths if os.path.exists(p)), None)
    test_path = next((p for p in candidate_test_paths if os.path.exists(p)), None)

    train_trips = sorted(os.listdir(train_path)) if train_path else []
    test_trips = sorted(os.listdir(test_path)) if test_path else []

    return {
        'train_path': train_path,
        'test_path': test_path,
        'train_trips': train_trips,
        'test_trips': test_trips
    }
