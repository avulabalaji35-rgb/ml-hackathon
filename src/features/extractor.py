"""
High-Performance Feature Extraction and Candidate Road Segment Generator for PathSnap.
Calculates spatial, heading, kinematic, and topological features for
observation-candidate arc pairs using vectorized NumPy preprocessing and Shapely 2.0 batch STRtree queries.
"""

import math
import numpy as np
import pandas as pd
from shapely import points, linestrings, buffer
from shapely.strtree import STRtree

EARTH_RADIUS = 6371000.0


def compute_bearing(lat1, lon1, lat2, lon2):
    """Calculates forward azimuth from (lat1, lon1) to (lat2, lon2) in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def heading_difference(h1, h2):
    """Calculates the minimal angular difference between two bearings in [0, 180] degrees."""
    diff = abs(h1 - h2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def align_route_to_track(trip):
    """
    Chronologically aligns the sequence of ground-truth traversed arcs in trip.route
    to the sequential GPS observations in trip.track.
    """
    if trip.route is None or len(trip.route) == 0:
        return None

    track = trip.track
    route_arcs = trip.route['arc_id'].values
    arcs = trip.arcs
    nodes = trip.nodes

    node_lons = nodes['longitude'].values
    node_lats = nodes['latitude'].values
    max_node = len(nodes)

    u_all = arcs['from_node_id'].values
    v_all = arcs['to_node_id'].values

    # Pre-fetch arc endpoints
    endpoints = {}
    for a in route_arcs:
        if a not in endpoints and a < len(arcs):
            u, v = u_all[a], v_all[a]
            if u < max_node and v < max_node:
                endpoints[a] = (node_lons[u], node_lats[u], node_lons[v], node_lats[v])

    assigned_arcs = []
    curr_r_idx = 0
    N_route = len(route_arcs)
    t_lons = track['longitude'].values
    t_lats = track['latitude'].values

    for t_idx in range(len(track)):
        px, py = t_lons[t_idx], t_lats[t_idx]
        window_end = min(N_route, curr_r_idx + 10)
        best_r = curr_r_idx
        best_d = 1e9
        cos_lat = math.cos(math.radians(py))

        for r_i in range(curr_r_idx, window_end):
            a_id = route_arcs[r_i]
            if a_id in endpoints:
                ux, uy, vx, vy = endpoints[a_id]
                dx = (vx - ux) * math.radians(1.0) * EARTH_RADIUS * cos_lat
                dy = (vy - uy) * math.radians(1.0) * EARTH_RADIUS
                seg_sq = dx * dx + dy * dy

                px_m = (px - ux) * math.radians(1.0) * EARTH_RADIUS * cos_lat
                py_m = (py - uy) * math.radians(1.0) * EARTH_RADIUS

                if seg_sq == 0:
                    d = math.hypot(px_m, py_m)
                else:
                    t = max(0.0, min(1.0, (px_m * dx + py_m * dy) / seg_sq))
                    d = math.hypot(px_m - t * dx, py_m - t * dy)

                if d < best_d:
                    best_d = d
                    best_r = r_i

        curr_r_idx = best_r
        assigned_arcs.append(route_arcs[curr_r_idx])

    return np.array(assigned_arcs)


class CandidateExtractor:
    """Fast candidate road segment generator and feature extractor using batch STRtree queries."""
    def __init__(self, search_radius=35.0):
        self.search_radius = search_radius

    def extract_for_trip(self, trip, sample_step=1, max_candidates_per_obs=10):
        track = trip.track
        nodes = trip.nodes
        arcs = trip.arcs
        G = trip.graph

        gt_aligned = align_route_to_track(trip)

        center_lat = track['latitude'].mean()
        center_lon = track['longitude'].mean()
        cos_lat = math.cos(math.radians(center_lat))
        R = EARTH_RADIUS

        # Vectorized segment filtering
        u_arr = arcs['from_node_id'].values
        v_arr = arcs['to_node_id'].values
        arc_ids_all = arcs.index.values

        node_lons = nodes['longitude'].values
        node_lats = nodes['latitude'].values
        max_node = len(nodes)

        valid_nodes = (u_arr < max_node) & (v_arr < max_node)
        u_arr = u_arr[valid_nodes]
        v_arr = v_arr[valid_nodes]
        arc_ids_all = arc_ids_all[valid_nodes]

        u_lons = node_lons[u_arr]
        u_lats = node_lats[u_arr]
        v_lons = node_lons[v_arr]
        v_lats = node_lats[v_arr]

        # Bounding box buffer
        lat_buf = (self.search_radius + 60.0) / (math.pi * R / 180.0)
        lon_buf = lat_buf / cos_lat
        lat_min, lat_max = track['latitude'].min() - lat_buf, track['latitude'].max() + lat_buf
        lon_min, lon_max = track['longitude'].min() - lon_buf, track['longitude'].max() + lon_buf

        in_bbox = (
            ((lat_min <= u_lats) & (u_lats <= lat_max)) | ((lat_min <= v_lats) & (v_lats <= lat_max))
        ) & (
            ((lon_min <= u_lons) & (u_lons <= lon_max)) | ((lon_min <= v_lons) & (v_lons <= lon_max))
        )

        u_arr = u_arr[in_bbox]
        v_arr = v_arr[in_bbox]
        arc_ids_all = arc_ids_all[in_bbox]
        u_lons = u_lons[in_bbox]
        u_lats = u_lats[in_bbox]
        v_lons = v_lons[in_bbox]
        v_lats = v_lats[in_bbox]

        if len(arc_ids_all) == 0:
            return pd.DataFrame()

        u_x = np.radians(u_lons - center_lon) * R * cos_lat
        u_y = np.radians(u_lats - center_lat) * R
        v_x = np.radians(v_lons - center_lon) * R * cos_lat
        v_y = np.radians(v_lats - center_lat) * R

        non_degen = (u_x != v_x) | (u_y != v_y)
        u_x, u_y = u_x[non_degen], u_y[non_degen]
        v_x, v_y = v_x[non_degen], v_y[non_degen]
        u_lons, u_lats = u_lons[non_degen], u_lats[non_degen]
        v_lons, v_lats = v_lons[non_degen], v_lats[non_degen]
        u_arr, v_arr = u_arr[non_degen], v_arr[non_degen]
        arc_ids_all = arc_ids_all[non_degen]

        # Build lines vectorized
        coords = np.column_stack([
            np.column_stack([u_x, u_y])[:, np.newaxis, :],
            np.column_stack([v_x, v_y])[:, np.newaxis, :]
        ])
        lines_geom = linestrings(coords)
        tree = STRtree(lines_geom)

        # GPS trajectory points
        sampled_indices = np.arange(0, len(track), sample_step)
        t_lats = track['latitude'].values[sampled_indices]
        t_lons = track['longitude'].values[sampled_indices]
        t_times = track['timestamp'].values[sampled_indices]

        px_arr = np.radians(t_lons - center_lon) * R * cos_lat
        py_arr = np.radians(t_lats - center_lat) * R

        # Compute speed & bearing vectorized
        n_pts = len(sampled_indices)
        gps_speeds = np.zeros(n_pts)
        gps_bearings = np.zeros(n_pts)

        for i in range(n_pts):
            if i > 0 and t_times[i] > t_times[i-1]:
                dt = t_times[i] - t_times[i-1]
                gps_speeds[i] = math.hypot(px_arr[i] - px_arr[i-1], py_arr[i] - py_arr[i-1]) / dt
            elif i < n_pts - 1 and t_times[i+1] > t_times[i]:
                dt = t_times[i+1] - t_times[i]
                gps_speeds[i] = math.hypot(px_arr[i+1] - px_arr[i], py_arr[i+1] - py_arr[i]) / dt

            if i < n_pts - 1:
                gps_bearings[i] = compute_bearing(t_lats[i], t_lons[i], t_lats[i+1], t_lons[i+1])
            elif i > 0:
                gps_bearings[i] = compute_bearing(t_lats[i-1], t_lons[i-1], t_lats[i], t_lons[i])

        # Batch query in C
        pt_geoms = points(np.column_stack([px_arr, py_arr]))
        pt_buffers = buffer(pt_geoms, self.search_radius)
        pt_idx_match, line_idx_match = tree.query(pt_buffers)

        # Group matches by point index
        pt_to_lines = {}
        for p_i, l_i in zip(pt_idx_match, line_idx_match):
            if p_i not in pt_to_lines:
                pt_to_lines[p_i] = []
            pt_to_lines[p_i].append(l_i)

        records = []
        for p_i in range(n_pts):
            t_idx = sampled_indices[p_i]
            px, py = px_arr[p_i], py_arr[p_i]
            speed = gps_speeds[p_i]
            bearing = gps_bearings[p_i]
            true_arc = gt_aligned[t_idx] if gt_aligned is not None else None

            candidate_line_indices = pt_to_lines.get(p_i, [])
            if len(candidate_line_indices) == 0:
                nearest_l = tree.nearest(pt_geoms[p_i])
                candidate_line_indices = [nearest_l]

            candidates = []
            for l_i in candidate_line_indices:
                a_id = arc_ids_all[l_i]
                ux, uy = u_x[l_i], u_y[l_i]
                vx, vy = v_x[l_i], v_y[l_i]

                dx = vx - ux
                dy = vy - uy
                seg_len_sq = dx * dx + dy * dy
                seg_len = math.sqrt(seg_len_sq)

                t_proj = max(0.0, min(1.0, ((px - ux) * dx + (py - uy) * dy) / seg_len_sq)) if seg_len_sq > 0 else 0.0
                proj_x = ux + t_proj * dx
                proj_y = uy + t_proj * dy

                dist_to_seg = math.hypot(px - proj_x, py - proj_y)
                dist_u = math.hypot(px - ux, py - uy)
                dist_v = math.hypot(px - vx, py - vy)

                arc_bearing = compute_bearing(u_lats[l_i], u_lons[l_i], v_lats[l_i], v_lons[l_i])
                heading_err = heading_difference(bearing, arc_bearing)
                speed_len_ratio = (speed / seg_len) if seg_len > 0 else 0.0

                u_id, v_id = u_arr[l_i], v_arr[l_i]
                in_deg_u = G.in_degree(u_id) if u_id in G else 1
                out_deg_v = G.out_degree(v_id) if v_id in G else 1

                is_match = 1 if (true_arc is not None and a_id == true_arc) else 0

                candidates.append({
                    'trip_id': trip.trip_id,
                    'obs_idx': t_idx,
                    'arc_id': a_id,
                    'dist_to_segment': dist_to_seg,
                    'heading_diff': heading_err,
                    'dist_to_u': dist_u,
                    'dist_to_v': dist_v,
                    'segment_length': seg_len,
                    'proj_ratio': t_proj,
                    'gps_speed_mps': speed,
                    'speed_length_ratio': speed_len_ratio,
                    'node_in_degree_u': in_deg_u,
                    'node_out_degree_v': out_deg_v,
                    'is_match': is_match
                })

            candidates.sort(key=lambda x: x['dist_to_segment'])
            candidates = candidates[:max_candidates_per_obs]

            for rank, item in enumerate(candidates, start=1):
                item['spatial_rank'] = rank
                records.append(item)

        return pd.DataFrame(records)
