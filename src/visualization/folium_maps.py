"""
Folium Visualization Module for PathSnap Map-Matching and Trajectory Reconstruction.
Creates interactive multi-layer Leaflet/Folium maps visualizing:
1. Road Network Arcs & Nodes
2. Raw GPS Trajectory (points + sequence polyline)
3. Ground Truth Traversed Route
"""

import os
import folium
from folium import plugins


def create_trip_map(trip, max_network_arcs=2000, show_network=True):
    """
    Builds an interactive Folium map for a TripData instance.
    
    Parameters:
    - trip: TripData object
    - max_network_arcs: max number of background road arcs to render (for browser performance)
    - show_network: bool, whether to include road network layer
    
    Returns:
    - folium.Map instance
    """
    track_df = trip.track
    nodes_df = trip.nodes
    arcs_df = trip.arcs
    route_df = trip.route

    # Center map on centroid of GPS trajectory
    center_lat = track_df['latitude'].mean()
    center_lon = track_df['longitude'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )

    # 1. Road Network Layer (Background)
    if show_network and arcs_df is not None and len(arcs_df) > 0:
        network_group = folium.FeatureGroup(name="Road Network (Arcs)", show=False)
        
        # Determine bounding box with small padding
        lat_min, lat_max = track_df['latitude'].min() - 0.01, track_df['latitude'].max() + 0.01
        lon_min, lon_max = track_df['longitude'].min() - 0.01, track_df['longitude'].max() + 0.01
        
        count = 0
        for arc_id, row in arcs_df.iterrows():
            u = row['from_node_id']
            v = row['to_node_id']
            if u in nodes_df.index and v in nodes_df.index:
                u_lat, u_lon = nodes_df.loc[u, 'latitude'], nodes_df.loc[u, 'longitude']
                v_lat, v_lon = nodes_df.loc[v, 'latitude'], nodes_df.loc[v, 'longitude']
                
                # Only draw if within padded trajectory bounds
                if (lat_min <= u_lat <= lat_max or lat_min <= v_lat <= lat_max) and \
                   (lon_min <= u_lon <= lon_max or lon_min <= v_lon <= lon_max):
                    folium.PolyLine(
                        locations=[[u_lat, u_lon], [v_lat, v_lon]],
                        color="#808080",
                        weight=2,
                        opacity=0.6,
                        tooltip=f"Arc ID: {arc_id} ({u}->{v})"
                    ).add_to(network_group)
                    count += 1
                    if count >= max_network_arcs:
                        break
        network_group.add_to(m)

    # 2. Ground Truth Route Layer
    if route_df is not None and len(route_df) > 0:
        gt_group = folium.FeatureGroup(name="Ground Truth Route (Reference)", show=True)
        route_coords = []
        for seq_idx, row in route_df.iterrows():
            arc_id = row['arc_id']
            if arc_id in arcs_df.index:
                u = arcs_df.loc[arc_id, 'from_node_id']
                v = arcs_df.loc[arc_id, 'to_node_id']
                if u in nodes_df.index and v in nodes_df.index:
                    u_pt = [nodes_df.loc[u, 'latitude'], nodes_df.loc[u, 'longitude']]
                    v_pt = [nodes_df.loc[v, 'latitude'], nodes_df.loc[v, 'longitude']]
                    route_coords.append(u_pt)
                    route_coords.append(v_pt)
                    folium.PolyLine(
                        locations=[u_pt, v_pt],
                        color="#28a745", # Vibrant Green
                        weight=5,
                        opacity=0.85,
                        tooltip=f"GT Step {seq_idx}: Arc {arc_id}"
                    ).add_to(gt_group)
        gt_group.add_to(m)

    # 3. Raw GPS Trajectory Line
    gps_line_group = folium.FeatureGroup(name="Raw GPS Path", show=True)
    gps_coords = track_df[['latitude', 'longitude']].values.tolist()
    folium.PolyLine(
        locations=gps_coords,
        color="#0066cc", # Vibrant Blue
        weight=2.5,
        opacity=0.7,
        dash_array="5, 5"
    ).add_to(gps_line_group)
    gps_line_group.add_to(m)

    # 4. Raw GPS Observation Points
    gps_pts_group = folium.FeatureGroup(name="Raw GPS Points", show=True)
    for idx, row in track_df.iterrows():
        popup_txt = f"<b>Obs #{idx}</b><br>Time: {row['timestamp']:.1f}s<br>Lat: {row['latitude']:.6f}<br>Lon: {row['longitude']:.6f}"
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3.5,
            color="#003399",
            fill=True,
            fill_color="#3399ff",
            fill_opacity=0.9,
            popup=popup_txt,
            tooltip=f"GPS #{idx} (t={row['timestamp']:.0f}s)"
        ).add_to(gps_pts_group)
        
    # Mark Start and End Points
    start_pt = [track_df.iloc[0]['latitude'], track_df.iloc[0]['longitude']]
    end_pt = [track_df.iloc[-1]['latitude'], track_df.iloc[-1]['longitude']]
    folium.Marker(
        start_pt,
        popup=f"Start: Trip {trip.trip_id}",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(gps_pts_group)
    folium.Marker(
        end_pt,
        popup=f"End: Trip {trip.trip_id}",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(gps_pts_group)

    gps_pts_group.add_to(m)

    # Add Map Controls
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)

    # Fit bounds to trajectory
    m.fit_bounds([
        [track_df['latitude'].min(), track_df['longitude'].min()],
        [track_df['latitude'].max(), track_df['longitude'].max()]
    ])

    return m


def save_trip_map(trip, output_html_path, max_network_arcs=2000):
    """Generates and saves trip map to HTML file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    m = create_trip_map(trip, max_network_arcs=max_network_arcs)
    m.save(output_html_path)
    return output_html_path
