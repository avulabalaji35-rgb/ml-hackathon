"""
Hidden Markov Model (HMM) and Viterbi Sequential Route Reconstruction Engine for PathSnap.
Combines machine-learning emission probabilities with road-network topological transition costs
to decode physically realistic and continuous vehicle routes.
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
import joblib

from src.features.extractor import CandidateExtractor, align_route_to_track, EARTH_RADIUS


class MapMatchingHMM:
    """
    HMM Route Reconstruction Engine using trained ML classifier emissions and
    road network geodesic shortest-path transitions.
    """
    def __init__(self, model_or_path="models/tuned_model.joblib", beta=6.0, search_radius=35.0):
        self.beta = beta
        self.search_radius = search_radius
        self.extractor = CandidateExtractor(search_radius=search_radius)

        if isinstance(model_or_path, str):
            self.model = joblib.load(model_or_path)
        else:
            self.model = model_or_path

        self.feature_cols = [
            'dist_to_segment', 'heading_diff', 'dist_to_u', 'dist_to_v',
            'segment_length', 'proj_ratio', 'gps_speed_mps', 'speed_length_ratio',
            'node_in_degree_u', 'node_out_degree_v', 'spatial_rank'
        ]

    def decode_trip(self, trip, sample_step=1):
        """
        Runs full map matching on a trip.
        Returns:
            dict containing:
            - 'pointwise_classifier_arcs': raw classifier top-1 predictions
            - 'pointwise_hmm_arcs': HMM Viterbi decoded arc per observation
            - 'reconstructed_route': full continuous sequence of traversed arc IDs
            - 'ground_truth_aligned': true assigned arc per observation (if route supplied)
            - 'metrics': accuracy and continuity metrics
        """
        track = trip.track
        nodes = trip.nodes
        arcs = trip.arcs
        G = trip.graph

        # 1. Extract candidates and compute classifier emission probabilities
        df_feat = self.extractor.extract_for_trip(trip, sample_step=sample_step, max_candidates_per_obs=8)
        if len(df_feat) == 0:
            return None

        X = df_feat[self.feature_cols].values
        probs = self.model.predict_proba(X)[:, 1]
        df_feat['prob'] = probs

        # 2. Organize observations into time-indexed candidate pools
        obs_groups = df_feat.groupby('obs_idx')
        obs_indices = sorted(obs_groups.groups.keys())
        T = len(obs_indices)

        candidates_per_step = []
        emissions_per_step = []
        classifier_top1_arcs = []

        for obs_idx in obs_indices:
            group = obs_groups.get_group(obs_idx).sort_values(by='prob', ascending=False)
            cands = group['arc_id'].values
            p_vals = group['prob'].values

            classifier_top1_arcs.append(cands[0])

            # Normalize probabilities across candidates with epsilon smoothing
            p_norm = p_vals / (np.sum(p_vals) + 1e-9)
            p_smooth = np.clip(p_norm, 1e-6, 1.0)
            log_emissions = np.log(p_smooth)

            candidates_per_step.append(cands)
            emissions_per_step.append(log_emissions)

        # Pre-cache arc endpoint nodes
        u_arr = arcs['from_node_id'].to_dict()
        v_arr = arcs['to_node_id'].to_dict()

        # 3. Viterbi Trellis Initialization
        V = [emissions_per_step[0].copy()] # V_0
        backpointers = []

        # Precompute GPS timestamps and coords for step-to-step transitions
        t_lats = track.loc[obs_indices, 'latitude'].values
        t_lons = track.loc[obs_indices, 'longitude'].values
        t_times = track.loc[obs_indices, 'timestamp'].values

        # Shortest path distance cache to avoid redundant Dijkstra calculations
        sp_cache = {}

        def get_shortest_path_dist(u, v):
            if u == v:
                return 0.0
            key = (u, v)
            if key in sp_cache:
                return sp_cache[key]
            try:
                # Use topological distance / hop count scaled by mean segment length (45m)
                # or shortest path length
                length = nx.shortest_path_length(G, source=u, target=v) * 45.0
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                length = float('inf')
            sp_cache[key] = length
            return length

        # 4. Viterbi Forward Induction
        for t in range(1, T):
            prev_cands = candidates_per_step[t - 1]
            curr_cands = candidates_per_step[t]
            curr_emissions = emissions_per_step[t]

            dt = max(1.0, t_times[t] - t_times[t - 1])
            # Haversine distance between consecutive GPS fixes
            phi1, phi2 = math.radians(t_lats[t-1]), math.radians(t_lats[t])
            dphi = phi2 - phi1
            dlam = math.radians(t_lons[t] - t_lons[t-1])
            a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2.0)**2
            dist_gps = 2.0 * EARTH_RADIUS * math.asin(math.sqrt(min(1.0, a)))

            eff_beta = self.beta * (1.0 + 0.15 * math.sqrt(dt))

            K_prev = len(prev_cands)
            K_curr = len(curr_cands)

            V_curr = np.zeros(K_curr)
            bp_curr = np.zeros(K_curr, dtype=int)

            for j in range(K_curr):
                curr_arc = curr_cands[j]
                curr_u = u_arr.get(curr_arc, -1)

                best_val = -1e9
                best_prev = 0

                for i in range(K_prev):
                    prev_arc = prev_cands[i]
                    prev_v = v_arr.get(prev_arc, -1)

                    # Compute transition cost
                    if prev_arc == curr_arc:
                        # Same segment
                        log_trans = -0.1
                    elif prev_v == curr_u:
                        # Topologically connected
                        log_trans = 0.0
                    else:
                        d_net = get_shortest_path_dist(prev_v, curr_u)
                        if math.isinf(d_net):
                            log_trans = -15.0 # Unreachable disconnected penalty
                        else:
                            diff = abs(d_net - dist_gps)
                            log_trans = - (diff / eff_beta) - math.log(eff_beta)

                    score = V[t - 1][i] + log_trans
                    if score > best_val:
                        best_val = score
                        best_prev = i

                V_curr[j] = best_val + curr_emissions[j]
                bp_curr[j] = best_prev

            V.append(V_curr)
            backpointers.append(bp_curr)

        # 5. Backtracking
        best_last_idx = int(np.argmax(V[-1]))
        decoded_indices = [best_last_idx]

        for t in range(T - 2, -1, -1):
            prev_idx = backpointers[t][decoded_indices[-1]]
            decoded_indices.append(prev_idx)

        decoded_indices.reverse()
        hmm_decoded_arcs = [candidates_per_step[t][decoded_indices[t]] for t in range(T)]

        # 6. Reconstruct Continuous Route Path
        reconstructed_route = []
        for i in range(len(hmm_decoded_arcs)):
            a = hmm_decoded_arcs[i]
            if len(reconstructed_route) == 0:
                reconstructed_route.append(a)
            elif reconstructed_route[-1] != a:
                # Check topological connection
                last_arc = reconstructed_route[-1]
                last_v = v_arr.get(last_arc, -1)
                next_u = u_arr.get(a, -1)

                if last_v == next_u:
                    reconstructed_route.append(a)
                else:
                    # Fill path via Dijkstra if available
                    try:
                        node_path = nx.shortest_path(G, source=last_v, target=next_u)
                        # Recover intermediate arcs
                        for n_i in range(len(node_path) - 1):
                            edge_data = G.get_edge_data(node_path[n_i], node_path[n_i + 1])
                            if edge_data and 'arc_id' in edge_data:
                                reconstructed_route.append(edge_data['arc_id'])
                        reconstructed_route.append(a)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        reconstructed_route.append(a)

        # 7. Calculate Metrics against Ground Truth
        gt_aligned = align_route_to_track(trip)
        gt_at_obs = gt_aligned[obs_indices] if gt_aligned is not None else None

        metrics = {}
        if gt_at_obs is not None:
            clf_acc = float(np.mean(np.array(classifier_top1_arcs) == gt_at_obs))
            hmm_acc = float(np.mean(np.array(hmm_decoded_arcs) == gt_at_obs))
            metrics['classifier_only_accuracy'] = clf_acc
            metrics['hmm_viterbi_accuracy'] = hmm_acc
            metrics['accuracy_improvement'] = hmm_acc - clf_acc

        # Calculate route continuity (% of consecutive steps that connect)
        def calc_continuity(arc_seq):
            if len(arc_seq) <= 1:
                return 1.0
            matches = 0
            for i in range(len(arc_seq) - 1):
                a1, a2 = arc_seq[i], arc_seq[i+1]
                if a1 == a2 or v_arr.get(a1, -1) == u_arr.get(a2, -2):
                    matches += 1
            return float(matches / (len(arc_seq) - 1))

        metrics['classifier_continuity'] = calc_continuity(classifier_top1_arcs)
        metrics['hmm_continuity'] = calc_continuity(hmm_decoded_arcs)
        metrics['reconstructed_route_continuity'] = calc_continuity(reconstructed_route)
        metrics['reconstructed_route_length'] = len(reconstructed_route)

        return {
            'trip_id': trip.trip_id,
            'obs_indices': obs_indices,
            'classifier_top1_arcs': classifier_top1_arcs,
            'hmm_decoded_arcs': hmm_decoded_arcs,
            'reconstructed_route': reconstructed_route,
            'ground_truth_route': trip.route['arc_id'].values.tolist() if trip.route is not None else [],
            'metrics': metrics
        }
