"""
FastAPI Backend Server for PathSnap Map-Matching & Route Reconstruction (Laptop 1).
Hosts the trained classifier, HMM/Viterbi implementation, road-network cache,
and server-side evaluation pipeline with strictly isolated ground-truth labels.
"""

import os
import sys
import time
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.append(os.path.abspath("."))
from src.data.loader import load_trip, discover_trips, haversine_distance
from src.hmm.decoder import MapMatchingHMM
from src.visualization.folium_maps import create_trip_map

# Configure Server-side Logging (Mandatory for Evaluator Verification)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap API] %(message)s"
)
logger = logging.getLogger("pathsnap_api")

app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Backend service hosting trained candidate classifier and HMM/Viterbi decoder.",
    version="1.0.0"
)

# Enable CORS for two-laptop cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and path references
MODEL_PATH = "models/tuned_model.joblib"
EVALUATOR_PATHS = [
    os.path.join(".", "evaluator_test", "evaluator_test"),
    os.path.join(".", "evaluator_test"),
    os.path.join(".", "Student folder", "Student folder", "student_test"),
    os.path.join(".", "data", "student_test")
]

evaluator_dir = None
for p in EVALUATOR_PATHS:
    if os.path.exists(p) and len(os.listdir(p)) > 0:
        evaluator_dir = p
        break

logger.info(f"Initialized Evaluator Directory: {evaluator_dir}")

# Load HMM Engine
logger.info(f"Loading trained ML model from: {MODEL_PATH}")
hmm_engine = MapMatchingHMM(model_or_path=MODEL_PATH, beta=6.0, search_radius=35.0)
logger.info("HMM and Viterbi Decoder initialized successfully.")


# Response Models
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    evaluator_trips_available: int


class EvaluateResponse(BaseModel):
    trip_id: str
    observations_count: int
    top1_accuracy_percent: float
    route_continuity_percent: float
    reconstructed_route_length: int
    ground_truth_route_length: int
    inference_time_ms: float
    sample_predicted_arcs: List[int]
    sample_ground_truth_arcs: List[int]
    message: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    trips_count = len(os.listdir(evaluator_dir)) if evaluator_dir else 0
    return {
        "status": "healthy",
        "service": "PathSnap Distributed Map-Matching API",
        "version": "1.0.0",
        "model_loaded": hmm_engine.model is not None,
        "evaluator_trips_available": trips_count
    }


@app.get("/api/trips")
def list_evaluation_trips():
    """Lists the 10 reserved evaluation trips available on the server."""
    if not evaluator_dir or not os.path.exists(evaluator_dir):
        raise HTTPException(status_code=404, detail="Evaluator dataset directory not found.")
    
    trips = sorted(os.listdir(evaluator_dir))
    logger.info(f"Returning {len(trips)} evaluation trip options to client.")
    return {"trips": trips, "count": len(trips)}


@app.post("/api/evaluate/{trip_id}")
async def evaluate_trip(trip_id: str, request: Request):
    """
    Evaluator evaluation endpoint:
    Processes the selected trip through the complete prediction pipeline (features -> ML classifier -> HMM Viterbi),
    calculates prediction accuracy against internal server-side ground truth, and returns metrics.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Received inference & evaluation request from [{client_ip}] for Trip ID: {trip_id}")

    if not evaluator_dir:
        raise HTTPException(status_code=500, detail="Evaluator directory not configured.")

    trip_path = os.path.join(evaluator_dir, trip_id)
    if not os.path.exists(trip_path):
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found in evaluator partition.")

    t0 = time.time()
    # Load trip
    trip = load_trip(trip_path, partition="evaluator", trip_id=trip_id)
    logger.info(f"Loaded Trip {trip_id}: {len(trip.track)} GPS fixes, {len(trip.nodes)} nodes, {len(trip.arcs)} arcs.")

    # Execute Map Matching HMM Viterbi Decoder
    res = hmm_engine.decode_trip(trip, sample_step=3)
    if res is None:
        raise HTTPException(status_code=500, detail=f"Failed to decode route for Trip {trip_id}.")

    total_time_ms = (time.time() - t0) * 1000.0
    m = res['metrics']

    acc = m.get('hmm_viterbi_accuracy', 0.0) * 100.0
    cont = m.get('reconstructed_route_continuity', 1.0) * 100.0
    rec_len = m.get('reconstructed_route_length', len(res['reconstructed_route']))
    gt_len = len(res['ground_truth_route'])

    logger.info(f"Prediction Pipeline Completed for Trip {trip_id} in {total_time_ms:.1f} ms:")
    logger.info(f"  -> Top-1 Observation Accuracy: {acc:.2f}%")
    logger.info(f"  -> Reconstructed Route Continuity: {cont:.2f}%")
    logger.info(f"  -> Reconstructed Arcs: {rec_len} (Ground Truth: {gt_len})")

    # Format GPS points and predicted coordinates for UI map display
    track_df = trip.track.iloc[res['obs_indices']].copy()
    track_points = []
    for idx, row in track_df.iterrows():
        track_points.append({
            'lat': float(row['latitude']),
            'lon': float(row['longitude']),
            't': float(row['timestamp'])
        })

    return {
        "trip_id": trip_id,
        "observations_count": int(len(track_df)),
        "top1_accuracy_percent": round(float(acc), 2),
        "route_continuity_percent": round(float(cont), 2),
        "reconstructed_route_length": int(rec_len),
        "ground_truth_route_length": int(gt_len),
        "inference_time_ms": round(float(total_time_ms), 2),
        "sample_predicted_arcs": [int(x) for x in res['reconstructed_route'][:20]],
        "sample_ground_truth_arcs": [int(x) for x in res['ground_truth_route'][:20]],
        "track_points": track_points[:200], # Subsample for fast UI transmission
        "message": f"Successfully matched and evaluated trip {trip_id}."
    }


@app.get("/api/map/{trip_id}", response_class=HTMLResponse)
def get_trip_interactive_map(trip_id: str):
    """Generates on-demand interactive Folium map visualization for a trip."""
    if not evaluator_dir:
        raise HTTPException(status_code=500, detail="Evaluator directory not configured.")
    trip_path = os.path.join(evaluator_dir, trip_id)
    if not os.path.exists(trip_path):
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found.")

    trip = load_trip(trip_path, partition="evaluator", trip_id=trip_id)
    m = create_trip_map(trip, max_network_arcs=1200)
    return m.get_root().render()


# Mount UI static files
ui_dir = os.path.join(os.path.abspath("."), "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")


@app.get("/", response_class=HTMLResponse)
def root_redirect():
    """Redirect root to UI."""
    return """<!DOCTYPE html>
<html>
<head><meta http-equiv="refresh" content="0; url=/ui/index.html"></head>
<body><p>Redirecting to <a href="/ui/index.html">PathSnap Evaluator Dashboard</a>...</p></body>
</html>"""
