"""
Main entry point for running the PathSnap FastAPI server on Laptop 1.
Usage:
    python main.py --host 0.0.0.0 --port 8000
"""

import argparse
import uvicorn
from src.api.server import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PathSnap Distributed Map-Matching Backend")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    print("=" * 70)
    print("Starting PathSnap Distributed Map-Matching & Route Reconstruction API")
    print(f"Server binding to: http://{args.host}:{args.port}")
    print(f"Evaluator UI:      http://localhost:{args.port}/ui/index.html")
    print("=" * 70)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
