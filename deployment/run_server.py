#!/usr/bin/env python3
"""
Production server runner using Gunicorn
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Run the recommendation server"""

    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    sys.path.insert(0, os.getcwd())

    # Check if model and data files exist
    model_path = "./models/improved_svd_model.pkl"
    data_path = "./data/silver/interactions.parquet"

    # Set port
    port = os.getenv('PORT', 8082)

    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}")
        print("Make sure to download the SVD model from your teammate")

    if not os.path.exists(data_path):
        print(f"WARNING: Data file not found at {data_path}")
        print("Make sure interactions.parquet is in the data directory")

    # Production server with Gunicorn
    cmd = [
        "gunicorn",
        "--bind", f"0.0.0.0:{port}",
        "--workers", "2",
        "--timeout", "30",
        "--keep-alive", "2",
        "--max-requests", "1000",
        "--max-requests-jitter", "100",
        "--preload",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--chdir", "../",
        "ml_pipeline.serve.app:app"
    ]

    print(f"Starting recommendation server...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Server will be available at http://0.0.0.0:{port}")
    print(f"Test endpoint: http://0.0.0.0:{port}/recommend/12345")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()