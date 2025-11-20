# --- add in automated_retraining.py ---
import json, os, shutil
from datetime import datetime
from pathlib import Path
from .utils import sha256_file, atomic_write_json
from .registry import set_candidate

MODELS_DIR = Path("models")

def save_versioned_artifacts(results: dict, latest_model_path: str) -> str:
    """Create models/recsys-YYYYMMDD-HHMM with model & 3 JSONs, return version."""
    version = "recsys-" + datetime.utcnow().strftime("%Y%m%d-%H%M")
    ver_dir = MODELS_DIR / version
    ver_dir.mkdir(parents=True, exist_ok=True)

    # copy model
    model_filename = Path(latest_model_path).name
    ver_model_path = ver_dir / model_filename
    shutil.copy(latest_model_path, ver_model_path)
    artifact_hash = sha256_file(str(ver_model_path))

    # training_data_manifest.json
    time_window = results.get("time_window", {})
    manifest = {
        "time_window": {
            "start": time_window.get("start", "unknown"),
            "end":   time_window.get("end", "unknown")
        },
        "snapshots": results.get("training_data_snapshots", [])
    }
    (ver_dir / "training_data_manifest.json").write_text(json.dumps(manifest, indent=2))

    # feature_schema.json 
    #feature_schema = {
    #    "version": os.getenv("FEATURE_SCHEMA_VERSION", "feat-1.0.0"),
    #    "columns": [{"name": "user_id", "dtype": "int"},
    #                {"name": "movie_id", "dtype": "int"}],
    #    "schema_fingerprint": "sha256:placeholder"
    #}
    #(ver_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2))

    # model_card.json
    model_card = {
        "model_version": version,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_commit": results.get("training_commit", "unknown"),
        "training_data_manifest": "training_data_manifest.json",
        "metrics_offline": results.get("evaluation_metrics", {}),
        "hyperparams": results.get("hyperparams", {}),
        "artifact_hash": artifact_hash
    }
    (ver_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))
    return version
