import json
from pathlib import Path
from .utils import atomic_write_json

MODELS_DIR = Path("models")
REGISTRY = MODELS_DIR / "registry.json"

def load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"stable": None, "candidate": None, "history": []}

def set_candidate(version: str):
    reg = load_registry()
    reg["candidate"] = version
    reg["history"].insert(0, {"version": version, "stage": "candidate"})
    atomic_write_json(str(REGISTRY), reg)

def promote_to_stable(version: str):
    reg = load_registry()
    reg["stable"] = version
    reg["history"].insert(0, {"version": version, "stage": "stable"})
    atomic_write_json(str(REGISTRY), reg)
