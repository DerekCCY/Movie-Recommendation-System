# serve/provenance_logging.py
import json
import os
import time
from pathlib import Path

LOG_DIR = Path(os.getenv("PRED_LOG_DIR", "logs/predictions"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_prediction_log(record: dict) -> None:
    """
    Append one JSON line per prediction into logs/predictions/YYYY-MM-DD.jsonl
    """
    # Use UTC date for filename so it matches server Date/ts
    day = time.strftime("%Y-%m-%d", time.gmtime())
    path = LOG_DIR / f"{day}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
