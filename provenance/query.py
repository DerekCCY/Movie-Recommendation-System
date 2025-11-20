# provenance/query.py
import argparse
import glob
import json
import os
from pathlib import Path


def _find_request(request_id: str, log_dir: str) -> tuple[dict | None, str | None]:
    for path in sorted(glob.glob(os.path.join(log_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if j.get("request_id") == request_id:
                    return j, path
    return None, None


def _read_model_card(models_dir: str, version: str) -> dict:
    p = Path(models_dir) / version / "model_card.json"
    with p.open() as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Provenance lookup")
    ap.add_argument("--request_id", required=True, help="request_id to lookup")
    ap.add_argument("--log_dir", default="logs/predictions", help="directory of prediction jsonl logs")
    ap.add_argument("--models_dir", default="models", help="models root directory")
    args = ap.parse_args()

    rec, src = _find_request(args.request_id, args.log_dir)
    assert rec, "request_id not found in prediction logs"

    card = _read_model_card(args.models_dir, rec["model_version"])

    print(f"Request {rec['request_id']} @ {rec['ts']} (log: {src})")
    print(f" model_version: {rec['model_version']}  image: {rec.get('serving_image')}")
    print(f" training_commit: {card.get('training_commit')}")
    print(f" training_data_manifest: {card.get('training_data_manifest')}")
    print(f" feature_schema_version: {rec.get('feature_schema_version')}")
    print(f" experiment: {rec.get('experiment_id')} arm={rec.get('ab_arm')}")
    print(f" latency_ms: {rec.get('latency_ms')}")


if __name__ == "__main__":
    main()
