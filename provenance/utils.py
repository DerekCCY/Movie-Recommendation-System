import hashlib, json, os, tempfile

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return "sha256:" + h.hexdigest()

def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())

def atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp.")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
