"""Verify release file digests without ML dependencies."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf8"))
errors = []
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file():
        errors.append(item["path"] + ": missing")
        continue
    content = path.read_bytes()
    if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
        errors.append(item["path"] + ": mismatch")
if errors:
    raise SystemExit("Integrity check FAILED\n" + "\n".join(errors))
print(f"Integrity check passed: {len(manifest['files'])} files")
