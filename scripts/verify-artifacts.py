"""Read-only verification; no simulation or training is started."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'artifacts/manifest.json').read_text())
failed = []
for item in manifest['files']:
    path = (root/item['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failed.append(item['path']+' (missing or invalid path)')
        continue
    with path.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != item['sha256'] or path.stat().st_size != item['bytes']:
        failed.append(item['path']+' (checksum/size mismatch)')
if failed:
    raise SystemExit('Artifact verification failed:\n'+'\n'.join(failed))
print(f"Verified {len(manifest['files'])} artifact files; no simulation or training started.")
