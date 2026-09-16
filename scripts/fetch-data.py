"""Download exact public inputs; verify hashes; never overwrite mismatches."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
PROJECT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=PROJECT.parent/'connectome-data')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PROJECT/'artifacts/dataset.json').read_text())
    for name, item in manifest['inputs'].items():
        target = args.destination/name
        if target.exists():
            if sha(target) != item['sha256']:
                raise ValueError(f'Mismatched existing {target}; leaving it untouched')
            print('Verified', name)
            continue
        partial = target.with_suffix(target.suffix+'.partial')
        if partial.exists():
            raise FileExistsError(f'Inspect previous partial download first: {partial}')
        print('Downloading', name, flush=True)
        with urllib.request.urlopen(item['url'], timeout=120) as response, partial.open('xb') as output:
            while block := response.read(1024*1024):
                output.write(block)
        if sha(partial) != item['sha256']:
            raise ValueError(f'Checksum mismatch; retained for inspection: {partial}')
        partial.rename(target)
        print('Verified', name)

if __name__ == '__main__':
    main()
