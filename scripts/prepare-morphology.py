"""Build a browser-sized, provenance-preserving sample of MaleCNS SWC trees.

Raw SWCs are cached outside the Site. Original branch paths are simplified
with a verified 0.2 micrometer maximum deviation; branch topology is retained.
Run with the existing fly-runtime Python (numpy / pyarrow).
"""
from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import urllib.request

import numpy as np
import pyarrow.feather as feather

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "connectome-data"
CACHE = DATA / "skeletons-swc"
OUTPUT = PROJECT / "public" / "anatomy"
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/"
TOLERANCE_UM = .2
QUOTAS = {
    "ol_intrinsic": 760, "cb_intrinsic": 600, "vnc_intrinsic": 300,
    "visual_projection": 160, "descending_neuron": 140,
    "ascending_neuron": 100, "visual_centrifugal": 60,
    "ol_sensory": 60, "cb_sensory": 40, "vnc_sensory": 50,
    "vnc_motor": 30,
}


def select(annotations, body_ids):
    rng = np.random.default_rng(260912)
    annotations = annotations.reindex(body_ids)
    locations = np.full((len(body_ids), 3), np.nan, dtype=np.float32)
    for i, value in enumerate(annotations.somaLocation):
        if value is not None and len(value) == 3:
            locations[i] = value
    located = np.isfinite(locations).all(axis=1)
    # Match the running service's deterministic live sample, without stopping it.
    with np.load(DATA / "graph.npz") as graph:
        sensory, motor = graph["sensory"], graph["motor"]
    excluded = np.zeros(len(body_ids), dtype=bool)
    excluded[sensory] = excluded[motor] = True
    central = np.flatnonzero(~excluded)
    live = []
    for pool, quota in ((sensory, 96), (motor, 96), (central, 448)):
        pool = pool[located[pool]]
        live.extend(rng.choice(pool, min(quota, len(pool)), replace=False).tolist())
    live_ids = set(int(body_ids[i]) for i in live)
    selected = set(live_ids)
    # A deterministic type / side / neuromere-stratified sample gives coverage
    # to small compartments as well as the abundant optic-lobe cells.
    for superclass, quota in QUOTAS.items():
        pool = annotations[annotations.superclass == superclass]
        buckets = {}
        for body, row in pool.iterrows():
            if int(body) in selected:
                continue
            key = (str(row.type), str(row.somaSide), str(row.somaNeuromere))
            buckets.setdefault(key, []).append(int(body))
        keys = sorted(buckets)
        rng.shuffle(keys)
        for values in buckets.values():
            rng.shuffle(values)
        added = 0
        while keys and added < quota:
            next_keys = []
            for key in keys:
                if added >= quota:
                    break
                selected.add(buckets[key].pop())
                added += 1
                if buckets[key]:
                    next_keys.append(key)
            keys = next_keys
    return sorted(selected), live_ids


def fetch(body):
    path = CACHE / f"{body}.swc"
    if path.exists():
        return body, path.read_bytes(), None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(BASE + path.name, timeout=35) as response:
                raw = response.read()
            # A successful HTTP response must also be a nonempty SWC table.
            if not raw or b"<html" in raw[:200].lower():
                raise ValueError("Invalid SWC response")
            path.write_bytes(raw)
            return body, raw, None
        except Exception as error:
            if attempt == 2:
                return body, None, str(error)
            time.sleep(attempt + 1)


def simplify_tree(xyz, links):
    """RDP within each unbranched chain, retaining roots, forks, and tips.

    No new coordinates are introduced. Every shortcut follows a single source
    chain and all its source vertices are <= TOLERANCE_UM from the new segment.
    """
    parents = np.full(len(xyz), -1, dtype=np.int64)
    for child, parent in links:
        parents[child] = parent
    children = np.bincount(parents[parents >= 0], minlength=len(xyz))
    anchors = (children != 1) | (parents == -1)
    retained = set(np.flatnonzero(anchors).tolist())
    simplified = []
    max_error = 0.
    visited_edges = 0
    for end in np.flatnonzero(anchors):
        if parents[end] < 0:
            continue
        chain = [int(end)]
        node = int(parents[end])
        while True:
            chain.append(node)
            if anchors[node]:
                break
            node = int(parents[node])
        visited_edges += len(chain) - 1
        points = xyz[chain].astype(np.float64)
        stack = [(0, len(chain) - 1)]
        while stack:
            start, stop = stack.pop()
            error = 0.
            if stop > start + 1:
                delta = points[stop] - points[start]
                length_sq = float(delta @ delta)
                inner = points[start+1:stop]
                t = np.clip((inner-points[start]) @ delta / max(length_sq, 1e-20), 0., 1.)
                distances = np.linalg.norm(inner - (points[start] + t[:, None]*delta), axis=1)
                furthest = int(np.argmax(distances)) + start + 1
                error = float(distances.max())
                if error > TOLERANCE_UM:
                    stack.extend(((start, furthest), (furthest, stop)))
                    continue
            retained.update((chain[start], chain[stop]))
            simplified.append((chain[start], chain[stop]))
            max_error = max(max_error, error)
    if visited_edges != len(links):
        raise ValueError("Skeleton is not a rooted forest")
    selected = np.asarray(sorted(retained))
    mapping = np.full(len(xyz), -1, dtype=np.int64)
    mapping[selected] = np.arange(len(selected))
    return xyz[selected], mapping[np.asarray(simplified, dtype=np.int64)], max_error


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    annotations = feather.read_feather(DATA / "annotations.feather").set_index("bodyId")
    with np.load(DATA / "graph.npz") as graph:
        body_ids = graph["body_ids"]
    selected, live = select(annotations, body_ids)
    print(f"Fetching {len(selected)} released neuron skeletons; {len(live)} live IDs", flush=True)
    records, failures = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        for done, (body, raw, error) in enumerate(pool.map(fetch, selected), 1):
            if error:
                failures.append({"bodyId": body, "error": error})
            else:
                records[body] = raw
            if done % 100 == 0:
                print(f"Downloaded {done}/{len(selected)} ({len(failures)} unavailable)", flush=True)
    neurons, chunks = [], []
    positions, owners, edges = [], [], []
    vertex_count = 0
    total_vertices = total_edges = total_bytes = source_vertices = source_edges = 0
    max_deviation = 0.
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)

    def flush():
        nonlocal positions, owners, edges, vertex_count, total_bytes
        if not vertex_count:
            return
        xyz = np.concatenate(positions).astype("<f4")
        ids = np.concatenate(owners).astype("<u4")
        links = np.concatenate(edges).astype("<u4")
        # Header: magic, version, vertex count, edge count. Then tightly packed
        # float32 XYZ micrometers, uint32 cell indices, uint32 edge endpoint pairs.
        binary = struct.pack("<4sIII", b"MCNS", 1, len(xyz), len(links))
        binary += xyz.tobytes() + ids.tobytes() + links.tobytes()
        compressed = gzip.compress(binary, compresslevel=6, mtime=0)
        digest = hashlib.sha256(compressed).hexdigest()
        filename = f"morphology-{len(chunks):02d}-{digest[:10]}.bin.gz"
        (OUTPUT / filename).write_bytes(compressed)
        chunks.append({"url": "/anatomy/" + filename, "vertices": len(xyz),
                       "segments": len(links), "bytes": len(compressed), "sha256": digest})
        total_bytes += len(compressed)
        positions, owners, edges, vertex_count = [], [], [], 0

    for body in selected:
        if body not in records:
            continue
        raw = records[body]
        table = np.loadtxt(io.BytesIO(raw), comments="#", ndmin=2)
        if table.shape[1] != 7 or len(table) < 2 or not np.isfinite(table).all():
            raise ValueError(f"Invalid SWC structure for {body}")
        node_ids = table[:, 0].astype(np.int64)
        if len(np.unique(node_ids)) != len(node_ids):
            raise ValueError(f"Duplicate SWC node ID in {body}")
        id_to_index = {node: i for i, node in enumerate(node_ids)}
        links = []
        for child, parent in enumerate(table[:, 6].astype(np.int64)):
            if parent == -1:
                continue
            if parent not in id_to_index or parent == node_ids[child]:
                raise ValueError(f"Invalid parent reference in {body}")
            links.append((child, id_to_index[parent]))
        # SWCs are in 8 nm units. Preserve absolute XYZ and use a uniform unit conversion.
        xyz = (table[:, 2:5] * .008).astype(np.float32)
        original_vertices, original_edges = len(xyz), len(links)
        xyz, links, error_um = simplify_tree(xyz, links)
        max_deviation = max(max_deviation, error_um)
        source_vertices += original_vertices
        source_edges += original_edges
        row = annotations.loc[body]
        soma = row.somaLocation
        neurons.append({"id": body, "type": str(row.type) if row.type is not None else "Unassigned",
                        "superclass": str(row.superclass), "side": str(row.somaSide),
                        "soma": [round(float(x) * .008, 4) for x in soma] if soma is not None else None,
                        "live": body in live, "vertices": len(xyz), "segments": len(links),
                        "sourceVertices": original_vertices, "sourceSegments": original_edges,
                        "maxDeviationUm": error_um,
                        "sourceSha256": hashlib.sha256(raw).hexdigest()})
        minimum = np.minimum(minimum, xyz.min(axis=0))
        maximum = np.maximum(maximum, xyz.max(axis=0))
        positions.append(xyz)
        owners.append(np.full(len(xyz), len(neurons) - 1, dtype=np.uint32))
        edges.append(np.asarray(links, dtype=np.uint32).reshape(-1, 2) + vertex_count)
        vertex_count += len(xyz)
        total_vertices += len(xyz)
        total_edges += len(links)
        if vertex_count >= 420_000:
            flush()
    flush()
    manifest = {
        "version": 1, "dataset": "MaleCNS v1.0", "coordinateSpace": "Male CNS EM, unmirrored",
        "units": "micrometers", "sourceCoordinateUnitNm": 8,
        "source": "https://male-cns.janelia.org/download/", "swcBaseUrl": BASE,
        "attribution": "MaleCNS collaboration: FlyEM / HHMI Janelia, Cambridge, MRC LMB, Google Research",
        "license": "CC BY 4.0", "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
        "method": "Deterministic superclass/type/side/neuromere sample plus all current live display IDs. RDP simplification within unbranched source paths at 0.2 micrometer tolerance; original roots, forks and tips retained. No smoothing, warping or cross-neuron links.",
        "sourceVertexCount": source_vertices, "sourceSegmentCount": source_edges,
        "maxDeviationUm": max_deviation, "simplificationToleranceUm": TOLERANCE_UM,
        "representation": "Released centerline skeletons; not surface meshes or individual synapse locations. Soma markers show annotated positions with a uniform display size.",
        "neurons": neurons, "chunks": chunks, "bounds": [minimum.tolist(), maximum.tolist()],
        "vertexCount": total_vertices, "segmentCount": total_edges,
        "neuronCount": len(neurons), "liveNeuronCount": sum(n["live"] for n in neurons),
        "compressedBytes": total_bytes, "unavailable": failures,
        "fullGraphNeurons": len(body_ids),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("neuronCount", "liveNeuronCount", "vertexCount", "segmentCount", "compressedBytes", "bounds")}), flush=True)


if __name__ == "__main__":
    main()
