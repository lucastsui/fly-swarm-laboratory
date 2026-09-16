"""Audit packaged morphology integrity and source branch topology."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np

project = Path(__file__).resolve().parents[1]
root = project / "public" / "anatomy"
manifest = json.loads((root / "manifest.json").read_text())
neurons = manifest["neurons"]
vertices = segments = 0
source_ids = set(n["id"] for n in neurons)
assert len(source_ids) == manifest["neuronCount"]
assert manifest["maxDeviationUm"] <= manifest["simplificationToleranceUm"] == .2
assert not manifest["unavailable"]
audit = set(np.linspace(0, len(neurons)-1, 24).astype(int).tolist())
audit.add(max(range(len(neurons)), key=lambda i: neurons[i]["sourceVertices"]))
branch_audits = 0
for chunk in manifest["chunks"]:
    path = root / Path(chunk["url"]).name
    packed = path.read_bytes()
    assert len(packed) == chunk["bytes"]
    assert hashlib.sha256(packed).hexdigest() == chunk["sha256"]
    raw = gzip.decompress(packed)
    magic, version, count, edge_count = struct.unpack("<4sIII", raw[:16])
    assert magic == b"MCNS" and version == 1
    assert count == chunk["vertices"] and edge_count == chunk["segments"]
    assert len(raw) == 16 + count*16 + edge_count*8
    xyz = np.frombuffer(raw, dtype="<f4", offset=16, count=count*3).reshape(-1, 3)
    owner = np.frombuffer(raw, dtype="<u4", offset=16+count*12, count=count)
    links = np.frombuffer(raw, dtype="<u4", offset=16+count*16, count=edge_count*2).reshape(-1, 2)
    assert np.isfinite(xyz).all() and int(owner.max()) < len(neurons)
    assert int(links.max()) < count and (links[:, 0] != links[:, 1]).all()
    assert (owner[links[:, 0]] == owner[links[:, 1]]).all(), "Fabricated cross-neuron connection"
    parents = np.full(count, -1, dtype=np.int64)
    assert len(np.unique(links[:, 0])) == len(links), "Multiple parents in rendered forest"
    parents[links[:, 0]] = links[:, 1]
    children = np.bincount(links[:, 1], minlength=count)
    anchors = (children != 1) | (parents == -1)
    for index in set(int(v) for v in np.unique(owner)) & audit:
        neuron = neurons[index]
        source = (project.parent / "connectome-data" / "skeletons-swc" / f"{neuron['id']}.swc").read_bytes()
        assert hashlib.sha256(source).hexdigest() == neuron["sourceSha256"]
        swc = np.loadtxt(io.BytesIO(source), comments="#", ndmin=2)
        lookup = {int(node): i for i, node in enumerate(swc[:, 0])}
        source_parents = np.array([lookup.get(int(p), -1) for p in swc[:, 6]])
        source_children = np.bincount(source_parents[source_parents>=0], minlength=len(swc))
        source_anchors = (source_children != 1) | (source_parents == -1)
        expected_xyz = (swc[:, 2:5]*.008).astype(np.float32)
        # All retained coordinates are original source vertices, with identical
        # root/fork/tip coordinates and child counts after simplification.
        retained = xyz[owner == index]
        assert set(map(tuple, retained)).issubset(set(map(tuple, expected_xyz)))
        expected = sorted((tuple(p), int(c), bool(r)) for p,c,r in zip(expected_xyz[source_anchors], source_children[source_anchors], source_parents[source_anchors] < 0))
        actual_anchor = anchors & (owner == index)
        actual = sorted((tuple(p), int(c), bool(r)) for p,c,r in zip(xyz[actual_anchor], children[actual_anchor], parents[actual_anchor] < 0))
        assert expected == actual, f"Branch topology changed for {neuron['id']}"
        branch_audits += 1
    vertices += count
    segments += edge_count
assert vertices == manifest["vertexCount"] and segments == manifest["segmentCount"]
with urllib.request.urlopen("http://127.0.0.1:8767/api/engine/state", timeout=10) as response:
    state = json.load(response)
live = {n["id"] for n in state["brainView"]["nodes"]}
assert live <= source_ids, "Live model IDs missing from anatomy"
print(json.dumps({"neurons":len(neurons),"vertices":vertices,"segments":segments,
                  "sourceBranchAudits":branch_audits,"liveModelIdsMatched":len(live),
                  "maxDeviationUm":manifest["maxDeviationUm"],"compressedMB":round(manifest["compressedBytes"]/1e6,2),
                  "engineRunning":state["running"],"engineFlyCount":state["flyCount"]}))
