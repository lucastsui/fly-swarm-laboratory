"""Freeze a current local checkpoint and prepare private cluster bootstrap data."""
import argparse
import json
import secrets
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.storage.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def api(path, value=None):
        request = urllib.request.Request("http://127.0.0.1:8767/api/engine/" + path,
                                         data=None if value is None else json.dumps(value).encode(),
                                         headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=30) as response:
            return json.load(response)
    before, began = api("state"), time.perf_counter()
    time.sleep(20)
    after, elapsed = api("state"), time.perf_counter() - began
    args.baseline.write_text(json.dumps({"seconds": elapsed, "actionsPerSecond": (after["transitions"] - before["transitions"]) / elapsed,
                                         "updatesPerSecond": (after["updates"] - before["updates"]) / elapsed,
                                         "before": {k: before[k] for k in ("runId", "steps", "updates", "transitions")},
                                         "after": {k: after[k] for k in ("runId", "steps", "updates", "transitions")}}, indent=2))
    api("control", {"action": "pause"})
    saved = torch.load(root / ".runtime" / "parallel" / "training-state.pt", map_location="cpu", weights_only=True)
    torch.save({k: saved[k] for k in ("weights", "optimizer", "updates")}, args.storage / "bootstrap.pt")
    view = after["brainView"]
    ids = np.load(args.graph / "graph.npz")["body_ids"]
    lookup = {int(body): index for index, body in enumerate(ids)}
    metadata = {k: view[k] for k in ("nodes", "edges", "bounds")}
    metadata["indices"] = [lookup[node["id"]] for node in view["nodes"]]
    metadata["obstacles"] = after["obstacles"]
    (args.storage / "brain-view.json").write_text(json.dumps(metadata))
    configs = {"spark1": {"label": "DGX Spark 1", "host": "spark1", "environmentCount": 16, "flyCount": 4},
               "spark2": {"label": "DGX Spark 2", "host": "spark2", "environmentCount": 16, "flyCount": 4}}
    (args.storage / "workers.json").write_text(json.dumps(configs))
    token = args.storage / "cluster-token"
    if not token.exists():
        token.write_text(secrets.token_hex(32))
    print(json.dumps({"pausedLocalRun": saved["runId"], "checkpointUpdates": saved["updates"],
                      "checkpointSteps": saved["factory"]["steps"], "monitoredCells": len(metadata["indices"]),
                      "bootstrapBytes": (args.storage / "bootstrap.pt").stat().st_size}))


if __name__ == "__main__":
    main()
