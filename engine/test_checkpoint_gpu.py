"""GPU checkpoint round-trip using disposable state, never the live run."""
import json
import shutil
import tempfile
from pathlib import Path

import torch

from .service import SwarmEngine


def main():
    baseline = Path(__file__).resolve().parents[2] / "spark-benchmark" / "checkpoint-benchmark.pt"
    with tempfile.TemporaryDirectory(prefix="swarm-checkpoint-test-") as folder:
        storage = Path(folder)
        shutil.copy2(baseline, storage / "brain-weights.pt")
        original = SwarmEngine()
        original.storage = storage
        original.load()
        assert original.ready, original.error
        for _ in range(27):
            original.tick()
        original._save()
        restored = SwarmEngine()
        restored.storage = storage
        restored.load()
        assert restored.ready, restored.error
        assert restored.steps == original.steps == 27
        assert restored.updates == original.updates == 1
        assert restored.factory.checkpoint() == original.factory.checkpoint()
        assert torch.equal(restored.state, original.state)
        assert len(restored.rollout) == len(original.rollout) == 3
        for a, b in zip(restored.brain.parameters(), original.brain.parameters()):
            assert torch.equal(a, b)
        assert restored.optimizer.state_dict()["param_groups"] == original.optimizer.state_dict()["param_groups"]
        for key, a in restored.optimizer.state_dict()["state"].items():
            for field, tensor in a.items():
                assert torch.equal(tensor.cpu(), original.optimizer.state_dict()["state"][key][field].cpu())
        view = restored.payload(7, 3)
        assert view["selectedFactory"]["index"] == 7 and view["selectedFly"] == 3
        assert view["brainView"]["activity"] == restored.state.index_select(0, restored.brain_view_indices)[:, 31].cpu().tolist()
        original.set_running(True)
        original.set_running(False)
        elapsed = original.elapsed()
        original.set_running(True)
        original.set_running(False)
        assert original.elapsed() >= elapsed
        restored.reset(2, 50, 2, True)
        assert restored.steps == 0 and restored.agent_count == 4 and restored.updates == 1
        assert any(storage.joinpath("archives").glob("*/training-state.pt"))
        print(json.dumps({"checkpointRoundTrip": "pass", "worlds": 8, "fliesPerWorld": 4,
                          "recurrentState": "exact", "weights": "exact", "optimizer": "exact",
                          "partialRollout": "restored", "selectedFlyActivity": "exact", "resetArchive": "pass"}), flush=True)


if __name__ == "__main__":
    main()
