"""Run a short full-graph GPU throughput and gradient smoke test."""
import argparse
import json
import time
from pathlib import Path

import torch

from .brain import Brain

parser = argparse.ArgumentParser()
parser.add_argument("--flies", type=int, default=4)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--root", type=Path, default=Path(__file__).parents[2] / "connectome-data")
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("The selected runtime cannot access CUDA.")

torch.manual_seed(7)
brain = Brain(args.root, observations=22, device="cuda")
observation = torch.zeros((args.flies, 22), device="cuda")
state = brain.initial_state(args.flies)
with torch.no_grad():
    for _ in range(2):
        logits, values, state = brain(observation, state)
torch.cuda.synchronize()
state = state.detach()
start = time.perf_counter()
for _ in range(args.steps):
    logits, values, state = brain(observation, state)
    state = state.detach()
torch.cuda.synchronize()
forward_seconds = time.perf_counter() - start
loss = logits.square().mean() + values.square().mean()
start = time.perf_counter()
loss.backward()
torch.cuda.synchronize()
backward_seconds = time.perf_counter() - start
print(json.dumps({
    "device": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "neurons": brain.n,
    "directed_connections": int(brain.wiring.values().numel()),
    "flies_per_forward": args.flies,
    "timed_forward_passes": args.steps,
    "forward_seconds": round(forward_seconds, 3),
    "forward_passes_per_second": round(args.steps / forward_seconds, 2),
    "gradient_seconds": round(backward_seconds, 3),
    "logit_shape": list(logits.shape),
    "cuda_peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    "neural_activity_fraction": round(float((state.abs() > 0.05).float().mean().item()), 6),
    "scope": "Complete graph loaded; throughput smoke test, not a trained behavior result."
}, indent=2))
