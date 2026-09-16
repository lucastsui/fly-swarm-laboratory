"""Read-only full-graph before/after throughput and gradient comparison."""
import argparse
import ast
import importlib.util
import json
import math
import random
import statistics
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .brain import Brain
from .parallel import ParallelFactories
from .service import Factory, SwarmEngine


def load_baseline(folder):
    spec = importlib.util.spec_from_file_location("baseline_brain", folder / "brain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse((folder / "service.py").read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ("Factory", "SwarmEngine")]
    namespace = dict(globals(), OBSERVATIONS=22, ACTION_COUNT=6, UPDATE_INTERVAL=24, LEARNING_BATCH=32)
    exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + classes, type_ignores=[]), "baseline", "exec"), namespace)
    return module.Brain, namespace["SwarmEngine"]


def load_weights(brain, path):
    for name, value in torch.load(path, map_location="cuda", weights_only=True).items():
        target = getattr(brain, name)
        target.load_state_dict(value) if isinstance(target, torch.nn.Module) else target.data.copy_(value)


def optimizer(brain):
    return torch.optim.AdamW([
        {"params": brain.encoder.parameters(), "lr": 1e-4},
        {"params": [brain.log_gain, brain.bias], "lr": 1e-5},
        {"params": list(brain.readout.parameters()) + list(brain.actor.parameters()) + list(brain.critic.parameters()), "lr": 1e-4},
    ], weight_decay=1e-5)


def compare_math(old, new):
    torch.manual_seed(41)
    obs = torch.randn(3, 22, device="cuda") * .1
    states = torch.randn(new.n, 3, device="cuda") * .03
    results = []
    for brain in (old, new):
        brain.zero_grad(set_to_none=True)
        x, s = obs.clone().requires_grad_(), states.clone().requires_grad_()
        logits, value, activity = brain(x, s)
        loss = logits.square().mean() + value.square().mean() + activity.square().mean()
        loss.backward()
        results.append({"output": torch.cat([logits.flatten(), value, activity.flatten()]).detach(),
                        "observationGradient": x.grad, "stateGradient": s.grad,
                        **{name: param.grad.detach().clone() for name, param in brain.named_parameters()}})
    errors = {}
    for name in results[0]:
        a, b = results[0][name], results[1][name]
        errors[name] = float((a - b).abs().max())
        if not torch.allclose(a, b, atol=2e-6, rtol=2e-4):
            raise AssertionError(f"Forward/gradient mismatch: {name}: {errors[name]}")
    with torch.no_grad():
        together = new(obs, states)[2]
        separately = torch.cat([new(obs[i:i+1], states[:, i:i+1])[2] for i in range(3)], dim=1)
        if not torch.allclose(together, separately, atol=2e-5, rtol=2e-4):
            raise AssertionError("Batched fly states are not independent")
        errors["columnIndependence"] = float((together - separately).abs().max())
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=12)
    args = parser.parse_args()
    old_brain_class, old_engine_class = load_baseline(args.baseline)
    old = old_brain_class(args.graph, 22)
    new = Brain(args.graph, 22)
    load_weights(old, args.checkpoint)
    load_weights(new, args.checkpoint)
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "neurons": new.n, "edges": new.wiring._nnz(), "dtype": str(new.wiring.dtype),
              "validation": compare_math(old, new), "results": [],
              "scope": "Uncapped training, same checkpoint, reward rules, 24-tick updates and 32-sample minibatch. Throughput is not proof of learned coordination."}
    print(json.dumps({"validation": result["validation"]}), flush=True)
    for environments, flies in ((1, 1), (8, 4)):
        for label, brain, engine_class in (("before", old, old_engine_class), ("after", new, SwarmEngine)):
            torch.manual_seed(123)
            load_weights(brain, args.checkpoint)
            engine = engine_class()
            engine.brain = brain
            engine.optimizer = optimizer(brain)
            engine.environment_count = environments
            engine.fly_count = flies if label == "after" else environments * flies
            engine.factory = ParallelFactories(Factory, environments, flies)
            engine.state = brain.initial_state(environments * flies)
            engine.brain_view_indices = torch.arange(574, device="cuda")
            engine.last_checkpoint = time.perf_counter() + 1e9
            engine.ready = engine.running = True
            if label == "after":
                engine.resumed_at = time.perf_counter()
            for _ in range(24):
                engine.tick()
            timings = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.cycles):
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(24):
                    engine.tick()
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
            finite = all(bool(torch.isfinite(p.grad).all()) for p in brain.parameters() if p.grad is not None)
            if not finite:
                raise AssertionError("Nonfinite training gradients")
            item = {"version": label, "factories": environments, "fliesPerFactory": flies,
                    "factoryTicksPerSecond": 24 / statistics.median(timings),
                    "agentTransitionsPerSecond": 24 * environments * flies / statistics.median(timings),
                    "updatesPerSecond": 1 / statistics.median(timings),
                    "minibatchSamplesPerSecond": min(32, 24 * environments * flies) / statistics.median(timings),
                    "cycleSeconds": timings, "finiteGradients": finite,
                    "peakAllocatedGiB": torch.cuda.max_memory_allocated() / 2**30}
            result["results"].append(item)
            print(json.dumps(item), flush=True)
            del engine
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
