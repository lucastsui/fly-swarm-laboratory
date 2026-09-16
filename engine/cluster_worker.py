"""GPU actors with independent worlds and one remote, shared optimizer."""
import argparse
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .brain import Brain
from .cluster_protocol import ClusterClient, signature
from .factory import Factory
from .training import TrainingRuntime


class RemoteOptimizer:
    def __init__(self, worker):
        self.worker = worker

    def zero_grad(self, set_to_none=True):
        self.worker.brain.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.worker.submit_gradient()


class ClusterWorker(TrainingRuntime):
    def __init__(self, args):
        super().__init__(args.graph, args.storage, Factory)
        self.worker_id = args.worker
        self.metadata = json.loads(args.metadata.read_text())
        self.client = ClusterClient(args.coordinator, args.token_file.read_text().strip())
        self.pending = None
        self.last_contact = 0.0
        self.version = 0
        self.generation = None
        self.watch = {"environment": 0, "fly": 0}
        self.communication_ms = 0.0
        self.must_resync = False
        self.status = {"worker": self.worker_id, "ready": False}
        self.storage.mkdir(parents=True, exist_ok=True)

    def pull(self):
        result = self.client.tensor("/worker/parameters?worker=" + self.worker_id)
        self.accept_parameters(result)
        return result

    def accept_parameters(self, result):
        if result["signature"] != self.layout:
            raise ValueError("Worker and coordinator parameter layouts differ")
        flat = result["parameters"]
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError("Coordinator returned non-finite weights")
        # Copy into existing Parameters so all graph references remain valid.
        offset = 0
        with torch.no_grad():
            for parameter in self.brain.parameters():
                parameter.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
                offset += parameter.numel()
        if offset != flat.numel():
            raise ValueError("Wrong parameter vector size")
        self.version = result["version"]
        self.weight_hash = result["weightHash"]
        self.pending = result
        self.last_contact = time.monotonic()

    def submit_gradient(self):
        began = time.perf_counter()
        gradient = torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
                              for p in self.brain.parameters()]).cpu()
        message = {"worker": self.worker_id, "id": self.worker_id + "-" + uuid.uuid4().hex,
                   "signature": self.layout, "version": self.version, "generation": self.generation,
                   "gradient": gradient, "samples": min(len(self.rollout) * self.agent_count, max(32, self.agent_count))}
        self.accept_parameters(self.client.tensor("/worker/update", message))
        duration = 1000 * (time.perf_counter() - began)
        self.communication_ms = .8 * self.communication_ms + .2 * duration if self.communication_ms else duration

    def apply_controls(self, controls):
        if controls["generation"] != self.generation:
            if self.generation is not None:
                self._save(force=True)
                previous = self.storage / "worker-state.pt"
                previous.replace(self.storage / ("worker-state-" + self.generation + ".pt"))
            self.set_running(False)
            self.generation = controls["generation"]
            self.environment_count = controls["config"]["environmentCount"]
            self.fly_count = controls["config"]["flyCount"]
            self.validate_settings(self.fly_count, self.environment_count, controls["speed"])
            seed = int(hashlib.sha256((self.worker_id + self.generation).encode()).hexdigest()[:8], 16)
            self.rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            self._new_worlds(seed)
            self.steps, self.loss, self.active_seconds = 0, 0.0, 0.0
            self.rollout.clear()
            self.rate_samples.clear()
            saved_path = self.storage / "worker-state.pt"
            if saved_path.exists():
                saved = torch.load(saved_path, map_location="cpu", weights_only=True)
                if saved["generation"] == self.generation and saved["state"].shape == self.state.shape:
                    self.factory.restore(saved["factory"])
                    self.state.copy_(saved["state"])
                    self.steps, self.active_seconds = self.factory.steps, saved["seconds"]
            self.last_snapshot = 0
        was_running = self.running
        if self.learning != controls["learning"]:
            self.rollout.clear()
        self.learning, self.unlimited = controls["learning"], controls["unlimited"]
        self.target_speed = controls["speed"]
        self.watch = {"environment": min(controls["watch"]["environment"], self.environment_count - 1),
                      "fly": min(controls["watch"]["fly"], self.fly_count - 1)}
        self.set_running(controls["running"])
        if was_running and not self.running:
            self.rollout.clear()
            self._save(force=True)

    def _save(self, force=False):
        now = time.perf_counter()
        self.last_checkpoint = now
        if not self.ready or (not force and now - getattr(self, "saved_at", 0) < 120):
            return
        # The optimizer/model are checkpointed centrally. Stale partial rollouts
        # are deliberately discarded on reconnect; worlds and recurrent states survive.
        value = {"generation": self.generation, "version": self.version, "state": self.state.cpu(),
                 "factory": self.factory.checkpoint(), "seconds": self.elapsed()}
        temporary = self.storage / "worker-state.pt.tmp"
        torch.save(value, temporary)
        temporary.replace(self.storage / "worker-state.pt")
        self.saved_at = now

    def publish_snapshot(self):
        now = time.perf_counter()
        if self.factory is None or now - self.last_snapshot < .5:
            return
        environment, fly = self.watch["environment"], self.watch["fly"]
        selected = environment * self.fly_count + fly
        with torch.no_grad():
            activity = self.state[:, selected].index_select(0, self.brain_view_indices).cpu().tolist()
            means = self.state.abs().mean(dim=0).cpu().tolist()
            active = float((self.state[:, selected].abs() > .05).float().mean().cpu())
        worlds = []
        for index, world in enumerate(self.factory.worlds):
            flies = [{**{key: f[key] for key in ("id", "x", "y", "cargo", "action", "reward", "deliveries")},
                      "activity": means[index * self.fly_count + f["id"]]} for f in world.flies]
            worlds.append({"flies": flies, "stations": world.stations(), "events": list(world.events),
                           "selectedFactory": {"index": index, "products": world.products,
                                               "deliveries": world.deliveries, "reward": world.total_reward}})
        rate = 0.0
        if self.running and len(self.rate_samples) > 1:
            first, last = self.rate_samples[0], self.rate_samples[-1]
            rate = (last[1] - first[1]) * self.agent_count / max(last[0] - first[0], 1e-6)
        self.status = {"worker": self.worker_id, "generation": self.generation, "ready": self.ready,
                       "running": self.running, "error": self.error, "version": self.version,
                       "gpu": torch.cuda.get_device_name(), "memoryGB": torch.cuda.memory_allocated() / 2**30,
                       "actionsPerSecond": rate, "communicationMs": self.communication_ms,
                       "steps": self.steps, "transitions": self.steps * self.agent_count,
                       "reward": self.factory.total_reward, "products": self.factory.products,
                       "deliveries": self.factory.deliveries, "loss": self.loss,
                       "snapshot": {"worlds": worlds, "environment": environment, "fly": fly,
                                    "activity": activity, "activeFraction": active}}
        self.last_snapshot = now

    def heartbeat_loop(self):
        client = ClusterClient(self.client.url, self.client.token)
        while True:
            try:
                self.pending = client.json("/worker/heartbeat", self.status)
                self.last_contact = time.monotonic()
            except Exception:
                pass  # Training loop suspends within five seconds of connection loss.
            time.sleep(.5)

    def run(self):
        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; never substitute a smaller brain")
        meta = json.loads((self.graph / "metadata.json").read_text())
        if (meta.get("neurons"), meta.get("edges")) != (166700, 25582938):
            raise ValueError("Wrong connectome")
        self.brain = Brain(self.graph, 22)
        self.layout = signature(self.brain)
        self.optimizer = RemoteOptimizer(self)
        self.brain_view_indices = torch.tensor(self.metadata["indices"], device=self.device)
        self.apply_controls(self.pull())
        self.ready = True
        self.publish_snapshot()
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        print(f"{self.worker_id} ready: {self.environment_count} factories × {self.fly_count} flies; shared version {self.version}", flush=True)
        while True:
            began = time.perf_counter()
            try:
                if time.monotonic() - self.last_contact > 5:
                    self.must_resync = True
                    self.set_running(False)
                    self.error = "Coordinator connection lost; waiting to reconnect"
                else:
                    if self.must_resync:
                        self.rollout.clear()
                        self.pull()
                        self.must_resync, self.error = False, None
                    if self.pending:
                        self.apply_controls(self.pending)
                    if self.running:
                        self.tick()
                self.publish_snapshot()
            except Exception as exc:
                self.must_resync = True
                self.set_running(False)
                self.error = f"Worker suspended: {type(exc).__name__}: {exc}"
                print(self.error, flush=True)
                self.rollout.clear()
                time.sleep(1)
            delay = 0 if self.running and self.unlimited else 1 / self.target_speed if self.running else .05
            time.sleep(max(0, delay - (time.perf_counter() - began)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--coordinator", required=True)
    ClusterWorker(parser.parse_args()).run()


if __name__ == "__main__":
    main()
