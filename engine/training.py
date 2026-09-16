"""Batched full-graph training; display snapshots never read GPU state on request."""
import json
import shutil
import threading
import time
import uuid
from collections import deque

import numpy as np
import torch

from .brain import Brain
from .parallel import ParallelFactories

UPDATE_INTERVAL = 24
LEARNING_BATCH = 32
MAX_AGENTS = 128


class TrainingRuntime:
    def __init__(self, graph, storage, factory_class):
        self.graph, self.storage, self.factory_class = graph, storage, factory_class
        self.lock = threading.RLock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ready = self.running = False
        self.learning = self.unlimited = True
        self.error = self.checkpoint = None
        self.target_speed, self.environment_count, self.fly_count = 50, 8, 4
        self.steps = self.updates = 0
        self.loss = 0.0
        self.run_id = uuid.uuid4().hex[:10]
        self.factory = self.brain = self.optimizer = self.state = None
        self.rng = np.random.default_rng(41)
        self.rollout = []
        self.last_checkpoint = time.perf_counter()
        self.last_snapshot = 0.0
        self.rate_samples = deque(maxlen=2000)
        self.active_seconds = 0.0
        self.resumed_at = None
        self.snapshot = {}
        self.thread = None

    @property
    def agent_count(self):
        return self.environment_count * self.fly_count

    @staticmethod
    def validate_settings(count, environments, speed):
        if not 1 <= count <= 32 or not 1 <= environments <= 32 or count * environments > MAX_AGENTS:
            raise ValueError("Use 1–32 factories and 1–32 flies per factory, with at most 128 flies in total.")
        if not 1 <= speed <= 200:
            raise ValueError("Capped speed must be 1–200 factory ticks per second.")

    def _new_worlds(self, seed=41):
        self.factory = ParallelFactories(self.factory_class, self.environment_count, self.fly_count, seed)
        self.state = self.brain.initial_state(self.agent_count)

    def _load_weights(self, learned):
        for name, value in learned.items():
            target = getattr(self.brain, name)
            if name in ("encoder", "readout", "actor", "critic"):
                target.load_state_dict(value)
            else:
                target.data.copy_(value)

    def load(self):
        try:
            meta = json.loads((self.graph / "metadata.json").read_text(encoding="utf-8"))
            if (meta.get("release"), meta.get("neurons"), meta.get("edges")) != ("MaleCNS v1.0", 166700, 25582938):
                raise ValueError("The graph does not match the complete MaleCNS v1.0 registry.")
            if self.device.type != "cuda":
                raise RuntimeError("A CUDA-capable GPU is required for this full-connectome build.")
            self.brain = Brain(self.graph, 22, device=str(self.device))
            self.optimizer = torch.optim.AdamW([
                {"params": self.brain.encoder.parameters(), "lr": 1e-4},
                {"params": [self.brain.log_gain, self.brain.bias], "lr": 1e-5},
                {"params": list(self.brain.readout.parameters()) + list(self.brain.actor.parameters()) + list(self.brain.critic.parameters()), "lr": 1e-4},
            ], weight_decay=1e-5)
            complete = self.storage / "training-state.pt"
            saved = torch.load(complete, map_location="cpu", weights_only=True) if complete.exists() else None
            if saved:
                if saved.get("schema") != 2:
                    raise ValueError("Unsupported training checkpoint format.")
                settings = saved["settings"]
                self.fly_count, self.environment_count = settings["flyCount"], settings["environmentCount"]
                self.target_speed, self.unlimited = settings["speed"], settings["unlimited"]
                self.learning = settings["learning"]
                self.validate_settings(self.fly_count, self.environment_count, self.target_speed)
                self._load_weights(saved["weights"])
                self.optimizer.load_state_dict(saved["optimizer"])
                self._new_worlds()
                self.factory.restore(saved["factory"])
                self.state = saved["state"].to(self.device)
                self.steps = self.factory.steps
                self.updates, self.loss, self.run_id = saved["updates"], saved["loss"], saved["runId"]
                self.active_seconds = saved["activeSeconds"]
                self.rng.bit_generator.state = saved["numpyRandom"]
                torch.set_rng_state(saved["torchRandom"].cpu())
                torch.cuda.set_rng_state(saved["cudaRandom"].cpu())
                self.rollout = saved["rollout"]
                # Rewards are the only CPU tensors in a stored rollout.
                for row in self.rollout:
                    for key in ("observation", "state", "value", "action"):
                        row[key] = row[key].to(self.device)
                    row["reward"] = row["reward"].numpy()
                self.checkpoint = time.strftime("%H:%M:%S", time.localtime(complete.stat().st_mtime))
            else:
                legacy = self.storage / "brain-weights.pt"
                if legacy.exists():
                    self._load_weights(torch.load(legacy, map_location=self.device, weights_only=True))
                    run_file = self.storage / "run.json"
                    if run_file.exists():
                        old = json.loads(run_file.read_text(encoding="utf-8"))
                        self.updates = int(old.get("updates", 0))
                    self._archive("legacy-before-parallel")
                # Legacy runs did not persist positions or recurrent state. Start
                # fresh factories; never pretend the old counters describe them.
                self._new_worlds()
            self._prepare_brain_view()
            self.ready = True
            self.publish_snapshot()
            print(f"Loaded {self.brain.n:,} neurons · {self.environment_count} factories × {self.fly_count} flies.", flush=True)
        except Exception as exc:
            self.error = f"Could not load the full local brain: {type(exc).__name__}: {exc}"
            print(self.error, flush=True)

    def set_running(self, running):
        if running == self.running:
            return
        now = time.perf_counter()
        if running:
            self.resumed_at = now
            self.rate_samples.clear()
            self.rate_samples.append((now, self.steps))
        elif self.resumed_at is not None:
            self.active_seconds += now - self.resumed_at
            self.resumed_at = None
        self.running = running

    def elapsed(self):
        return self.active_seconds + (time.perf_counter() - self.resumed_at if self.resumed_at is not None else 0)

    def _archive(self, label=None):
        destination = self.storage / "archives" / f"{label or self.run_id}-{time.time_ns()}"
        destination.mkdir(parents=True, exist_ok=False)
        for name in ("brain-weights.pt", "run.json", "training-state.pt"):
            source = self.storage / name
            if source.exists():
                shutil.copy2(source, destination / name)

    def reset(self, count, speed, environments=8, unlimited=True):
        self.validate_settings(count, environments, speed)
        self.set_running(False)
        self._save()
        self._archive()
        self.fly_count, self.environment_count, self.target_speed, self.unlimited = count, environments, speed, unlimited
        self._new_worlds(seed=self.steps + 41)
        self.rollout.clear()
        self.steps, self.loss, self.active_seconds = 0, 0.0, 0.0
        self.rate_samples.clear()
        self.run_id = uuid.uuid4().hex[:10]
        self.error = None
        self._save()

    def learn(self):
        if not self.learning or len(self.rollout) < 8:
            self.rollout.clear()
            return
        rows = self.rollout
        # One small transfer per update, not per step. Large recurrent states
        # stay on the GPU, in the same float16 rollout storage used previously.
        rewards = np.stack([row["reward"] for row in rows])
        values = torch.stack([row["value"] for row in rows]).cpu().numpy()
        advantages = np.zeros_like(rewards)
        next_advantage = np.zeros(self.agent_count, dtype=np.float32)
        next_value = np.zeros(self.agent_count, dtype=np.float32)
        for index in range(len(rows) - 1, -1, -1):
            delta = rewards[index] + .985 * next_value - values[index]
            next_advantage = delta + .985 * .92 * next_advantage
            advantages[index] = next_advantage
            next_value = values[index]
        batch_size = min(len(rows) * self.agent_count, max(LEARNING_BATCH, self.agent_count))
        selected = self.rng.choice(len(rows) * self.agent_count, batch_size, replace=False)
        times, agents = np.divmod(selected, self.agent_count)
        observations = torch.stack([rows[t]["observation"][a] for t, a in zip(times, agents)]).float()
        states = torch.stack([rows[t]["state"][:, a] for t, a in zip(times, agents)], dim=1).float()
        actions = torch.stack([rows[t]["action"][a] for t, a in zip(times, agents)])
        target_advantage = torch.as_tensor(advantages[times, agents], device=self.device)
        target_return = torch.as_tensor((advantages + values)[times, agents], device=self.device)
        target_advantage = (target_advantage - target_advantage.mean()) / target_advantage.std(unbiased=False).clamp_min(.03)
        self.optimizer.zero_grad(set_to_none=True)
        logits, estimate, _ = self.brain(observations, states)
        distribution = torch.distributions.Categorical(logits=logits, validate_args=False)
        loss = -(distribution.log_prob(actions) * target_advantage.detach()).mean()
        loss = loss + .5 * torch.nn.functional.smooth_l1_loss(estimate, target_return.detach()) - .012 * distribution.entropy().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss; previous checkpoint retained.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.loss = float(loss.detach().cpu())
        self.updates += 1
        self.rollout.clear()

    def tick(self):
        obs_array = np.stack([self.factory.observation(fly) for fly in self.factory.flies])
        before = np.asarray([self.factory._point_dist((f["x"], f["y"]), f["goal_xy"]) for f in self.factory.flies])
        obs = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)
        state_before = self.state.detach()
        with torch.no_grad():
            logits, values, self.state = self.brain(obs, state_before)
            actions = torch.distributions.Categorical(logits=logits, validate_args=False).sample()
        rewards = self.factory.advance(actions.cpu().numpy(), before)
        self.steps = self.factory.steps
        if self.learning:
            self.rollout.append({"observation": obs.half(), "state": state_before.half(),
                                 "action": actions, "value": values, "reward": rewards})
            if len(self.rollout) >= UPDATE_INTERVAL:
                self.learn()
        now = time.perf_counter()
        self.rate_samples.append((now, self.steps))
        while len(self.rate_samples) > 2 and self.rate_samples[0][0] < now - 5:
            self.rate_samples.popleft()
        if self.factory.history and self.factory.history[-1]["step"] == self.steps:
            self.factory.history[-1]["seconds"] = self.elapsed()
        if now - self.last_snapshot >= .2:
            self.publish_snapshot()
        if now - self.last_checkpoint > 45:
            self._save()

    def _save(self):
        if not self.ready:
            return
        learned = {name: getattr(self.brain, name).state_dict() for name in ("encoder", "readout", "actor", "critic")}
        learned.update(log_gain=self.brain.log_gain.detach(), bias=self.brain.bias.detach())
        settings = {"flyCount": self.fly_count, "environmentCount": self.environment_count,
                    "speed": self.target_speed, "unlimited": self.unlimited, "learning": self.learning}
        complete = {"schema": 2, "settings": settings, "weights": learned,
                    "optimizer": self.optimizer.state_dict(), "state": self.state,
                    "factory": self.factory.checkpoint(), "updates": self.updates, "loss": self.loss,
                    "runId": self.run_id, "activeSeconds": self.elapsed(),
                    "numpyRandom": self.rng.bit_generator.state, "torchRandom": torch.get_rng_state(),
                    "cudaRandom": torch.cuda.get_rng_state(),
                    "rollout": [{**row, "reward": torch.from_numpy(row["reward"])} for row in self.rollout]}
        # The authoritative checkpoint is one atomic file, including worlds,
        # optimizer and separate neural states. Compatibility exports follow it.
        for name, value in (("training-state.pt", complete), ("brain-weights.pt", learned)):
            temporary = self.storage / (name + ".tmp")
            torch.save(value, temporary)
            temporary.replace(self.storage / name)
        summary = {**settings, "runId": self.run_id, "steps": self.steps, "updates": self.updates,
                   "products": self.factory.products, "history": list(self.factory.history)}
        temporary = self.storage / "run.json.tmp"
        temporary.write_text(json.dumps(summary), encoding="utf-8")
        temporary.replace(self.storage / "run.json")
        self.checkpoint = time.strftime("%H:%M:%S")
        self.last_checkpoint = time.perf_counter()
        self.publish_snapshot()

    def publish_snapshot(self):
        if self.factory is None or self.brain_view_indices is None:
            return
        # Sampling at 5 Hz is independent of training speed. Keep each fly's
        # measured activity separate: averaging a swarm hides its dynamics.
        with torch.no_grad():
            sampled = self.state.index_select(0, self.brain_view_indices).T.cpu().tolist()
            summaries = torch.stack((self.state.abs().mean(dim=0), (self.state.abs() > .05).float().mean(dim=0))).cpu().tolist()
            weight_change = float(self.brain.log_gain.abs().mean().cpu())
        worlds = []
        for index, world in enumerate(self.factory.worlds):
            flies = []
            for fly in world.flies:
                item = {key: fly[key] for key in ("id", "x", "y", "cargo", "action", "reward", "deliveries")}
                item["activity"] = summaries[0][index * self.fly_count + fly["id"]]
                flies.append(item)
            worlds.append({"flies": flies, "stations": world.stations(), "events": list(world.events),
                           "selectedFactory": {"index": index, "products": world.products,
                                               "deliveries": world.deliveries, "reward": world.total_reward}})
        rate = 0.0
        if self.running and len(self.rate_samples) > 1:
            first, last = self.rate_samples[0], self.rate_samples[-1]
            rate = (last[1] - first[1]) / max(last[0] - first[0], 1e-6)
        base = {
            "status": "Error" if self.error else ("Training" if self.running else "Ready"),
            "running": self.running, "learning": self.learning, "steps": self.steps,
            "episode": self.factory.products + 1, "flyCount": self.fly_count,
            "environmentCount": self.environment_count, "totalAgents": self.agent_count,
            "speed": self.target_speed, "unlimited": self.unlimited,
            "reward": self.factory.total_reward, "deliveries": self.factory.deliveries,
            "products": self.factory.products, "elapsed": int(self.elapsed()),
            "stepsPerSecond": rate, "transitionsPerSecond": rate * self.agent_count,
            "environmentStepsPerSecond": rate * self.environment_count,
            "transitions": self.steps * self.agent_count,
            "learningBatch": max(LEARNING_BATCH, self.agent_count), "updateInterval": UPDATE_INTERVAL,
            "updates": self.updates, "loss": self.loss, "history": list(self.factory.history),
            "obstacles": self.factory_class.BLOCKERS,
            "brain": {"neurons": self.brain.n, "edges": self.brain.wiring._nnz(), "contacts": 124177617,
                      "gpu": torch.cuda.get_device_name(0), "memoryGB": torch.cuda.memory_allocated() / 2**30,
                      "weightChange": weight_change, "model": "MaleCNS v1.0 full graph · rate dynamics"},
            "checkpoint": self.checkpoint, "error": self.error, "runId": self.run_id,
        }
        self.snapshot = {"base": base, "worlds": worlds, "activity": sampled, "active": summaries[1]}
        self.last_snapshot = time.perf_counter()

    def payload(self, environment=0, fly=0):
        # Immutable snapshot reference: no training lock, GPU work or device sync.
        snap = self.snapshot
        if not snap:
            return {}
        if not 0 <= environment < len(snap["worlds"]) or not 0 <= fly < snap["base"]["flyCount"]:
            raise ValueError("Selected factory or fly is outside this run.")
        agent = environment * snap["base"]["flyCount"] + fly
        return {**snap["base"], **snap["worlds"][environment], "selectedFly": fly,
                "brain": {**snap["base"]["brain"], "activeFraction": snap["active"][agent]},
                "brainView": {"nodes": self.brain_view_nodes, "edges": self.brain_view_edges,
                              "activity": snap["activity"][agent], "bounds": self.brain_view_bounds}}

    def loop(self):
        while True:
            began = time.perf_counter()
            delay = .05
            try:
                with self.lock:
                    if self.running and self.ready:
                        delay = 0 if self.unlimited else 1 / self.target_speed
                        self.tick()
            except Exception as exc:
                with self.lock:
                    self.set_running(False)
                    self.error = f"Simulation stopped safely: {type(exc).__name__}: {exc}"
                    self.publish_snapshot()
            # Yield even in unlimited mode so control requests can acquire lock.
            time.sleep(max(0, delay - (time.perf_counter() - began)))
