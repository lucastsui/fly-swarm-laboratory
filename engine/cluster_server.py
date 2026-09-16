"""Bounded-staleness asynchronous actor-critic parameter server and cluster controls."""
import argparse
import hashlib
import hmac
import json
import shutil
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from .cluster_protocol import ParameterModel, optimizer_for, pack, unpack, signature


class Coordinator:
    def __init__(self, bootstrap, metadata, storage, configs):
        self.lock = threading.RLock()
        self.storage = storage
        storage.mkdir(parents=True, exist_ok=True)
        self.metadata, self.configs = metadata, configs
        self.model = ParameterModel(bootstrap["weights"])
        self.optimizer = optimizer_for(self.model)
        if bootstrap.get("optimizer"):
            self.optimizer.load_state_dict(bootstrap["optimizer"])
        self.layout = signature(self.model)
        self.params = list(self.model.parameters())
        self.version = bootstrap.get("updates", 0)
        self.generation = uuid.uuid4().hex[:12]
        self.running, self.learning, self.unlimited, self.speed = False, True, True, 50
        self.workers = {}
        self.watch = {name: {"environment": 0, "fly": 0} for name in configs}
        self.seen = deque(maxlen=1024)
        self.history = deque(maxlen=1400)
        self.reward_window = deque(maxlen=31)
        self.seconds, self.started = 0.0, None
        self.checkpoint = None
        self.last_history = self.last_save = 0.0
        self.accepted = {name: 0 for name in configs}
        self.rejected = {name: 0 for name in configs}
        self.last_lag = {name: 0 for name in configs}
        self.samples = {name: 0 for name in configs}
        saved_path = storage / "cluster-state.pt"
        if saved_path.exists():
            saved = torch.load(saved_path, weights_only=True)
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.version, self.generation = saved["version"], saved["generation"]
            self.configs = {name: saved["configs"].get(name, config) for name, config in configs.items()}
            self.history = deque(saved["history"], maxlen=1400)
            self.seconds, self.seen = saved["seconds"], deque(saved["seen"], maxlen=1024)
            self.accepted = {name: saved["accepted"].get(name, 0) for name in configs}
            self.rejected = {name: saved["rejected"].get(name, 0) for name in configs}
            self.samples = {name: saved["samples"].get(name, 0) for name in configs}
        self._refresh_parameters()

    def elapsed(self):
        return self.seconds + (time.monotonic() - self.started if self.started is not None else 0)

    def _refresh_parameters(self):
        self.flat = torch.nn.utils.parameters_to_vector(self.params).detach().clone()
        self.weight_hash = hashlib.sha256(self.flat.numpy().tobytes()).hexdigest()

    def controls(self, worker):
        return {"running": self.running, "learning": self.learning, "unlimited": self.unlimited,
                "speed": self.speed, "generation": self.generation, "config": self.configs[worker],
                "watch": self.watch[worker], "version": self.version}

    def parameters(self, worker, accepted=None):
        return {**self.controls(worker), "parameters": self.flat.clone(), "signature": self.layout,
                "weightHash": self.weight_hash, "accepted": accepted,
                "acceptedUpdates": self.accepted[worker], "rejectedUpdates": self.rejected[worker]}

    def update(self, value):
        worker = value["worker"]
        with self.lock:
            if worker not in self.configs or value.get("signature") != self.layout:
                raise ValueError("Unknown worker or incompatible model layout")
            request_id = value["id"]
            if request_id in self.seen:
                return self.parameters(worker, accepted=False)
            lag = self.version - int(value["version"])
            self.last_lag[worker] = lag
            if (value["generation"] != self.generation or not self.running or not self.learning or not 0 <= lag <= 16):
                self.rejected[worker] += 1
                self.seen.append(request_id)
                return self.parameters(worker, accepted=False)
            gradient = value["gradient"]
            if gradient.dtype != torch.float32 or gradient.shape != self.flat.shape or not bool(torch.isfinite(gradient).all()):
                raise ValueError("Invalid or non-finite gradient")
            self.optimizer.zero_grad(set_to_none=True)
            offset = 0
            for param in self.params:
                param.grad = gradient[offset:offset + param.numel()].view_as(param)
                offset += param.numel()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.version += 1
            self.accepted[worker] += 1
            self.samples[worker] += int(value["samples"])
            self.seen.append(request_id)
            self._refresh_parameters()
            return self.parameters(worker, accepted=True)

    def heartbeat(self, value):
        worker = value["worker"]
        if worker not in self.configs:
            raise ValueError("Unknown worker")
        with self.lock:
            if value.get("generation") == self.generation:
                self.workers[worker] = {**value, "seenAt": time.monotonic()}
            return self.controls(worker)

    def save(self):
        with self.lock:
            value = {"schema": 1, "model": self.model.state_dict(), "weights": self.model.weights(),
                     "optimizer": self.optimizer.state_dict(), "version": self.version,
                     "generation": self.generation, "configs": self.configs, "history": list(self.history),
                     "seconds": self.elapsed(), "seen": list(self.seen), "accepted": self.accepted,
                     "rejected": self.rejected, "samples": self.samples}
            temp = self.storage / "cluster-state.pt.tmp"
            torch.save(value, temp)
            temp.replace(self.storage / "cluster-state.pt")
            self.checkpoint, self.last_save = time.strftime("%H:%M:%S"), time.monotonic()

    def control(self, command):
        with self.lock:
            action = command.get("action")
            if action == "start":
                if not self.running:
                    self.started = time.monotonic()
                    self.running = True
            elif action == "pause":
                if self.running:
                    self.seconds = self.elapsed()
                    self.started = None
                    self.running = False
                self.save()
            elif action == "configure":
                speed = int(command.get("speed", self.speed))
                if not 1 <= speed <= 200:
                    raise ValueError("Speed must be 1–200 ticks/s")
                self.speed = speed
                self.learning = bool(command.get("learning", self.learning))
                self.unlimited = bool(command.get("unlimited", self.unlimited))
            elif action == "reset":
                worker = command.get("worker", next(iter(self.configs)))
                count, flies = int(command.get("environmentCount", 8)), int(command.get("flyCount", 4))
                if worker not in self.configs or not 1 <= count <= 32 or not 1 <= flies <= 32 or count * flies > 128:
                    raise ValueError("At most 128 flies per worker, with 1–32 factories and flies per factory")
                self.control({"action": "pause"})
                archive = self.storage / "archives" / self.generation
                archive.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.storage / "cluster-state.pt", archive / "cluster-state.pt")
                self.configs[worker] = {**self.configs[worker], "environmentCount": count, "flyCount": flies}
                self.generation = uuid.uuid4().hex[:12]
                self.watch = {name: {"environment": 0, "fly": 0} for name in self.configs}
                self.workers.clear()
                self.history.clear()
                self.reward_window.clear()
                self.seconds = 0.0
                self.save()
            else:
                raise ValueError("Choose start, pause, configure, or reset")
            return {"ok": True, "running": self.running, "version": self.version}

    def totals(self):
        now = time.monotonic()
        records = []
        for worker, config in self.configs.items():
            item = self.workers.get(worker, {})
            age = now - item.get("seenAt", now - 1e6)
            active = bool(item.get("ready")) and age < 5 and not item.get("error")
            rate = item.get("actionsPerSecond", 0) if active and self.running else 0
            records.append({"id": worker, "label": config["label"], "host": config["host"],
                            "environmentCount": config["environmentCount"], "flyCount": config["flyCount"],
                            "agents": config["environmentCount"] * config["flyCount"],
                            "status": "Training" if active and self.running else "Ready" if active else "Offline" if age >= 5 else "Loading",
                            "actionsPerSecond": rate, "acceptedUpdates": self.accepted[worker],
                            "sampledTransitions": self.samples[worker], "rejectedUpdates": self.rejected[worker],
                            "modelVersion": item.get("version", 0), "lastGradientLag": self.last_lag[worker],
                            "gpu": item.get("gpu", "Loading"), "memoryGB": item.get("memoryGB", 0),
                            "error": item.get("error"), "communicationMs": item.get("communicationMs", 0)})
        sums = {key: sum(w.get(key, 0) for w in self.workers.values()) for key in ("reward", "products", "deliveries", "transitions")}
        sums["rate"] = sum(r["actionsPerSecond"] for r in records)
        return records, sums

    def sample_history(self):
        with self.lock:
            records, totals = self.totals()
            if self.running:
                self.reward_window.append((totals["transitions"], totals["reward"]))
                first = self.reward_window[0]
                delta = totals["transitions"] - first[0]
                self.history.append({"seconds": self.elapsed(), "step": totals["transitions"],
                                     "transitions": totals["transitions"], "reward": totals["reward"],
                                     "rate": 100 * (totals["reward"] - first[1]) / max(1, delta),
                                     "products": totals["products"], "deliveries": totals["deliveries"]})

    def payload(self, worker, environment, fly):
        with self.lock:
            if worker not in self.configs:
                raise ValueError("Unknown worker")
            config = self.configs[worker]
            if not 0 <= environment < config["environmentCount"] or not 0 <= fly < config["flyCount"]:
                raise ValueError("Selected factory or fly is outside this worker")
            self.watch[worker] = {"environment": environment, "fly": fly}
            records, totals = self.totals()
            item = self.workers.get(worker, {})
            snapshot = item.get("snapshot")
            if not snapshot:
                return None
            world = snapshot["worlds"][environment]
            measured = snapshot.get("environment") == environment and snapshot.get("fly") == fly
            view = {key: self.metadata[key] for key in ("nodes", "edges", "bounds")}
            view["activity"] = snapshot.get("activity", []) if measured else []
            this_worker = next(r for r in records if r["id"] == worker)
            return {**world, "status": "Training" if self.running else "Ready", "running": self.running,
                    "learning": self.learning, "speed": self.speed, "unlimited": self.unlimited,
                    "steps": item.get("steps", 0), "episode": totals["products"] + 1,
                    "environmentCount": config["environmentCount"], "flyCount": config["flyCount"],
                    "totalAgents": config["environmentCount"] * config["flyCount"],
                    "learningBatch": max(32, config["environmentCount"] * config["flyCount"]), "updateInterval": 24,
                    "environmentStepsPerSecond": this_worker["actionsPerSecond"] / config["flyCount"],
                    "transitions": totals["transitions"], "transitionsPerSecond": totals["rate"],
                    "stepsPerSecond": this_worker["actionsPerSecond"] / this_worker["agents"],
                    "products": totals["products"], "deliveries": totals["deliveries"], "reward": totals["reward"],
                    "elapsed": int(self.elapsed()), "updates": self.version, "loss": item.get("loss", 0),
                    "history": list(self.history), "selectedFly": fly, "viewPending": not measured,
                    "obstacles": self.metadata["obstacles"], "brainView": view,
                    "brain": {"neurons": 166700, "edges": 25582938, "contacts": 124177617,
                              "gpu": this_worker["gpu"], "memoryGB": this_worker["memoryGB"],
                              "activeFraction": snapshot.get("activeFraction", 0) if measured else 0,
                              "weightChange": 0, "model": "MaleCNS full graph · distributed rate model"},
                    "runId": self.generation + "-" + worker, "checkpoint": self.checkpoint,
                    "error": item.get("error"),
                    "cluster": {"selectedWorker": worker, "workers": records, "modelVersion": self.version,
                                "totalAgents": sum(r["agents"] for r in records), "totalFactories": sum(r["environmentCount"] for r in records),
                                "parameterHash": self.weight_hash, "maxGradientLag": 16,
                                "method": "Asynchronous actor–critic · one shared optimizer", "rewardWindowSeconds": 30}}


def serve(coordinator, token, bind, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, code, value, binary=False):
            body = value if binary else json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream" if binary else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def handle_request(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"detail": "Authentication required"})
                return
            try:
                uri = urlparse(self.path)
                query = parse_qs(uri.query)
                worker = query.get("worker", [next(iter(coordinator.configs))])[0]
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 16 * 1024 * 1024:
                    raise ValueError("Request too large")
                body = self.rfile.read(size)
                if uri.path == "/worker/parameters":
                    with coordinator.lock:
                        result = coordinator.parameters(worker)
                    self.reply(200, pack(result), True)
                elif uri.path == "/worker/update" and self.command == "POST":
                    self.reply(200, pack(coordinator.update(unpack(body))), True)
                elif uri.path == "/worker/heartbeat" and self.command == "POST":
                    self.reply(200, coordinator.heartbeat(json.loads(body)))
                elif uri.path == "/api/engine/control" and self.command == "POST":
                    self.reply(200, coordinator.control(json.loads(body)))
                elif uri.path == "/api/engine/state":
                    result = coordinator.payload(worker, int(query.get("environment", [0])[0]), int(query.get("fly", [0])[0]))
                    self.reply(200 if result else 503, result or {"detail": "Selected GPU worker is loading…"})
                elif uri.path == "/api/engine/health":
                    self.reply(200, {"ready": True, "workers": coordinator.totals()[0]})
                elif uri.path == "/cluster/checkpoint":
                    coordinator.save()
                    self.reply(200, (coordinator.storage / "cluster-state.pt").read_bytes(), True)
                else:
                    self.reply(404, {"detail": "Unknown route"})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"detail": str(exc)})
            except Exception as exc:
                print(f"Request failed: {type(exc).__name__}: {exc}", flush=True)
                self.reply(500, {"detail": "Cluster request failed; see coordinator log"})

        do_GET = do_POST = handle_request
    server = ThreadingHTTPServer((bind, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    from .deployment import COORDINATOR_HOST
    parser.add_argument("--peer-bind", default=COORDINATOR_HOST)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    torch.set_num_threads(2)
    coordinator = Coordinator(torch.load(args.bootstrap, weights_only=True), json.loads(args.metadata.read_text()), args.storage, json.loads(args.config.read_text()))
    token = args.token_file.read_text().strip()
    serve(coordinator, token, "127.0.0.1", args.port)
    serve(coordinator, token, args.peer_bind, args.port)
    print("Cluster coordinator ready; workers initially paused.", flush=True)
    while True:
        time.sleep(1)
        coordinator.sample_history()
        if time.monotonic() - coordinator.last_save > 30:
            coordinator.save()


if __name__ == "__main__":
    main()
