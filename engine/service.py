"""Local full-connectome experiment service for SwarmLab."""
from __future__ import annotations

import base64
import json
import math
import os
import random
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.feather as feather
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .brain import Brain
from .training import TrainingRuntime

PROJECT = Path(__file__).resolve().parents[1]
GRAPH = Path(os.environ.get("SWARM_GRAPH", str(PROJECT.parent / "connectome-data")))
LOCAL_STATE = Path(os.environ.get("SWARM_STATE", str(PROJECT / ".runtime" / "parallel")))
ORIGIN_FILE = Path(os.environ.get("SWARM_ORIGINS_FILE", str(LOCAL_STATE / "trusted-origins.json")))
PORT = int(os.environ.get("SWARM_PORT", "8767"))
OBSERVATIONS = 22
ACTION_COUNT = 6  # north, south, west, east, interact, wait
UPDATE_INTERVAL = 24
LEARNING_BATCH = 32

LOCAL_STATE.mkdir(parents=True, exist_ok=True)
if not ORIGIN_FILE.exists():
    ORIGIN_FILE.write_text(json.dumps(["http://localhost:5173", "http://127.0.0.1:5173"], indent=2))


class OriginGuard:
    """Allow browser access only from this local preview and the private site."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        origin = headers.get(b"origin", b"").decode("latin1")
        allowed = set(json.loads(ORIGIN_FILE.read_text(encoding="utf-8")))
        if origin and origin not in allowed:
            response = JSONResponse({"error": "This page is not allowed to control the local SwarmLab engine."}, status_code=403)
            return await response(scope, receive, send)
        if scope.get("method") == "OPTIONS":
            requested = headers.get(b"access-control-request-method", b"").decode("latin1")
            if requested not in ("GET", "POST"):
                response = Response(status_code=403)
                return await response(scope, receive, send)
            response_headers = {
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "600",
                "Vary": "Origin",
            }
            if origin:
                response_headers["Access-Control-Allow-Origin"] = origin
            if headers.get(b"access-control-request-private-network", b"").lower() == b"true":
                response_headers["Access-Control-Allow-Private-Network"] = "true"
            return await Response(status_code=204, headers=response_headers)(scope, receive, send)

        async def send_with_cors(message):
            if message["type"] == "http.response.start" and origin:
                message["headers"].extend([
                    (b"access-control-allow-origin", origin.encode("latin1")),
                    (b"vary", b"Origin"),
                ])
            await send(message)

        await self.app(scope, receive, send_with_cors)


from .factory import Factory

class SwarmEngine(TrainingRuntime):
    def __init__(self):
        super().__init__(GRAPH, LOCAL_STATE, Factory)
        self.brain_view_nodes: list[dict[str, Any]] = []
        self.brain_view_edges: list[list[int]] = []
        self.brain_view_indices: torch.Tensor | None = None
        self.brain_view_activity = np.zeros(0, dtype=np.float32)
        self.brain_view_bounds: list[list[float]] = []
        self.brain_anatomy: dict[str, Any] = {}

    def _prepare_brain_view(self):
        """Build a small functional schematic from real sampled connectome nodes and edges."""
        with np.load(GRAPH / "graph.npz") as graph:
            count = self.brain.n
            sensory = graph["sensory"]
            motor = graph["motor"]
            excluded = np.zeros(count, dtype=np.bool_)
            excluded[sensory] = True
            excluded[motor] = True
            central = np.flatnonzero(~excluded)
            rng = np.random.default_rng(260912)
            body_ids = graph["body_ids"]
            annotations = feather.read_feather(
                GRAPH / "annotations.feather", columns=["bodyId", "somaLocation"]
            ).set_index("bodyId").reindex(body_ids)
            coordinates = np.full((count, 3), np.nan, dtype=np.float32)
            for index, location in enumerate(annotations["somaLocation"].to_numpy()):
                if location is not None and len(location) == 3:
                    coordinates[index] = location
            located = np.isfinite(coordinates).all(axis=1)
            sensory = sensory[located[sensory]]
            motor = motor[located[motor]]
            central = central[located[central]]
            sensory_sample = rng.choice(sensory, min(96, len(sensory)), replace=False)
            motor_sample = rng.choice(motor, min(96, len(motor)), replace=False)
            central_sample = rng.choice(central, min(448, len(central)), replace=False)
            selected = np.sort(np.concatenate((sensory_sample, central_sample, motor_sample)))
            located_coordinates = coordinates[located]
            self.brain_view_bounds = [
                [float(located_coordinates[:, axis].min()), float(located_coordinates[:, axis].max())]
                for axis in range(3)
            ]
            map_width, map_height = 160, 112
            x_bounds, y_bounds = self.brain_view_bounds[0], self.brain_view_bounds[1]
            density, _, _ = np.histogram2d(
                located_coordinates[:, 1], located_coordinates[:, 0],
                bins=(map_height, map_width), range=(y_bounds, x_bounds),
            )
            from scipy.ndimage import gaussian_filter
            density = gaussian_filter(density, sigma=1.15, mode="constant")
            density = np.sqrt(np.log1p(density) / np.log1p(density.max()))
            self.brain_anatomy = {
                "width": map_width,
                "height": map_height,
                "locatedNeurons": int(located.sum()),
                "density": base64.b64encode((density * 255).astype(np.uint8).tobytes()).decode("ascii"),
            }

            groups = np.ones(count, dtype=np.int8)
            groups[sensory] = 0
            groups[motor] = 2
            ranks = np.full(count, -1, dtype=np.int32)
            ranks[selected] = np.arange(len(selected), dtype=np.int32)
            self.brain_view_nodes = [
                {
                    "id": int(body_ids[neuron_id]),
                    "group": int(groups[neuron_id]),
                    "position": [float(value) for value in coordinates[neuron_id]],
                }
                for neuron_id in selected
            ]

            crow, col, values = graph["crow"], graph["col"], graph["values"]
            edges = []
            for target, neuron_id in enumerate(selected):
                start, end = int(crow[neuron_id]), int(crow[neuron_id + 1])
                for edge_index in range(start, end):
                    source = int(ranks[col[edge_index]])
                    if source >= 0 and source != target:
                        edges.append([source, target, 1 if values[edge_index] >= 0 else -1])
            self.brain_view_edges = edges
            self.brain_view_indices = torch.as_tensor(selected, device=self.device, dtype=torch.long)
            self.brain_view_activity = np.zeros(len(selected), dtype=np.float32)


engine = SwarmEngine()


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=engine.load, name="swarm-loader", daemon=True).start()
    engine.thread = threading.Thread(target=engine.loop, name="swarm-loop", daemon=True)
    engine.thread.start()
    yield
    with engine.lock:
        engine.set_running(False)
        engine._save()


app = FastAPI(title="SwarmLab local engine", lifespan=lifespan)
app.add_middleware(OriginGuard)


@app.get("/api/engine/health")
def health():
    return {"ready": engine.ready, "error": engine.error, "device": str(engine.device)}


@app.get("/api/engine/state")
def state(environment: int = 0, fly: int = 0):
    if not engine.ready:
        raise HTTPException(503, engine.error or "Loading the complete MaleCNS graph onto the local GPU…")
    try:
        return engine.payload(environment, fly)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/engine/brain-anatomy")
def brain_anatomy():
    with engine.lock:
        if not engine.ready:
            raise HTTPException(503, "The MaleCNS spatial map is still loading.")
        return engine.brain_anatomy


@app.post("/api/engine/control")
async def control(request: Request):
    try:
        value = await request.json()
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON command.")
        action = value.get("action")
        with engine.lock:
            if not engine.ready:
                raise HTTPException(503, "The full connectome is still loading.")
            if action == "start":
                if not engine.running:
                    engine.error = None
                    engine.set_running(True)
            elif action == "pause":
                engine.set_running(False)
                engine._save()
            elif action == "configure":
                if "learning" in value:
                    engine.learning = bool(value["learning"])
                    engine.rollout.clear()
                if "unlimited" in value:
                    engine.unlimited = bool(value["unlimited"])
                if "speed" in value:
                    speed = int(value["speed"])
                    engine.validate_settings(engine.fly_count, engine.environment_count, speed)
                    engine.target_speed = speed
            elif action == "reset":
                engine.reset(int(value.get("flyCount", engine.fly_count)), int(value.get("speed", engine.target_speed)),
                             int(value.get("environmentCount", engine.environment_count)), bool(value.get("unlimited", engine.unlimited)))
            else:
                raise ValueError("Choose start, pause, configure, or reset.")
            engine.publish_snapshot()
            return engine.payload()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, access_log=False, log_level="warning")
