"""Small, authenticated tensor messages; never send the fixed graph or neural states."""
import hashlib
import io
from http.client import HTTPException
import json
import time
import urllib.error
import urllib.request

import torch
from torch import nn


def pack(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def unpack(value):
    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)


def signature(model):
    layout = [(name, list(param.shape)) for name, param in model.named_parameters()]
    return hashlib.sha256(json.dumps(layout).encode()).hexdigest()


def optimizer_for(model):
    return torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": [model.log_gain, model.bias], "lr": 1e-5},
        {"params": list(model.readout.parameters()) + list(model.actor.parameters()) + list(model.critic.parameters()), "lr": 1e-4},
    ], weight_decay=1e-5)


class ParameterModel(nn.Module):
    """Only learned parameters on the coordinator CPU; GPU workers own the full graph."""
    def __init__(self, weights):
        super().__init__()
        self.log_gain = nn.Parameter(weights["log_gain"].clone())
        self.bias = nn.Parameter(weights["bias"].clone())
        self.encoder = nn.Linear(weights["encoder"]["weight"].shape[1], weights["encoder"]["weight"].shape[0])
        motor = weights["readout"]["0.weight"].numel()
        self.readout = nn.Sequential(nn.LayerNorm(motor), nn.Linear(motor, 64), nn.Tanh())
        self.actor, self.critic = nn.Linear(64, 6), nn.Linear(64, 1)
        for name in ("encoder", "readout", "actor", "critic"):
            getattr(self, name).load_state_dict(weights[name])

    def weights(self):
        result = {name: getattr(self, name).state_dict() for name in ("encoder", "readout", "actor", "critic")}
        return {**result, "log_gain": self.log_gain.detach(), "bias": self.bias.detach()}


class ClusterClient:
    def __init__(self, url, token):
        self.url, self.token = url.rstrip("/"), token
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def raw(self, path, body=None, content_type="application/octet-stream", retries=3):
        # The caller supplies a stable update ID, so retrying cannot apply a gradient twice.
        for attempt in range(retries):
            try:
                request = urllib.request.Request(self.url + path, data=body, headers={
                    "Authorization": "Bearer " + self.token, "Content-Type": content_type,
                })
                with self.opener.open(request, timeout=15) as response:
                    return response.read()
            except urllib.error.HTTPError:
                raise
            except (OSError, TimeoutError, HTTPException):
                if attempt == retries - 1:
                    raise
                time.sleep(.25 * (attempt + 1))

    def tensor(self, path, value=None):
        return unpack(self.raw(path, None if value is None else pack(value)))

    def json(self, path, value=None):
        return json.loads(self.raw(path, None if value is None else json.dumps(value, allow_nan=False).encode(), "application/json"))
