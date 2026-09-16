"""Training-only matched cargo contexts and frozen-parent locomotion targets.

These are synthetic observation sequences, NOT physical rollouts. No extra
network, sensory gate, runtime controller or decoder is created. Targets for
speed/turn are cached from the one canonical brain BEFORE its first update.
"""
import math
import numpy as np
import torch
from .layout_recovery_world import RecoveryWorld
from .layout_recovery_teacher import LocalTeacher, KINDS
from .supervised_joint import raw_readout

VERSION = 'matched-cargo-grip-parent-motion-v1'


def matched_examples(seed=9320001, scenes=64, burn=64, frames=32):
    if scenes < 4 or burn < 1 or frames < 1:
        raise ValueError('Positive bounds and at least four scenes required')
    rng = np.random.default_rng(seed)
    observations, targets, metadata = [], [], []
    for scene in range(scenes):
        layout_seed = int(rng.integers(9320000, 9329000))
        kind = KINDS[int(rng.integers(4))]
        world = RecoveryWorld(layout_seed, flies=1, kind=kind)
        agent = world.agents[0]
        box = scene % 4
        # Isolate local handling without selecting a target in the sensory map.
        # All four boxes retain their ordinary simultaneous visual channels.
        for i, station in enumerate(world.stations[:3]):
            station['stock'] = 1 if i == box else 0
        station = world.stations[box]
        angle = float(rng.uniform(-math.pi, math.pi))
        near = scene % 4 != (scene // 4) % 4  # 75%, uncoupled from box role
        distance = float(rng.uniform(.3, .7) if near else rng.uniform(1.1, 4.))
        agent.x = float(np.clip(station['x']+distance*math.cos(angle), .23, 19.77))
        agent.y = float(np.clip(station['y']+distance*math.sin(angle), .23, 13.77))
        agent.heading = float(rng.uniform(-math.pi, math.pi))
        agent.speed = float(rng.uniform(.2, 1.))
        agent.turn = float(rng.uniform(-.8, .8))
        current, labels = [], []
        agent.cargo = 0
        empty = world.sensory()[0].copy()
        for cargo in range(4):
            agent.cargo = cargo
            current.append(world.sensory()[0])
            labels.append(LocalTeacher().label(agent))
        current = np.asarray(current, np.float32)
        sequence = np.broadcast_to(current, (burn+frames, 4, current.shape[1])).copy()
        transition = (scene // 4) % 2 == 0
        if transition:
            # Identical visible box, then changed cargo feedback. This targets
            # releasing the grip after pickup, without an action override.
            sequence[:burn] = empty
        observations.append(sequence)
        targets.append(np.broadcast_to(np.asarray(labels), (frames, 4, 3)).copy())
        metadata.append({'seed': layout_seed, 'kind': kind, 'box': box, 'near': near,
                         'cargoTransition': transition, 'syntheticTrainingOnly': True})
    return np.concatenate(observations, axis=1), np.concatenate(targets, axis=1), metadata


@torch.no_grad()
def cache_parent_motion(model, observations, targets, burn, batch=16):
    """Return plain CPU arrays, with no second brain or retained autograd graph."""
    if batch % 4:
        raise ValueError('Keep four-cargo scene groups intact')
    device = model.tonic.device
    anchored = targets.copy()
    weights = model.weights()
    for offset in range(0, observations.shape[1], batch):
        state = None
        for frame, obs in enumerate(observations[:, offset:offset+batch]):
            _, state = model(torch.as_tensor(obs, device=device), 4, state, weights)
            if frame >= burn:
                anchored[frame-burn, offset:offset+batch, :2] = raw_readout(model, state)[:, :2].cpu().numpy()
    return anchored


def cargo_contrast(prediction, target):
    """Match within-scene grip differences; a constant output cannot satisfy it."""
    pred = prediction[..., 2].reshape(-1, prediction.shape[-2]//4, 4)
    label = target[..., 2].reshape_as(pred)
    delta = pred-label
    return (delta-delta.mean(-1, keepdim=True)).square().mean()


@torch.no_grad()
def calibration_progress(model, bank, actors=16):
    """Fixed TRAINING-subset fit check, never a service/generalization score."""
    before = model.checkpoint_hash()
    state = None
    weights = model.weights()
    predictions = []
    for frame, obs in enumerate(bank.observations[:, :actors]):
        _, state = model(torch.as_tensor(obs, device=model.tonic.device), 4, state, weights)
        if frame >= bank.burn:
            predictions.append(raw_readout(model, state).cpu().numpy())
    p = np.asarray(predictions)
    y = bank.targets[:, :actors]
    positive = y[..., 2] > 1.
    response = p[..., 2] > 1.
    recall = float(response[positive].mean()) if positive.any() else None
    false_positive = float(response[~positive].mean()) if (~positive).any() else None
    if model.checkpoint_hash() != before:
        raise AssertionError('Fit logging changed parameters')
    return {'checkpointHash': before, 'trainingSubsetOnly': True, 'isServiceEvidence': False,
            'actors': p.shape[1], 'frames': p.shape[0], 'gripRecall': recall,
            'gripFalsePositiveRate': false_positive,
            'gripBalancedAccuracy': .5*(recall+1-false_positive) if recall is not None and false_positive is not None else None,
            'gripRange': [float(p[..., 2].min()), float(p[..., 2].max())],
            'parentMotionMSE': ((p[..., :2]-y[..., :2])**2).mean((0, 1)).tolist()}


class CalibrationBank:
    def __init__(self, model, seed, scenes, burn, frames):
        self.burn = burn
        self.observations, teacher_targets, self.metadata = matched_examples(seed, scenes, burn, frames)
        self.targets = cache_parent_motion(model, self.observations, teacher_targets, burn)
        self.parent_hash = model.checkpoint_hash()

    def sample(self, rng, device, scenes=4):
        selected = rng.integers(len(self.metadata), size=scenes)
        indices = (selected[:, None]*4+np.arange(4)).ravel()
        x = torch.as_tensor(self.observations[:, indices], device=device)
        y = torch.as_tensor(self.targets[:, indices], device=device)
        return x, y, selected.tolist()

    def subset(self, selected):
        selected = np.asarray(selected, dtype=np.int64)
        if selected.ndim != 1 or not len(selected) or len(set(selected.tolist())) != len(selected):
            raise ValueError('Distinct scene indices required')
        if selected.min() < 0 or selected.max() >= len(self.metadata):
            raise ValueError('Scene index outside bank')
        bank = object.__new__(CalibrationBank)
        indices = (selected[:, None]*4+np.arange(4)).ravel()
        bank.observations = self.observations[:, indices].copy()
        bank.targets = self.targets[:, indices].copy()
        bank.metadata = [{**self.metadata[i], 'sourceSceneIndex': int(i)} for i in selected]
        bank.burn, bank.parent_hash = self.burn, self.parent_hash
        return bank

    def all_examples(self, device):
        return (torch.as_tensor(self.observations, device=device),
                torch.as_tensor(self.targets, device=device),
                [m.get('sourceSceneIndex', i) for i, m in enumerate(self.metadata)])

    def save(self, path):
        if path.exists():
            raise FileExistsError('Preserve calibration bank')
        np.savez_compressed(path, observations=self.observations, targets=self.targets,
                            parent_hash=np.asarray(self.parent_hash), version=np.asarray(VERSION))
