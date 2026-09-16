"""Static local motor discrimination diagnostic, NOT physical service evidence."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from .layout_recovery_brain import load_model
from .layout_recovery_world import RecoveryWorld
from .layout_recovery_teacher import LocalTeacher, KINDS
from .layout_recovery_protocol import atomic_json
from .supervised_joint import raw_readout


def examples(seed=9190000, scenes=32):
    rng = np.random.default_rng(seed)
    obs, labels = [], []
    for i in range(scenes):
        world = RecoveryWorld(int(rng.integers(9190000, 9199000)), flies=1, kind=KINDS[i % 4])
        agent = world.agents[0]
        station = world.stations[i % 4]
        angle = float(rng.uniform(-math.pi, math.pi))
        distance = float(rng.uniform(.3, .7) if i % 3 else rng.uniform(2., 4.))
        agent.x = float(np.clip(station['x']+distance*math.cos(angle), .23, 19.77))
        agent.y = float(np.clip(station['y']+distance*math.sin(angle), .23, 13.77))
        agent.heading = float(rng.uniform(-math.pi, math.pi))
        for station in world.stations[1:3]:
            station['stock'] = int(rng.integers(3))
        for cargo in range(4):
            agent.cargo = cargo
            obs.append(world.sensory()[0])
            labels.append(LocalTeacher().label(agent))
    return np.asarray(obs), np.asarray(labels)


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostic')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate)
    model.eval().requires_grad_(False)
    obs, target = examples()
    prediction = []
    with torch.no_grad():
        weights = model.weights()
        for offset in range(0, len(obs), 16):
            _, state = model(torch.as_tensor(obs[offset:offset+16], device='cuda'), 128, weights=weights)
            prediction.extend(raw_readout(model, state).cpu().tolist())
    prediction = np.asarray(prediction)
    positive = target[:, 2] > 1.
    mask = np.abs(target[:, 1]) > .2
    result = {'checkpointHash': model.checkpoint_hash(), 'staticDiagnosticNotService': True,
              'cases': len(obs), 'targetGripPositive': int(positive.sum()),
              'gripRecall': float((prediction[positive, 2] > 1).mean()),
              'gripFalsePositiveRate': float((prediction[~positive, 2] > 1).mean()),
              'gripRange': [float(prediction[:, 2].min()), float(prediction[:, 2].max())],
              'turnDirectionAccuracy': float((np.sign(prediction[mask, 1]) == np.sign(target[mask, 1])).mean()),
              'mseByHead': ((prediction-target)**2).mean(0).tolist(),
              'prediction': prediction.tolist(), 'target': target.tolist()}
    atomic_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('prediction', 'target')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
