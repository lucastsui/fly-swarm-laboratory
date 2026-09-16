"""Bounded scheduling benchmark, not training or a service-quality evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluate_supervised import FrozenPolicy
from .fast_frozen_policy import FastFrozenPolicy, SplitFrozenPolicy
from .layout_recovery_brain import load_model
from .layout_recovery_world import RecoveryWorld


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve previous measurements')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    report = {'checkpointHash': before, 'device': torch.cuda.get_device_name(),
              'training': False, 'serviceEvidence': False, 'neuralSubsteps': 4,
              'physicsStepSeconds': .05, 'results': {},
              'sourceSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    baseline = FrozenPolicy(model, args.flies)
    policies = {'original': baseline, 'cached-csr': FastFrozenPolicy(model, args.flies)}
    try:
        policies['cuda-graph'] = FastFrozenPolicy(model, args.flies, cuda_graph=True)
    except RuntimeError as error:
        report['cudaGraphUnavailable'] = str(error)
    if args.index32:
        policies['cuda-graph-int32'] = FastFrozenPolicy(model, args.flies, cuda_graph=True, index32=True)
    if args.split:
        policies['cuda-graph-per-fly'] = SplitFrozenPolicy(model, args.flies)
    # Same evolving sensory stream. Compare ALL neuron states and motor outputs.
    world = RecoveryWorld(5500000, flies=args.flies, continuous=True, kind='wide')
    error = {name: 0. for name in policies if name != 'original'}
    for frame in range(args.parity_frames):
        obs = world.sensory()
        actions = baseline.act(obs)
        for name, policy in policies.items():
            if name == 'original':
                continue
            actual = policy.act(obs)
            difference = float((policy.state - baseline.state).abs().max())
            error[name] = max(error[name], difference)
            torch.testing.assert_close(policy.state, baseline.state, atol=1e-6, rtol=1e-4)
            np.testing.assert_allclose([[a['speed'], a['turn']] for a in actual],
                                       [[a['speed'], a['turn']] for a in actions], atol=1e-5, rtol=1e-4)
            assert [a['interact'] for a in actual] == [a['interact'] for a in actions]
        world.advance(actions)
    report['parity'] = {'frames': args.parity_frames, 'allNeurons': model.n,
                        'maxStateAbsoluteError': error, 'interactionDecisionsEqual': True}
    print('PARITY ' + json.dumps(report['parity']), flush=True)
    # Rotate order to limit thermal/order bias. Include sensory, collisions and
    # CPU action readback. Each backend advances its own identical-seed world.
    samples = {name: [] for name in policies}
    names = list(policies)
    for repeat in range(3):
        for name in names[repeat:] + names[:repeat]:
            policy = policies[name]
            policy.reset()
            world = RecoveryWorld(5500000, flies=args.flies, continuous=True, kind='wide')
            for _ in range(10):
                world.advance(policy.act(world.sensory()))
            torch.cuda.synchronize()
            began = time.perf_counter()
            for _ in range(args.frames):
                world.advance(policy.act(world.sensory()))
            elapsed = time.perf_counter() - began
            samples[name].append(args.frames * .05 / elapsed)
            print('BENCH ' + json.dumps({'backend': name, 'repeat': repeat,
                                         'simulationSpeed': samples[name][-1]}), flush=True)
    for name, speeds in samples.items():
        report['results'][name] = {'simulationSpeeds': speeds,
                                  'medianSimulationSpeed': float(np.median(speeds))}
    assert model.checkpoint_hash() == before
    report['weightsUnchanged'] = True
    report['componentMilliseconds'] = {}
    for name, policy in policies.items():
        policy.reset()
        world = RecoveryWorld(5500000, flies=args.flies, continuous=True, kind='wide')
        timings = np.zeros(3)
        for _ in range(30):
            t0 = time.perf_counter()
            obs = world.sensory()
            t1 = time.perf_counter()
            actions = policy.act(obs)
            t2 = time.perf_counter()
            world.advance(actions)
            t3 = time.perf_counter()
            timings += [t1-t0, t2-t1, t3-t2]
        report['componentMilliseconds'][name] = dict(zip(('sensory', 'policy', 'physics'), (timings*1000/30).tolist()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print('BENCHMARK_DONE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--parity-frames', type=int, default=200)
    parser.add_argument('--flies', type=int, default=4)
    parser.add_argument('--index32', action='store_true')
    parser.add_argument('--split', action='store_true')
    main(parser.parse_args())
