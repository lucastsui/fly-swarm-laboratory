"""Bounded repeatability check of unchanged GPU inference on training inputs.

No optimizer, body commands, teacher input, parameter updates or service claim.
Strict deterministic mode is tested explicitly, never silently substituted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from .layout_correction_prefix import load_histories, DATA_SOURCES
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def compare_outputs(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite outputs required')
    diff = np.abs(a.astype(np.float64)-b)
    locations = np.argwhere(a != b)
    return {'bitwiseEqual': bool(np.array_equal(a, b)), 'differentValues': len(locations),
            'maximumDifference': float(diff.max()),
            'firstDifferentFrameActorHead': locations[0].tolist() if len(locations) else None,
            'gripDecisionDifferences': int(np.count_nonzero((a[..., 2] > .025) != (b[..., 2] > .025)))}


@torch.no_grad()
def one_pass(model, observations):
    state, outputs, weights = None, [], model.weights()
    for x in observations:
        actions, state = model(x, 4, state, weights)
        outputs.append(actions.cpu().numpy())
    result = np.stack(outputs)
    if not torch.isfinite(state).all():
        raise ValueError('Nonfinite final recurrent state')
    return result, hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()


def main(args):
    if args.out.exists() or not 4 <= args.frames <= 200 or not 2 <= args.repeats <= 3:
        raise ValueError('New output and bounded experiment required')
    torch.set_num_threads(4)
    episodes, data = load_histories(args.dataset, replay=False)
    # Same128-column batch width as paired physical evaluation, but exclusively
    # original TRAINING sensory histories. Duplicate actors are explicit.
    x = np.concatenate([a['observations'][:args.frames] for a,_ in episodes], axis=1)
    x = np.tile(x, (1, (128+x.shape[1]-1)//x.shape[1], 1))[:, :128]
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    before, fixed = model.checkpoint_hash(), model.fixed_hash
    if before != args.expected_parameter_hash:
        raise ValueError('Wrong frozen candidate')
    hashes = {'candidate': file_hash(args.candidate), 'datasetManifest': file_hash(args.dataset/'manifest.json')}
    sources = {n: file_hash(Path(__file__).with_name(n)) for n in (*DATA_SOURCES, Path(__file__).name)}
    observations = torch.as_tensor(x, device=model.tonic.device)
    initial_mode = torch.are_deterministic_algorithms_enabled()
    results = []
    try:
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic)
            row = {'strictDeterministicAlgorithms': deterministic, 'runs': []}
            first = None
            try:
                for i in range(args.repeats):
                    output, state_hash = one_pass(model, observations)
                    record = {'run': i+1, 'outputHash': hashlib.sha256(output.tobytes()).hexdigest(),
                              'finalStateHash': state_hash}
                    if first is not None:
                        record['versusFirst'] = compare_outputs(first, output)
                    else:
                        first = output
                    row['runs'].append(record)
                row['finished'] = True
            except RuntimeError as error:
                row.update(finished=False, error=str(error), errorType=type(error).__name__)
            results.append(row)
            print(json.dumps(row), flush=True)
    finally:
        torch.use_deterministic_algorithms(initial_mode)
    if (model.checkpoint_hash() != before or model.fingerprint() != fixed
            or file_hash(args.candidate) != hashes['candidate']
            or file_hash(args.dataset/'manifest.json') != hashes['datasetManifest']
            or any(file_hash(Path(__file__).with_name(n)) != h for n,h in sources.items())):
        raise ValueError('Original files or weights changed')
    result = {'schema': 'frozen-forward-repeatability-v1', 'results': results, 'parameterHash': before,
              'fixedHash': fixed, 'hashes': hashes, 'sourceHashes': sources,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
              'workspaceConfig': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
              'frames': args.frames, 'actors': 128, 'neuralSubsteps': 4,
              'originalTrainingSeeds': [m['seed'] for _,m in episodes], 'trainingActorsTiledTo128': True,
              'teacherActions': False, 'parametersUnchanged': True, 'optimizerUsed': False,
              'isServiceEvidence': False}
    atomic_json(args.out, result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('root', 'candidate', 'dataset', 'out'):
        p.add_argument('--'+n, required=True, type=Path)
    p.add_argument('--expected-parameter-hash', required=True)
    p.add_argument('--frames', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    main(p.parse_args())
