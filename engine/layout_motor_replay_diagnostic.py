"""Frozen motor replay on original training histories, not closed-loop service.

Compare the candidate's actual fixed decoder with recorded behavior actions.
Full-history candidate prefix states prevent artificial recurrent resets. Neither
teacher labels nor sampling metadata enter the model. Overlapping windows are
reported explicitly and unique physical frames are summarized separately.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_operation_sampling import OperationCache
from .layout_correction_prefix import load_histories
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def summarize(rows):
    if not rows:
        return {'frames': 0}
    a = np.asarray([r['candidate'] for r in rows], dtype=np.float64)
    b = np.asarray([r['behavior'] for r in rows], dtype=np.float64)
    motion = np.abs(a[:, :2]-b[:, :2])
    flip = a[:, 2] != b[:, 2]
    ready = np.asarray([r['cooldown'] <= .05 for r in rows])
    loaded = np.asarray([r['cargo'] > 0 for r in rows])
    return {'frames': len(rows), 'movementMaxAbs': motion.max(0).tolist(),
            'movementRms': np.sqrt(np.mean(motion**2, axis=0)).tolist(),
            'movementP99Abs': np.quantile(motion, .99, axis=0).tolist(),
            'gripFlips': int(flip.sum()), 'readyGripFlips': int((flip & ready).sum()),
            'readyLoadedGripFlips': int((flip & ready & loaded).sum()),
            'readyEmptyGripFlips': int((flip & ready & ~loaded).sum())}


@torch.no_grad()
def replay(model, bank, episodes):
    weights = model.weights()
    unique, windows = {}, []
    duplicate_max = 0.
    for first in range(0, len(bank.manifest['windows']), 16):
        ids = list(range(first, min(first+16, len(bank.manifest['windows']))))
        x, _, state, _ = bank.batch(ids, model.tonic.device)
        outputs = []
        for frame in range(96):
            actions, state = model(x[frame], 4, state, weights)
            outputs.append(actions.cpu().numpy())
        outputs = np.stack(outputs)
        if not np.isfinite(outputs).all():
            raise ValueError('Nonfinite fixed-decoder output')
        for actor, index in enumerate(ids):
            spec = bank.manifest['windows'][index]
            original, meta = episodes[spec['episode']]
            rows = []
            # Last32frames use the original cache's unmodified score interval.
            for offset in range(64, 96):
                tick, fly = spec['start']+offset, spec['fly']
                decoded = outputs[offset, actor]
                row = {'episode': spec['episode'], 'seed': meta['seed'], 'kind': meta['kind'],
                       'tick': tick, 'fly': fly, 'cargo': int(original['bodies'][tick, fly, 5]),
                       'cooldown': float(original['cooldowns'][tick, fly]),
                       'candidate': [float(decoded[0]), float(decoded[1]), int(decoded[2] > .025)],
                       'candidateGripRate': float(decoded[2]),
                       'behavior': original['appliedActions'][tick, fly].tolist()}
                key = (spec['episode'], tick, fly)
                if key in unique:
                    old = unique[key]
                    duplicate_max = max(duplicate_max, abs(old['candidateGripRate']-row['candidateGripRate']),
                                        max(abs(a-b) for a,b in zip(old['candidate'], row['candidate'])))
                else:
                    unique[key] = row
                rows.append(row)
            windows.append({'windowIndex': index, **spec, **summarize(rows)})
        print(json.dumps({'frozenMotorReplayWindows': first+len(ids)}), flush=True)
    rows = list(unique.values())
    return {'summary': summarize(rows), 'families': {k: summarize([r for r in rows if r['kind']==k])
            for k in sorted({r['kind'] for r in rows})}, 'windows': windows,
            'uniquePhysicalFrames': rows, 'duplicateReplayMaxDifference': duplicate_max,
            'overlappingWindowsCountedOnceInSummary': True}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve original diagnostics')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    if before != args.expected_parameter_hash:
        raise ValueError('Wrong candidate')
    bank = OperationCache(args.cache, args.dataset, before, model.fixed_hash)
    episodes, data = load_histories(args.dataset, replay=False)
    paths = {'candidate': args.candidate, 'cacheManifest': args.cache/'manifest.json',
             'cache': args.cache/'cache.npz', 'datasetManifest': args.dataset/'manifest.json'}
    paths.update({'source:'+p.name: p for p in [Path(__file__),
                  Path(__file__).with_name('layout_operation_sampling.py'),
                  Path(__file__).with_name('layout_recovery_brain.py'),
                  Path(__file__).with_name('supervised_steering.py')]})
    hashes = {k: file_hash(p) for k,p in paths.items()}
    result = replay(model, bank, episodes)
    if (model.checkpoint_hash() != before or model.fingerprint() != model.fixed_hash
            or any(file_hash(p) != hashes[k] for k,p in paths.items())):
        raise ValueError('Diagnostic changed original parameters or inputs')
    result.update(schema='frozen-fixed-decoder-training-history-replay-v1',
                  parameterHash=before, behaviorParameterHash=data['parameterHash'],
                  fixedHash=model.fixed_hash, hashes=hashes, device=torch.cuda.get_device_name(),
                  parametersUnchanged=True, fullHistoryPrefix=True, teacherActions=False,
                  optimizerUsed=False, isServiceEvidence=False,
                  limitation='Counterfactual actions on recorded behavior histories; not candidate closed-loop trajectories.')
    atomic_json(args.out, result)
    print('MOTOR_REPLAY_FINISHED '+json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'dataset', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-parameter-hash', required=True)
    main(p.parse_args())
