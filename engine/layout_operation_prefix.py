"""Physical-operation-focused, versioned TRAINING cache, no optimizer.

Separate actual teacher-opposed source returns from teacher-supported returns
and forward transfers. Keep original sensory histories and labels unchanged.
Frozen parent motion targets protect locomotion in the next grip experiment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_correction_prefix import load_histories, DATA_SOURCES, TRAIN_LOW, TRAIN_HIGH
from .layout_batched_prefix_cache import fill_states
from .layout_recovery_teacher import KINDS, LABEL_VERSION
from .layout_recovery_brain import load_model
from .layout_recovery_world import INTERFACE, PHYSICS
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json
from .supervised_joint import raw_readout

VERSION = 'physical-operation-grip-parent-motion-prefix-v1'
CONTEXTS = ('missed-teacher-pickup', 'forward-operation', 'teacher-opposed-return', 'teacher-supported-return')


def operation_specs(episodes):
    specs = []
    for episode, (a, m) in enumerate(episodes):
        if m['frames'] < 96 or not TRAIN_LOW <= m['seed'] < TRAIN_HIGH or m['kind'] not in KINDS:
            raise ValueError('Full96-frame training histories required')
        candidates = {(fly, c): [] for fly in range(4) for c in CONTEXTS}
        for fly in range(4):
            missed = ((a['bodies'][:-1, fly, 5] == 0) & (a['labels'][:, fly, 2] > 1)
                      & (a['appliedActions'][:, fly, 2] < .5) & (a['cooldowns'][:-1, fly] <= .05))
            candidates[fly, CONTEXTS[0]] = np.flatnonzero(missed).tolist()
        for event in m['interactionEvents']:
            tick, fly = round(event['time']/.05)-1, event['fly']
            if not 0 <= tick < m['frames'] or fly not in range(4):
                raise ValueError('Invalid original event time/actor')
            on = a['labels'][tick, fly, 2] > 1
            kind = event['event']
            if kind == 'return':
                if (a['bodies'][tick, fly, 5] != event['cargoBefore']
                        or a['bodies'][tick+1, fly, 5] != 0 or event['cargoAfter'] != 0
                        or event['nearestBox'] != event['cargoBefore']-1
                        or event['boxDistance'] >= .8 or a['appliedActions'][tick, fly, 2] < .5):
                    raise ValueError('Return must be an actual source operation')
                candidates[fly, CONTEXTS[3 if on else 2]].append(tick)
            elif kind in ('transfer', 'delivery') and on:
                candidates[fly, CONTEXTS[1]].append(tick)
        rng = np.random.default_rng(m['seed']+420024)
        for (fly, category), ticks in candidates.items():
            eligible = sorted({t for t in ticks if 80 <= t < m['frames']-16})
            if not eligible:
                continue
            for tick in sorted(rng.choice(eligible, min(2, len(eligible)), replace=False)):
                specs.append({'episode': episode, 'seed': m['seed'], 'kind': m['kind'], 'fly': fly,
                              'category': category, 'focusTick': int(tick), 'start': int(tick)-80,
                              'stop': int(tick)+16, 'lossStart': int(tick)-16})
    if any(not any(s['kind'] == k and s['category'] == c for s in specs) for k in KINDS for c in CONTEXTS):
        raise ValueError('Missing actual operation family/context')
    return specs


@torch.no_grad()
def parent_motion(model, x, states):
    weights, result = model.weights(), []
    for first in range(0, len(x), 16):
        xx = torch.as_tensor(np.concatenate(x[first:first+16], axis=1), device=model.tonic.device)
        state = torch.as_tensor(np.concatenate(states[first:first+16], axis=1), device=model.tonic.device)
        part = []
        for frame in range(96):
            _, state = model(xx[frame], 4, state, weights)
            if frame >= 64:
                part.append(raw_readout(model, state)[:, :2].cpu().numpy())
        result.extend(np.stack(part).transpose(1, 0, 2)[:, :, None, :])
    return np.stack(result).astype(np.float32)


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve all original prefixes')
    episodes, data = load_histories(args.dataset, replay=True)
    specs = operation_specs(episodes)
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    if before != args.expected_parameter_hash or model.fixed_hash != data['fixedHash']:
        raise ValueError('Wrong frozen parent')
    sources = set(DATA_SOURCES) | {'layout_operation_prefix.py', 'layout_correction_prefix.py',
              'layout_batched_prefix_cache.py', 'layout_demonstration_cache.py',
              'layout_demonstration_train.py', 'layout_demonstration_curriculum.py', 'supervised_joint.py'}
    m = {'schema': VERSION, 'parameterHash': before, 'fixedHash': model.fixed_hash,
         'candidateFileHash': file_hash(args.candidate), 'datasetManifestHash': file_hash(args.dataset/'manifest.json'),
         'datasetFiles': data['episodes'], 'behaviorParameterHash': data['parameterHash'],
         'behaviorCanonicalRun': data['canonicalRun'], 'behaviorCanonicalVersion': data['canonicalVersion'],
         'prefixCanonicalRun': args.canonical_run, 'prefixCanonicalVersion': args.version,
         'sourceHashes': {n: file_hash(Path(__file__).parent/n) for n in sorted(sources)},
         'windows': specs, 'finished': False, 'interface': INTERFACE, 'physics': PHYSICS,
         'control': 'learner-only', 'teacherActions': False, 'optimizerUsed': False,
         'labelsUnchanged': True, 'labelVersion': LABEL_VERSION, 'isServiceEvidence': False,
         'motionTargets': 'unchanged parent predictions on original physical observations and full prefix',
         'returnLabelsAreTeacherPreferencesNotProofOfOptimality': True,
         'burn': 64, 'lossFrames': 32, 'fullHistoryPrefix': True}
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json', m)
    try:
        all_states = fill_states(model, episodes, specs, 32)
        x = np.stack([episodes[s['episode']][0]['observations'][s['start']:s['stop'], s['fly']:s['fly']+1] for s in specs])
        y = np.stack([episodes[s['episode']][0]['labels'][s['start']:s['stop'], s['fly']:s['fly']+1] for s in specs])
        states = np.stack([all_states[i, :, s['fly']:s['fly']+1] for i, s in enumerate(specs)])
        motion = parent_motion(model, x, states)
        if (not np.isfinite(motion).all() or motion.shape != (len(specs), 32, 1, 2)
                or model.checkpoint_hash() != before or model.fingerprint() != model.fixed_hash
                or file_hash(args.candidate) != m['candidateFileHash']
                or file_hash(args.dataset/'manifest.json') != m['datasetManifestHash']
                or any(file_hash(args.dataset/e['file']) != e['sha256'] for e in m['datasetFiles'])
                or any(file_hash(Path(__file__).parent/n) != h for n, h in m['sourceHashes'].items())):
            raise ValueError('Changed or invalid physical prefix/motion provenance')
        with (args.out/'cache.npz').open('xb') as f:
            np.savez_compressed(f, observations=x, labels=y, states=states, parent_motion=motion)
        m.update(finished=True, parametersUnchanged=True, cacheFileHash=file_hash(args.out/'cache.npz'),
                 audit=model.audit(model.log_gains.detach().cpu().numpy()))
        atomic_json(args.out/'manifest.json', m)
        print('OPERATION_PREFIX_FINISHED '+json.dumps({'windows': len(specs), 'parameterHash': before,
              'cacheFileHash': m['cacheFileHash']}), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'dataset', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-parameter-hash', required=True)
    p.add_argument('--canonical-run', required=True)
    p.add_argument('--version', type=int, required=True)
    main(p.parse_args())
