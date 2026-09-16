"""Frozen, checkpoint-versioned recurrent prefix states for TRAINING demos.

No optimizer, brain-driven physical rollout, or inference teacher. The recorded
teacher trajectory supplies inputs only; every cached hidden state is obtained
by the unchanged full connectome consuming its entire preceding sensory history.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_recovery_demonstrations import load_episode, file_hash, SCHEMA, CONTROL
from .layout_recovery_world import INTERFACE, PHYSICS
from .layout_recovery_teacher import KINDS
from .layout_recovery_brain import load_model
from .layout_recovery_protocol import atomic_json

CACHE_SCHEMA = 'checkpoint-prefix-demonstration-cache-v1'


def load_dataset(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('control') != CONTROL
            or manifest.get('finished') is not True or manifest.get('isBrainEvidence') is not False):
        raise ValueError('Dataset is not completed teacher-only TRAINING data')
    if not 1 <= len(manifest['episodes']) <= 64:
        raise ValueError('Invalid bounded demonstration dataset')
    episodes, names, seeds = [], set(), set()
    for entry in manifest['episodes']:
        path = (folder/entry['file']).resolve()
        if path.parent != folder or path.suffix != '.npz' or path.name in names:
            raise ValueError('Invalid or duplicate episode path')
        if file_hash(path) != entry['sha256']:
            raise ValueError('Demonstration file hash mismatch')
        arrays, meta = load_episode(path)
        if meta['seed'] in seeds or any(meta[k] != entry[k] for k in ('seed', 'kind', 'frames')):
            raise ValueError('Episode metadata mismatch/duplicate seed')
        expected_cargo = arrays['bodies'][:-1, :, 5, None] == np.arange(1, 4)
        if not np.array_equal(arrays['observations'][:, :, 126:129], expected_cargo):
            raise ValueError('Cargo channel disagrees with recorded body')
        episodes.append((arrays, meta))
        names.add(path.name)
        seeds.add(meta['seed'])
    if {meta['kind'] for _, meta in episodes} != set(KINDS):
        raise ValueError('All four layout families must be present')
    return episodes, manifest


def window_specs(meta, burn, frames, event_examples=2):
    length = burn+frames
    if min(burn, frames, event_examples) < 1 or meta['frames'] < length:
        raise ValueError('Invalid recurrent window')
    rng = np.random.default_rng(meta['seed']+410003)
    buckets = {}
    for event in meta['interactionEvents']:
        name = event['event']
        if name not in ('pickup', 'transfer', 'delivery'):
            continue
        stage = event['cargoAfter'] if name == 'pickup' else event['cargoBefore']
        tick = round(event['time']/meta['dt'])-1
        buckets.setdefault(f'{name}-{stage}', []).append(tick)
        if name == 'pickup':
            # Learn grip release after genuine pickup/cargo transition, not
            # just the positive contact frame. Input still contains all boxes.
            buckets.setdefault(f'after-pickup-{stage}', []).append(tick+frames)
    selected, seen = [], set()
    def add(category, start):
        start = int(np.clip(start, 0, meta['frames']-length))
        if start not in seen:
            selected.append({'category': category, 'start': start, 'stop': start+length,
                             'lossStart': start+burn, 'seed': meta['seed'], 'kind': meta['kind']})
            seen.add(start)
    for category, ticks in sorted(buckets.items()):
        ticks = np.unique(ticks)
        for tick in rng.choice(ticks, min(event_examples, len(ticks)), replace=False):
            add(category, tick-burn-frames//2)
    for start in np.linspace(0, meta['frames']-length, 4).astype(int):
        add('uniform-trajectory', start)
    return selected


def prefix_states(model, observations, starts, progress=False):
    """Yield state BEFORE selected frame; encode every preceding frame once."""
    starts = set(starts)
    if not starts or min(starts) < 0 or max(starts) >= len(observations):
        raise ValueError('Invalid prefix positions')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Prefix cache requires a frozen evaluation model')
    device = model.tonic.device
    with torch.no_grad():
        weights = model.weights()
        state = torch.zeros((model.n, observations.shape[1]), device=device)
        for frame in range(max(starts)+1):
            if frame in starts:
                yield frame, state
            if frame < max(starts):
                _, state = model(torch.as_tensor(observations[frame], device=device), 4, state, weights)
            if progress and frame and frame % 1200 == 0:
                print(json.dumps({'frozenTrainingPrefixFrames': frame,
                                  'teacherDataNotBrainService': True}), flush=True)


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve prefix caches')
    if not 1 <= args.worlds <= 4:
        raise ValueError('Cache at most four worlds per forward batch')
    episodes, dataset = load_dataset(args.dataset)
    model = load_model(args.root, args.candidate)
    model.eval().requires_grad_(False)
    before = model.checkpoint_hash()
    args.out.mkdir(parents=True)
    specs = [{**spec, 'episode': i} for i, (_, meta) in enumerate(episodes)
             for spec in window_specs(meta, args.burn, args.gradient_frames, args.event_examples)]
    x, y = [], []
    states = np.empty((len(specs), model.n, 4), np.float32)
    written = np.zeros(len(specs), bool)
    for spec in specs:
        arrays = episodes[spec['episode']][0]
        sl = slice(spec['start'], spec['stop'])
        x.append(arrays['observations'][sl])
        y.append(arrays['labels'][sl])
    manifest = {'schema': CACHE_SCHEMA, 'control': CONTROL, 'isBrainEvidence': False,
                'brainControlsTeacherWorlds': False, 'optimizerUsed': False,
                'interface': INTERFACE, 'physics': PHYSICS, 'parameterHash': before,
                'candidateFileHash': file_hash(args.candidate), 'fixedHash': model.fixed_hash,
                'datasetManifestHash': file_hash(args.dataset/'manifest.json'),
                'datasetFiles': [{'file': d['file'], 'sha256': d['sha256']} for d in dataset['episodes']],
                'burn': args.burn, 'gradientFrames': args.gradient_frames,
                'prefixSemantics': 'Full history from empty neural state at episode start, '
                                   'all frames encoded with one frozen checkpoint; state is before window start.',
                'sourceHash': file_hash(Path(__file__)),
                'sourceHashes': {name: file_hash(Path(__file__).parent/name) for name in
                                 ('layout_recovery_brain.py', 'layout_excitability.py',
                                  'layout_brain.py', 'supervised_steering.py')},
                'windows': specs, 'finished': False}
    atomic_json(args.out/'manifest.json', manifest)
    for offset in range(0, len(episodes), args.worlds):
        indexes = list(range(offset, min(offset+args.worlds, len(episodes))))
        lengths = {len(episodes[i][0]['observations']) for i in indexes}
        if len(lengths) != 1:
            raise ValueError('Grouped demonstration horizons must match')
        observations = np.concatenate([episodes[i][0]['observations'] for i in indexes], axis=1)
        group = [(j, spec) for j, spec in enumerate(specs) if spec['episode'] in indexes]
        pending = {}
        for j, spec in group:
            pending.setdefault(spec['start'], []).append((j, indexes.index(spec['episode'])))
        print('PREFIX_GROUP '+json.dumps({'episodes': indexes, 'windows': len(group)}), flush=True)
        for frame, state in prefix_states(model, observations, pending, progress=True):
            for j, local in pending[frame]:
                states[j] = state[:, 4*local:4*local+4].cpu().numpy()
                written[j] = True
        del observations
    if not written.all() or not np.isfinite(states).all() or model.checkpoint_hash() != before:
        raise AssertionError('Prefix integrity failed')
    model.audit(model.log_gains.detach().cpu().numpy())
    with (args.out/'cache.npz').open('xb') as handle:
        np.savez_compressed(handle, observations=np.asarray(x), labels=np.asarray(y), states=states)
    manifest.update(finished=True, parametersUnchanged=True,
                    cacheFileHash=file_hash(args.out/'cache.npz'))
    atomic_json(args.out/'manifest.json', manifest)
    print('FROZEN_PREFIX_CACHE_COMPLETE '+json.dumps({'windows': len(specs), 'parameterHash': before,
                                                    'isBrainEvidence': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'dataset', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--burn', type=int, default=64)
    parser.add_argument('--gradient-frames', type=int, default=32)
    parser.add_argument('--event-examples', type=int, default=2)
    main(parser.parse_args())
