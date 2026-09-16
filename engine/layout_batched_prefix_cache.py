"""Faster versioned TRAINING-prefix generation using independent world batches.

Same full connectome, observations and before-window state as the original
cache generator. More worlds share a forward pass; no brain state is shared
between actors. This frozen preprocessing job owns NO optimizer.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_demonstration_cache import load_dataset, window_specs, prefix_states, CACHE_SCHEMA
from .layout_demonstration_curriculum import DATA_SOURCES
from .layout_demonstration_train import CACHE_MODEL_SOURCES
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash, CONTROL
from .layout_recovery_world import INTERFACE, PHYSICS
from .layout_recovery_protocol import atomic_json


def fill_states(model, episodes, specs, worlds=32):
    if type(worlds) is not int or not 1 <= worlds <= 32:
        raise ValueError('Bounded1..32 independent-world inference batches required')
    if not episodes or not specs:
        raise ValueError('Physical episodes and windows required')
    states = np.empty((len(specs), model.n, 4), np.float32)
    written = np.zeros(len(specs), bool)
    for offset in range(0, len(episodes), worlds):
        indexes = list(range(offset, min(offset+worlds, len(episodes))))
        if len({len(episodes[i][0]['observations']) for i in indexes}) != 1:
            raise ValueError('Batch horizons must match')
        observations = np.concatenate([episodes[i][0]['observations'] for i in indexes], axis=1)
        pending = {}
        for j, spec in enumerate(specs):
            if spec['episode'] in indexes:
                pending.setdefault(spec['start'], []).append((j, indexes.index(spec['episode'])))
        if not pending:
            raise ValueError('Every physical batch requires training windows')
        print('BATCHED_PREFIX_GROUP '+json.dumps({'episodes': indexes, 'actors': len(indexes)*4,
                                                'windows': sum(map(len, pending.values()))}), flush=True)
        for frame, state in prefix_states(model, observations, pending, progress=True):
            for j, local in pending[frame]:
                states[j] = state[:, 4*local:4*local+4].cpu().numpy()
                written[j] = True
        del observations
    if not written.all() or not np.isfinite(states).all():
        raise ValueError('Incomplete or nonfinite full-history cache')
    return states


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve all prefix caches')
    if type(args.worlds) is not int or not 1 <= args.worlds <= 32:
        raise ValueError('Bounded1..32 world batch required')
    torch.set_num_threads(4)
    episodes, dataset = load_dataset(args.dataset)
    engine = Path(__file__).parent
    if (set(dataset.get('sourceHashes', {})) != set(DATA_SOURCES)
            or any(file_hash(engine/n) != dataset['sourceHashes'][n] for n in DATA_SOURCES)):
        raise ValueError('Physical data source mismatch')
    candidate_hash = file_hash(args.candidate)
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    specs = [{**s, 'episode': i} for i, (_, meta) in enumerate(episodes)
             for s in window_specs(meta, 64, 32, 2)]
    inputs, labels = [], []
    for s in specs:
        arrays = episodes[s['episode']][0]
        inputs.append(arrays['observations'][s['start']:s['stop']])
        labels.append(arrays['labels'][s['start']:s['stop']])
    m = {'schema': CACHE_SCHEMA, 'control': CONTROL, 'isBrainEvidence': False,
         'brainControlsTeacherWorlds': False, 'optimizerUsed': False, 'interface': INTERFACE, 'physics': PHYSICS,
         'parameterHash': before, 'candidateFileHash': candidate_hash, 'fixedHash': model.fixed_hash,
         'datasetManifestHash': file_hash(args.dataset/'manifest.json'),
         'datasetFiles': [{'file': e['file'], 'sha256': e['sha256']} for e in dataset['episodes']],
         'burn': 64, 'gradientFrames': 32, 'windows': specs, 'finished': False,
         'prefixSemantics': 'Full history from empty neural state at episode start; state BEFORE window start.',
         'sourceHash': file_hash(Path(__file__)),
         'generatorVersion': 'independent-batched-full-prefix-v1', 'worldsPerForwardBatch': args.worlds,
         'sourceHashes': {n: file_hash(engine/n) for n in sorted(CACHE_MODEL_SOURCES)},
         'helperSourceHashes': {n: file_hash(engine/n) for n in ('layout_demonstration_cache.py',
                                                              'layout_demonstration_train.py',
                                                              'layout_demonstration_curriculum.py')},
         'originalPhysicsSourceHashes': dataset['sourceHashes'], 'device': torch.cuda.get_device_name()}
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json', m)
    try:
        states = fill_states(model, episodes, specs, args.worlds)
        if (model.checkpoint_hash() != before or model.fingerprint() != model.fixed_hash
                or file_hash(Path(__file__)) != m['sourceHash']
                or file_hash(args.candidate) != candidate_hash
                or file_hash(args.dataset/'manifest.json') != m['datasetManifestHash']
                or any(file_hash(args.dataset/e['file']) != e['sha256'] for e in m['datasetFiles'])
                or any(file_hash(engine/n) != h for n, h in {**m['sourceHashes'], **m['helperSourceHashes'],
                                                            **m['originalPhysicsSourceHashes']}.items())):
            raise ValueError('Frozen model/data/source identity changed')
        audit = model.audit(model.log_gains.detach().cpu().numpy())
        with (args.out/'cache.npz').open('xb') as stream:
            np.savez_compressed(stream, observations=np.asarray(inputs), labels=np.asarray(labels), states=states)
        m.update(finished=True, parametersUnchanged=True, audit=audit, cacheFileHash=file_hash(args.out/'cache.npz'))
        atomic_json(args.out/'manifest.json', m)
        print('BATCHED_PREFIX_FINISHED '+json.dumps({'windows': len(specs), 'parameterHash': before,
                                                   'cacheFileHash': m['cacheFileHash']}), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'dataset', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=32)
    main(parser.parse_args())
