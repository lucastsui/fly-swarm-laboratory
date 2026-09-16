"""Versioned TRAINING-only timing labels for every physical demonstration.

The original dataset, observations, bodies and motion labels are immutable.
Every added grip command is replayed through the unchanged body and must be
ignored by its actual cooldown. No teacher or new controller enters inference.
This prepares data only; it neither starts an optimizer nor changes a brain.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .layout_cooldown_labels import VERSION, extend_labels, verify_equivalent_replay
from .layout_demonstration_cache import load_dataset
from .layout_demonstration_curriculum import DATA_SOURCES
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

SCHEMA = 'complete-physical-cooldown-label-dataset-v1'
LABEL_SOURCES = ('layout_cooldown_dataset.py', 'layout_cooldown_labels.py',
                 'layout_demonstration_cache.py', 'layout_demonstration_curriculum.py')


def source_identities(dataset_manifest):
    engine = Path(__file__).parent
    original = dataset_manifest.get('sourceHashes', {})
    if (set(original) != set(DATA_SOURCES)
            or any(file_hash(engine/n) != original[n] for n in DATA_SOURCES)):
        raise ValueError('Original physical demonstration sources changed')
    return {n: file_hash(engine/n) for n in LABEL_SOURCES}


def build_dataset(dataset, out):
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve all label datasets')
    original_hash = file_hash(dataset/'manifest.json')
    episodes, original = load_dataset(dataset)
    sources = source_identities(original)
    result = {'schema': SCHEMA, 'labelVersion': VERSION, 'extraFrames': 4,
              'datasetManifestHash': original_hash, 'originalDatasetFiles': original['episodes'],
              'originalPhysicsSourceHashes': original['sourceHashes'], 'sourceHashes': sources,
              'allEpisodesIncluded': True, 'teacherAtInference': False, 'newRuntimeController': False,
              'physicalTrajectoriesUnchanged': True, 'motionLabelsUnchanged': True,
              'optimizerUsed': False, 'isServiceEvidence': False, 'episodes': [], 'finished': False}
    out.mkdir(parents=True)
    atomic_json(out/'manifest.json', result)
    try:
        for index, (arrays, meta) in enumerate(episodes):
            target = extend_labels(arrays['labels'], meta, 4)
            audit = verify_equivalent_replay(arrays, meta, target)
            name = f'labels-{index:03d}.npz'
            with (out/name).open('xb') as stream:
                np.savez_compressed(stream, revised=target)
            entry = {'episode': index, 'file': name, 'sha256': file_hash(out/name), **audit,
                     'originalPositiveLabels': int((arrays['labels'][..., 2] > 1).sum()),
                     'revisedPositiveLabels': int((target[..., 2] > 1).sum())}
            result['episodes'].append(entry)
            atomic_json(out/'manifest.json', result)
            print('COMPLETE_DATASET_COOLDOWN_REPLAY '+json.dumps(entry), flush=True)
        if (file_hash(dataset/'manifest.json') != original_hash
                or any(file_hash(dataset/e['file']) != e['sha256'] for e in original['episodes'])
                or source_identities(original) != sources):
            raise ValueError('Original dataset or generator changed during replay')
        result['finished'] = True
        atomic_json(out/'manifest.json', result)
        print('COOLDOWN_DATASET_FINISHED '+json.dumps({'episodes': len(episodes),
              'manifestHash': file_hash(out/'manifest.json'), 'isServiceEvidence': False}), flush=True)
        return result
    except BaseException as error:
        atomic_json(out/'failure.json', {'type': type(error).__name__, 'error': str(error),
                                        'completedEpisodes': len(result['episodes'])})
        raise


def load_revision(folder, dataset):
    """Validate all data/labels before any future TRAINING consumer uses them.

    Recompute every revised target from the verified original physical events;
    a hash update cannot disguise modified motion labels or invented grip labels.
    The expensive full body replay was performed by the recorded generator.
    """
    folder, dataset = Path(folder).resolve(), Path(dataset).resolve()
    m = json.loads((folder/'manifest.json').read_text())
    expected = {'schema': SCHEMA, 'labelVersion': VERSION, 'extraFrames': 4,
                'allEpisodesIncluded': True, 'teacherAtInference': False, 'newRuntimeController': False,
                'physicalTrajectoriesUnchanged': True, 'motionLabelsUnchanged': True,
                'optimizerUsed': False, 'isServiceEvidence': False, 'finished': True}
    if any(m.get(k) != v for k, v in expected.items()) or (folder/'failure.json').exists():
        raise ValueError('Incomplete or incompatible timing label dataset')
    episodes, original = load_dataset(dataset)
    if (m['datasetManifestHash'] != file_hash(dataset/'manifest.json')
            or m['originalDatasetFiles'] != original['episodes']
            or m['originalPhysicsSourceHashes'] != original['sourceHashes']
            or m['sourceHashes'] != source_identities(original)
            or len(m['episodes']) != len(episodes)):
        raise ValueError('Revision dataset/source identity mismatch')
    targets = []
    for index, ((arrays, meta), entry) in enumerate(zip(episodes, m['episodes'])):
        path = (folder/entry['file']).resolve()
        if (path.parent != folder or path.name != f'labels-{index:03d}.npz'
                or file_hash(path) != entry['sha256'] or entry['episode'] != index
                or any(entry[k] != meta[k] for k in ('seed', 'kind', 'frames'))
                or entry.get('allChangesIgnoredByActualCooldown') is not True
                or entry.get('everyObservationBodyAndEventExactlyMatched') is not True
                or entry.get('isServiceEvidence') is not False):
            raise ValueError('Invalid episode label/replay provenance')
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != {'revised'}:
                raise ValueError('Unexpected revised label arrays')
            revised = values['revised'].copy()
        computed = extend_labels(arrays['labels'], meta, 4)
        if (revised.dtype != np.float32 or not np.isfinite(revised).all()
                or not np.array_equal(revised, computed)
                or entry['changedGripLabels'] != int(np.count_nonzero(revised[..., 2] != arrays['labels'][..., 2]))
                or entry['originalPositiveLabels'] != int((arrays['labels'][..., 2] > 1).sum())
                or entry['revisedPositiveLabels'] != int((revised[..., 2] > 1).sum())):
            raise ValueError('Revised targets differ from original verified events')
        targets.append(revised)
    return targets, m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build_dataset(args.dataset, args.out)
