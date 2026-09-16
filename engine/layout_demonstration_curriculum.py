"""Explicit, audited expansion of TRAINING demonstrations during Adam resume.

This changes data only. It does not change the graph, forward model, loss,
optimizer controls or physical inference, and never consumes evaluation seeds.
"""
from collections import Counter
from pathlib import Path
import numpy as np
from .layout_demonstration_cache import load_dataset, window_specs
from .layout_recovery_demonstrations import file_hash


DATA_SOURCES = ('layout_recovery_demonstrations.py', 'layout_recovery_teacher.py',
                'layout_recovery_world.py', 'layout_world.py', 'swarm_world.py',
                'haul_world.py', 'layout_interaction_trace.py')


def validate_curriculum_change(dataset, bank, old):
    """Verify larger, real teacher-only data and EVERY cached input/target."""
    folder = Path(dataset).resolve()
    fresh = bank.manifest
    manifest_hash = file_hash(folder/'manifest.json')
    if manifest_hash == old['datasetManifestHash'] or fresh['datasetManifestHash'] != manifest_hash:
        raise ValueError('Curriculum expansion requires a different, matching dataset')
    episodes, manifest = load_dataset(folder)  # Numeric/cargo checks and TRAINING seeds only.
    engine = Path(__file__).parent
    if (set(manifest.get('sourceHashes', {})) != set(DATA_SOURCES)
            or any(file_hash(engine/name) != manifest['sourceHashes'][name] for name in DATA_SOURCES)
            or fresh.get('sourceHash') != file_hash(engine/'layout_demonstration_cache.py')):
        raise ValueError('Curriculum data/cache generation source mismatch')
    files = [{'file': entry['file'], 'sha256': entry['sha256']} for entry in manifest['episodes']]
    if files != fresh['datasetFiles']:
        raise ValueError('Curriculum dataset files mismatch')
    old_seeds = {(w['kind'], w['seed']) for w in old['windows']}
    old_counts = Counter(kind for kind, _ in old_seeds)
    new_counts = Counter(meta['kind'] for _, meta in episodes)
    if set(old_counts) != set(new_counts) or any(new_counts[k] <= old_counts[k] for k in old_counts):
        raise ValueError('Expand the number of training episodes in every layout family')
    specs = [{**spec, 'episode': i} for i, (_, meta) in enumerate(episodes)
             for spec in window_specs(meta, fresh['burn'], fresh['gradientFrames'], event_examples=2)]
    if fresh['windows'] != specs or len(bank.x) != len(specs) or len(bank.y) != len(specs):
        raise ValueError('Curriculum event-balanced window catalog mismatch')
    for index, spec in enumerate(specs):
        arrays = episodes[spec['episode']][0]
        part = slice(spec['start'], spec['stop'])
        if (not np.array_equal(bank.x[index], arrays['observations'][part])
                or not np.array_equal(bank.y[index], arrays['labels'][part])):
            raise ValueError('Curriculum cached inputs/labels differ from physical demonstrations')
    return {'dataset': str(folder), 'previousDatasetManifestHash': old['datasetManifestHash'],
            'datasetManifestHash': manifest_hash, 'datasetFiles': files,
            'previousEpisodesPerFamily': dict(old_counts), 'episodesPerFamily': dict(new_counts),
            'seeds': [meta['seed'] for _, meta in episodes], 'windows': len(specs),
            'allCachedInputsAndTargetsMatched': True, 'teacherTrainingOnly': True,
            'isServiceEvidence': False, 'optimizerMomentsReset': False,
            'validatorSourceHash': file_hash(Path(__file__))}
