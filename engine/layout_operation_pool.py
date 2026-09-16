"""Explicit two-bank training pool; original source caches remain immutable.

Each input independently passes OperationCache's original source, data, array
and canonical-version checks. No old file is retagged as a larger dataset.
Labels and selection metadata are never inputs to the connectome.
"""
from pathlib import Path
import numpy as np
from .layout_operation_sampling import OperationCache
from .layout_operation_prefix import KINDS, CONTEXTS
from .layout_recovery_demonstrations import file_hash


def pool_metadata(manifests):
    if len(manifests) != 2:
        raise ValueError('Exactly two independently verified banks required')
    keys = ('parameterHash', 'fixedHash', 'behaviorParameterHash', 'behaviorCanonicalRun',
            'behaviorCanonicalVersion', 'prefixCanonicalRun', 'prefixCanonicalVersion',
            'candidateFileHash', 'interface', 'physics', 'labelVersion', 'burn', 'lossFrames')
    first = manifests[0]
    if (any(any(m[k] != first[k] for k in keys) for m in manifests)
            or first['parameterHash'] != first['behaviorParameterHash']
            or first['prefixCanonicalRun'] != first['behaviorCanonicalRun']
            or first['prefixCanonicalVersion'] != first['behaviorCanonicalVersion']
            or first['burn'] != 64 or first['lossFrames'] != 32):
        raise ValueError('Same-version current-policy banks with original windows required')
    windows, seen_seeds, seen_data, offset = [], set(), set(), 0
    for bank_index, m in enumerate(manifests):
        if m['datasetManifestHash'] in seen_data:
            raise ValueError('Duplicated source bank')
        seen_data.add(m['datasetManifestHash'])
        files = m['datasetFiles']
        seeds = [e['seed'] for e in files]
        if len(set(seeds)) != len(seeds) or seen_seeds.intersection(seeds):
            raise ValueError('Independent physical episodes required; no duplicate seeds')
        seen_seeds.update(seeds)
        for index, original in enumerate(m['windows']):
            if (not 0 <= original['episode'] < len(files)
                    or original['seed'] != files[original['episode']]['seed']
                    or original['kind'] != files[original['episode']]['kind']):
                raise ValueError('Original episode mapping changed')
            windows.append({**original, 'episode': offset+original['episode'],
                            'sourceBank': bank_index, 'sourceWindow': index,
                            'sourceEpisode': original['episode'],
                            'sourceDatasetManifestHash': m['datasetManifestHash']})
        offset += len(files)
    return {**{k: first[k] for k in keys}, 'schema': 'explicit-two-operation-bank-pool-v1',
            'windows': windows, 'episodes': offset, 'sourceBanks': manifests,
            'physicalHistoriesAndLabelsUnchanged': True, 'isServiceEvidence': False,
            'episodeMapping': 'sourceEpisode plus cumulative source dataset episode count'}


class OperationPool(OperationCache):
    def __init__(self, caches, datasets, parent_hash, fixed_hash, neurons=166700):
        if len(caches) != 2 or len(datasets) != 2:
            raise ValueError('Two explicit cache/dataset pairs required')
        caches, datasets = [Path(p) for p in caches], [Path(p) for p in datasets]
        banks = [OperationCache(c, d, parent_hash, fixed_hash, neurons) for c,d in zip(caches,datasets)]
        self.manifest = pool_metadata([b.manifest for b in banks])
        self.manifest['sourceLocations'] = [
            {'cache': str(c.resolve()), 'dataset': str(d.resolve()),
             'cacheManifestHash': file_hash(c/'manifest.json'), 'cacheFileHash': file_hash(c/'cache.npz')}
            for c,d in zip(caches,datasets)]
        for name in ('x', 'y', 'states', 'motion'):
            setattr(self, name, np.concatenate([getattr(b, name) for b in banks], axis=0))
        self.groups = {(k,c): [] for k in KINDS for c in CONTEXTS}
        for i, w in enumerate(self.manifest['windows']):
            self.groups[w['kind'],w['category']].append(i)
        if any(not indexes for indexes in self.groups.values()):
            raise ValueError('Every original family/context required')
        # The unchanged single-step solver uses the first window per group.
        # Alternate which independent source supplies it: eight contexts from
        # each bank, rather than silently taking every gradient from bank zero.
        for ki,k in enumerate(KINDS):
            for ci,c in enumerate(CONTEXTS):
                preferred=(ki+ci)%2
                self.groups[k,c].sort(key=lambda i:(self.manifest['windows'][i]['sourceBank']!=preferred,i))
        self.manifest['selectedGradientSourceRule']='sourceBank=(familyIndex+contextIndex)%2; first original window in that source'
