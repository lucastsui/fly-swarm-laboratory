"""Strict original-event cache reader and balanced independent-actor batches."""
import json
from pathlib import Path
import numpy as np
import torch
from .layout_operation_prefix import VERSION, CONTEXTS, operation_specs
from .layout_correction_prefix import load_histories
from .layout_recovery_teacher import KINDS, LABEL_VERSION
from .layout_recovery_world import INTERFACE, PHYSICS
from .layout_recovery_demonstrations import file_hash


class OperationCache:
    def __init__(self, folder, dataset, parent_hash, fixed_hash, neurons=166700):
        folder, dataset = Path(folder), Path(dataset)
        m = json.loads((folder/'manifest.json').read_text())
        expected = {'schema': VERSION, 'parameterHash': parent_hash, 'fixedHash': fixed_hash,
                    'finished': True, 'parametersUnchanged': True, 'interface': INTERFACE, 'physics': PHYSICS,
                    'control': 'learner-only', 'teacherActions': False, 'optimizerUsed': False,
                    'labelsUnchanged': True, 'labelVersion': LABEL_VERSION, 'isServiceEvidence': False,
                    'burn': 64, 'lossFrames': 32, 'fullHistoryPrefix': True}
        if any(m.get(k) != v for k, v in expected.items()) or (folder/'failure.json').exists():
            raise ValueError('Incomplete, incompatible or assisted operation cache')
        required = {'layout_operation_prefix.py', 'layout_correction_prefix.py', 'supervised_joint.py'}
        if not required.issubset(m['sourceHashes']) or any(Path(n).name != n or
                file_hash(Path(__file__).parent/n) != h for n, h in m['sourceHashes'].items()):
            raise ValueError('Changed operation prefix source')
        episodes, data = load_histories(dataset, replay=False)
        if (file_hash(dataset/'manifest.json') != m['datasetManifestHash'] or data['episodes'] != m['datasetFiles']
                or data['parameterHash'] != m['behaviorParameterHash'] or data['fixedHash'] != fixed_hash
                or data['canonicalRun'] != m['behaviorCanonicalRun']
                or data['canonicalVersion'] != m['behaviorCanonicalVersion']
                or operation_specs(episodes) != m['windows'] or file_hash(folder/'cache.npz') != m['cacheFileHash']):
            raise ValueError('Original operation histories/provenance changed')
        with np.load(folder/'cache.npz', allow_pickle=False) as z:
            if set(z.files) != {'observations', 'labels', 'states', 'parent_motion'}:
                raise ValueError('Wrong cache arrays')
            self.x, self.y, self.states, self.motion = [z[k].copy() for k in ('observations','labels','states','parent_motion')]
        n = len(m['windows'])
        for a, shape in zip((self.x,self.y,self.states,self.motion),
                            ((n,96,1,297),(n,96,1,3),(n,neurons,1),(n,32,1,2))):
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('Invalid operation cache numeric values')
        self.groups = {(k,c): [] for k in KINDS for c in CONTEXTS}
        for i, s in enumerate(m['windows']):
            original = episodes[s['episode']][0]; sl = slice(s['start'],s['stop']); f = s['fly']
            if (not np.array_equal(self.x[i], original['observations'][sl, f:f+1])
                    or not np.array_equal(self.y[i], original['labels'][sl, f:f+1])):
                raise ValueError('Original sensory observations or labels altered')
            self.groups[s['kind'],s['category']].append(i)
        if any(not ids for ids in self.groups.values()):
            raise ValueError('Missing actual service operation context')
        self.manifest = m

    def batch(self, ids, device):
        return tuple(torch.as_tensor(np.concatenate([a[i] for i in ids],axis=1), device=device)
                     for a in (self.x,self.y,self.states,self.motion))

    def sample(self, rng, device):
        ids = [int(rng.choice(self.groups[k,c])) for k in KINDS for c in CONTEXTS]
        return (*self.batch(ids,device), {'source': VERSION, 'actors': 16,
                   'originalLabelsUnchanged': True, 'teacherActions': False,
                   'selections': [{'windowIndex': i, **self.manifest['windows'][i]} for i in ids]})

    def diagnostic_indexes(self):
        return sorted({i for ids in self.groups.values() for i in (ids[0],ids[-1])})
