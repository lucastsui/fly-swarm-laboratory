import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .layout_demonstration_curriculum import DATA_SOURCES, validate_curriculum_change
from .layout_demonstration_cache import CACHE_SCHEMA, window_specs, prefix_states
from .layout_demonstration_train import DemonstrationCache, CACHE_MODEL_SOURCES, validate_bounds
from .layout_recovery_demonstrations import collect_episode, file_hash, SCHEMA, CONTROL
from .layout_recovery_teacher import KINDS
from .layout_recovery_world import INTERFACE, PHYSICS
from .test_layout_microfit_flow import TinyTrainBrain


def physical_curriculum_fixture(folder, model, candidate):
    """Real tiny teacher trajectories and full-history tiny-brain prefixes."""
    dataset, cache = folder/'expanded-data', folder/'expanded-cache'
    dataset.mkdir()
    cache.mkdir()
    engine = Path(__file__).parent
    manifest = {'schema': SCHEMA, 'control': CONTROL, 'finished': True, 'isBrainEvidence': False,
                'sourceHashes': {n: file_hash(engine/n) for n in DATA_SOURCES}, 'episodes': []}
    specs, x, y, states = [], [], [], []
    training = model.training
    requires = [p.requires_grad for p in model.parameters()]
    model.eval().requires_grad_(False)
    try:
        for j, kind in enumerate(KINDS):
            for i in range(2):
                arrays, meta = collect_episode(9360100+1000*j+i, kind, 8, random_starts=bool(i))
                name = f'{kind}-{meta["seed"]}.npz'
                np.savez_compressed(dataset/name, **arrays, metadata=np.asarray(json.dumps(meta)))
                manifest['episodes'].append({'file': name, 'sha256': file_hash(dataset/name),
                                              'seed': meta['seed'], 'kind': kind, 'frames': 8})
                episode = len(manifest['episodes'])-1
                windows = window_specs(meta, 2, 2, 2)
                prefixes = {s: v.cpu().numpy().copy() for s, v in
                            prefix_states(model, arrays['observations'], [w['start'] for w in windows])}
                for spec in windows:
                    specs.append({**spec, 'episode': episode})
                    part = slice(spec['start'], spec['stop'])
                    x.append(arrays['observations'][part])
                    y.append(arrays['labels'][part])
                    states.append(prefixes[spec['start']])
    finally:
        model.train(training)
        for p, flag in zip(model.parameters(), requires):
            p.requires_grad_(flag)
    (dataset/'manifest.json').write_text(json.dumps(manifest))
    np.savez_compressed(cache/'cache.npz', observations=np.asarray(x), labels=np.asarray(y),
                        states=np.asarray(states))
    meta = {'schema': CACHE_SCHEMA, 'control': CONTROL, 'finished': True,
            'isBrainEvidence': False, 'optimizerUsed': False, 'parametersUnchanged': True,
            'brainControlsTeacherWorlds': False, 'parameterHash': model.checkpoint_hash(),
            'candidateFileHash': file_hash(candidate), 'fixedHash': model.fixed_hash,
            'interface': INTERFACE, 'physics': PHYSICS, 'burn': 2, 'gradientFrames': 2,
            'datasetManifestHash': file_hash(dataset/'manifest.json'),
            'datasetFiles': [{k: e[k] for k in ('file', 'sha256')} for e in manifest['episodes']],
            'cacheFileHash': file_hash(cache/'cache.npz'), 'windows': specs,
            'sourceHash': file_hash(engine/'layout_demonstration_cache.py'),
            'sourceHashes': {n: file_hash(engine/n) for n in CACHE_MODEL_SOURCES}}
    (cache/'manifest.json').write_text(json.dumps(meta))
    return dataset, cache


class CurriculumTests(unittest.TestCase):
    def test_real_data_prefix_and_every_window_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            candidate = folder/'candidate.npz'
            candidate.write_bytes(b'test-parent')
            model = TinyTrainBrain()
            dataset, cache = physical_curriculum_fixture(folder, model, candidate)
            bank = DemonstrationCache(cache, model.checkpoint_hash(), model.fixed_hash, 2, 2, 4)
            old = {'datasetManifestHash': 'old', 'windows': [
                {'kind': k, 'seed': 9360000+1000*i} for i, k in enumerate(KINDS)]}
            record = validate_curriculum_change(dataset, bank, old)
            self.assertEqual(record['episodesPerFamily'], dict.fromkeys(KINDS, 2))
            self.assertEqual(record['previousEpisodesPerFamily'], dict.fromkeys(KINDS, 1))
            self.assertFalse(record['isServiceEvidence'])
            self.assertTrue(record['allCachedInputsAndTargetsMatched'])
            self.assertEqual(record['windows'], 32)
            original = copy.deepcopy(bank.manifest)
            for change in ({'datasetManifestHash': 'wrong'}, {'sourceHash': 'wrong'},
                           {'datasetFiles': []}, {'windows': []}):
                bank.manifest = {**original, **change}
                with self.assertRaises(ValueError):
                    validate_curriculum_change(dataset, bank, old)
            bank.manifest = original
            for value in (bank.x, bank.y):
                saved = value[0, 0, 0, 0]
                value[0, 0, 0, 0] += 1
                with self.assertRaisesRegex(ValueError, 'inputs/labels'):
                    validate_curriculum_change(dataset, bank, old)
                value[0, 0, 0, 0] = saved
            for unchanged in ({**old, 'datasetManifestHash': original['datasetManifestHash']},
                              {**old, 'windows': original['windows']}):
                with self.assertRaises(ValueError):
                    validate_curriculum_change(dataset, bank, unchanged)
            # Wrong physics/held-out seeds cannot be laundered by updating hashes.
            data_manifest = json.loads((dataset/'manifest.json').read_text())
            entry = data_manifest['episodes'][0]
            path = dataset/entry['file']
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files if k != 'metadata'}
                metadata = json.loads(str(archive['metadata']))
            for change in ({'seed': 9801000}, {'teacherActions': False}, {'injectedCargo': True}):
                np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps({**metadata, **change})))
                entry['sha256'] = file_hash(path)
                (dataset/'manifest.json').write_text(json.dumps(data_manifest))
                bank.manifest['datasetManifestHash'] = file_hash(dataset/'manifest.json')
                with self.assertRaises(ValueError):
                    validate_curriculum_change(dataset, bank, old)

    def test_expansion_flag_without_resume_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'completed source'):
            validate_bounds(SimpleNamespace(resume_new_demonstration_dataset=Path('new-data')))


if __name__ == '__main__':
    unittest.main()
