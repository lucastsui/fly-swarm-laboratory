import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .layout_cooldown_dataset import build_dataset, load_revision
from .layout_demonstration_curriculum import DATA_SOURCES
from .layout_recovery_demonstrations import collect_episode, file_hash, SCHEMA, CONTROL
from .layout_recovery_teacher import KINDS


def physical_fixture(folder):
    folder.mkdir()
    m = {'schema': SCHEMA, 'control': CONTROL, 'finished': True, 'isBrainEvidence': False,
         'sourceHashes': {n: file_hash(Path(__file__).parent/n) for n in DATA_SOURCES}, 'episodes': []}
    for j, kind in enumerate(KINDS):
        arrays, meta = collect_episode(9360123+1000*j, kind, 500, random_starts=False)
        name = f'{kind}.npz'
        np.savez_compressed(folder/name, **arrays, metadata=np.asarray(json.dumps(meta)))
        m['episodes'].append({'file': name, 'sha256': file_hash(folder/name),
                              **{k: meta[k] for k in ('seed', 'kind', 'frames')}})
    (folder/'manifest.json').write_text(json.dumps(m))


class CompleteCooldownDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data, self.out = self.folder/'physical', self.folder/'revision'
        physical_fixture(self.data)
        self.hashes = {p.name: file_hash(p) for p in self.data.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.manifest = build_dataset(self.data, self.out)

    def test_all_real_episodes_replayed_and_original_files_immutable(self):
        targets, m = load_revision(self.out, self.data)
        self.assertEqual(len(targets), 4)
        self.assertEqual({e['kind'] for e in m['episodes']}, set(KINDS))
        self.assertGreater(sum(e['changedGripLabels'] for e in m['episodes']), 0)
        self.assertFalse(m['optimizerUsed'])
        self.assertFalse(m['isServiceEvidence'])
        self.assertEqual(self.hashes, {p.name: file_hash(p) for p in self.data.iterdir()})
        with self.assertRaises(FileExistsError):
            build_dataset(self.data, self.out)

    def test_reject_tampered_labels_even_with_updated_file_hash(self):
        path = self.out/'labels-000.npz'
        with np.load(path) as values:
            target = values['revised'].copy()
        target[0, 0, 0] += .25
        np.savez_compressed(path, revised=target)
        m = copy.deepcopy(self.manifest)
        m['episodes'][0]['sha256'] = file_hash(path)
        (self.out/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'targets differ'):
            load_revision(self.out, self.data)

    def test_reject_incomplete_changed_sources_missing_episodes_and_path_escape(self):
        for change in ({'finished': False}, {'sourceHashes': {}}, {'episodes': []},
                       {'datasetManifestHash': 'wrong'}, {'extraFrames': 5}):
            (self.out/'manifest.json').write_text(json.dumps({**self.manifest, **change}))
            with self.assertRaises(ValueError):
                load_revision(self.out, self.data)
        m = copy.deepcopy(self.manifest)
        m['episodes'][0]['file'] = '../escape.npz'
        (self.out/'manifest.json').write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            load_revision(self.out, self.data)


if __name__ == '__main__':
    unittest.main()
