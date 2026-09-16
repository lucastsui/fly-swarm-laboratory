import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .layout_demonstration_train import (DemonstrationCache, CACHE_MODEL_SOURCES,
                                        training_scores, validate_bounds, main)
from .layout_demonstration_cache import CACHE_SCHEMA
from .layout_recovery_demonstrations import CONTROL, file_hash
from .layout_recovery_world import INTERFACE, PHYSICS
from .layout_recovery_teacher import KINDS
from .layout_recovery_curriculum import STRATIFIED_VERSION
from .test_layout_microfit_flow import TinyTrainBrain


def tiny_cache(folder, model):
    folder.mkdir()
    x = np.ones((4, 4, 4, 297), np.float32)
    y = np.full((4, 4, 4, 3), .2, np.float32)
    y[:, :, ::2, 2] = 2.2
    np.savez_compressed(folder/'cache.npz', observations=x, labels=y,
                        states=np.zeros((4, 4, 4), np.float32))
    value = {'schema': CACHE_SCHEMA, 'control': CONTROL, 'finished': True,
             'isBrainEvidence': False, 'optimizerUsed': False, 'parametersUnchanged': True,
             'brainControlsTeacherWorlds': False, 'parameterHash': model.checkpoint_hash(),
             'fixedHash': model.fixed_hash, 'interface': INTERFACE, 'physics': PHYSICS,
             'burn': 2, 'gradientFrames': 2, 'datasetManifestHash': 'unit-test-dataset',
             'cacheFileHash': file_hash(folder/'cache.npz'),
             'sourceHashes': {name: file_hash(Path(__file__).parent/name) for name in CACHE_MODEL_SOURCES},
             'windows': [{'seed': 9360000+i*1000, 'kind': kind, 'start': 0, 'stop': 4,
                          'lossStart': 2, 'category': 'fixture'} for i, kind in enumerate(KINDS)]}
    (folder/'manifest.json').write_text(json.dumps(value))
    return value


class DemonstrationTrainingTests(unittest.TestCase):
    def test_loader_samples_one_world_per_family_and_rejects_wrong_parent(self):
        model = TinyTrainBrain()
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)/'cache'
            meta = tiny_cache(folder, model)
            bank = DemonstrationCache(folder, model.checkpoint_hash(), model.fixed_hash, 2, 2, 4)
            x, y, state, detail = bank.sample(np.random.default_rng(1), 'cpu')
            self.assertEqual(tuple(x.shape), (4, 16, 297))
            self.assertEqual(tuple(y.shape), (4, 16, 3))
            self.assertEqual(tuple(state.shape), (4, 16))
            self.assertEqual({w['kind'] for w in detail['windows']}, set(KINDS))
            with self.assertRaises(ValueError):
                DemonstrationCache(folder, 'wrong-parent', model.fixed_hash, 2, 2, 4)
            for change in ({'isBrainEvidence': True}, {'control': 'learner-only'}, {'finished': False},
                           {'sourceHashes': {}}, {'cacheFileHash': 'wrong'}):
                (folder/'manifest.json').write_text(json.dumps({**meta, **change}))
                with self.assertRaises(ValueError):
                    DemonstrationCache(folder, model.checkpoint_hash(), model.fixed_hash, 2, 2, 4)

    def test_actual_main_saves_and_logs_separate_teacher_and_correction_paths(self):
        model = TinyTrainBrain()
        model.interface = INTERFACE
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            folder = Path(root)
            cache = folder/'cache'
            tiny_cache(cache, model)
            token = folder/'token'
            token.write_text('unit-test-token')
            args = SimpleNamespace(root=folder, candidate=folder/'parent.npz', cache=cache,
                                   out=folder/'run', token_file=token, cert=folder/'cert', key=folder/'key',
                                   updates=4, burn=2, gradient_frames=2, save_every=2,
                                   lr=.0003, tonic_lr=.000001, seed=9380001)
            packet = (np.ones((4, 4, 297), np.float32), np.full((4, 4, 3), .2, np.float32),
                      np.zeros((4, 4), np.float32), {'control': 'learner-only', 'teacherActions': False,
                                                  'version': 0, 'id': 'unit-test-rollout', 'worlds': []})
            prefix = 'engine.layout_demonstration_train.'
            stack.enter_context(patch(prefix+'load_model', return_value=model))
            stack.enter_context(patch(prefix+'save_model', side_effect=lambda path, brain: path.write_bytes(b'fixture')))
            stack.enter_context(patch(prefix+'start_server', return_value=SimpleNamespace(shutdown=lambda: None)))
            stack.enter_context(patch(prefix+'Exchange.pop', return_value=packet))
            stack.enter_context(patch(prefix+'torch.cuda.get_device_name', return_value='CPU test'))
            stack.enter_context(patch(prefix+'torch.cuda.max_memory_allocated', return_value=0))
            stack.enter_context(patch(prefix+'time.sleep'))
            stack.enter_context(patch('builtins.print'))
            main(args)
            result = json.loads((args.out/'result.json').read_text())
            self.assertEqual((result['demoUpdates'], result['correctionUpdates']), (3, 1))
            history = json.loads((args.out/'history.json').read_text())
            self.assertEqual(history[-1]['source'], 'spark2-learner-only-correction')
            self.assertTrue(all(not row['trainingScores']['isServiceEvidence'] for row in history))
            self.assertTrue((args.out/'optimizer-4.pt').exists())
            self.assertTrue((args.out/'candidate-4.npz').exists())
            self.assertTrue(json.loads((args.out/'status.json').read_text())['finished'])
            with self.assertRaises(FileExistsError):
                main(args)
            fresh = TinyTrainBrain()
            fresh.interface = INTERFACE
            recent_args = SimpleNamespace(**{**vars(args), 'out': folder/'recent-run', 'recent_corrections': True,
                                             'require_stratified_corrections': True})
            stratified_packet = (np.tile(packet[0], (1, 4, 1)), np.tile(packet[1], (1, 4, 1)),
                                 np.tile(packet[2], (1, 4)), {**packet[3], 'worlds': [
                {'kind': kind, 'familyStratified': True, 'curriculum': STRATIFIED_VERSION} for kind in KINDS]})
            with patch(prefix+'load_model', return_value=fresh), patch(prefix+'pop_recent_correction', return_value=stratified_packet) as recent:
                main(recent_args)
            recent.assert_called_once()
            recent_manifest = json.loads((recent_args.out/'manifest.json').read_text())
            self.assertEqual(recent_manifest['correctionSampling'], 'newest queued packet at selection time')
            rejected_args = SimpleNamespace(**{**vars(recent_args), 'out': folder/'reject-run'})
            rejected_model = TinyTrainBrain()
            rejected_model.interface = INTERFACE
            with patch(prefix+'load_model', return_value=rejected_model), patch(prefix+'pop_recent_correction', return_value=packet):
                with self.assertRaisesRegex(ValueError, 'four-family'):
                    main(rejected_args)
            self.assertTrue(json.loads((rejected_args.out/'status.json').read_text())['finished'])
            self.assertTrue((rejected_args.out/'failure.json').exists())
            self.assertFalse((rejected_args.out/'candidate-4.npz').exists())

    def test_prefix_age_cannot_be_silently_extended(self):
        args = SimpleNamespace(updates=81, burn=64, gradient_frames=32, save_every=20,
                               lr=.0003, tonic_lr=.000001)
        with self.assertRaises(ValueError):
            validate_bounds(args)
        args.updates = 80
        validate_bounds(args)

    def test_absent_grip_class_is_unavailable_not_perfect_score(self):
        target = torch.full((4, 4, 3), .2)
        scores = training_scores(target, target)
        self.assertIsNone(scores['gripRecall'])
        self.assertEqual(scores['gripFalsePositiveRate'], 0.)
        self.assertFalse(scores['isServiceEvidence'])


if __name__ == '__main__':
    unittest.main()
