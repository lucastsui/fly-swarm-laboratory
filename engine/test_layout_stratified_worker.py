import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .layout_stratified_worker import worker_world, validate_family_coverage
from .layout_recovery_curriculum import balanced_world, RECOVERY_FAILURES
from .layout_recovery_teacher import KINDS
from .layout_recovery_worker import main as worker_main
from .layout_recovery_protocol import unpack_rollout
from .layout_recovery_world import INTERFACE


class StratifiedWorkerTests(unittest.TestCase):
    def test_each_packet_and_independent_reset_keeps_all_families(self):
        rng = np.random.default_rng(9340101)
        entries = [worker_world(rng, 100+i, i, balanced=True, stratified=True) for i in range(4)]
        for episode in range(104, 204):
            slot = int(rng.integers(4))
            entries[slot] = worker_world(rng, episode, slot, balanced=True, stratified=True)
            metadata = {'worlds': [meta for _, meta in entries]}
            validate_family_coverage(metadata)
            self.assertEqual([m['kind'] for _, m in entries], list(KINDS))

    def test_scenarios_and_recovery_stages_remain_independent_of_family(self):
        rng = np.random.default_rng(9340101)
        groups = set()
        for episode in range(128):
            for slot, kind in enumerate(KINDS):
                world, meta = worker_world(rng, episode, slot, balanced=True, stratified=True)
                groups.add((kind, meta['scenario'], meta['blockedStage'], meta['randomStarts']))
                self.assertEqual(meta['kind'], kind)
                if meta['replayedFailure']:
                    self.assertIn((meta['seed'], kind), RECOVERY_FAILURES)
                else:
                    self.assertTrue(9200000 <= meta['seed'] < 9290000)
                if meta['scenario'] == 'normal':
                    self.assertEqual([a.cargo for a in world.agents], [0]*4)
                if meta['scenario'] == 'blocked':
                    self.assertEqual([a.cargo for a in world.agents], [meta['blockedStage']]*4)
        for kind in KINDS:
            for scenario, stage in [('normal', None), ('loaded', None), ('blocked', 1), ('blocked', 2)]:
                for random_start in (False, True):
                    self.assertIn((kind, scenario, stage, random_start), groups)

    def test_default_sampler_unchanged_and_bad_configuration_rejected(self):
        a, ma = balanced_world(np.random.default_rng(19), 100)
        b, mb = worker_world(np.random.default_rng(19), 100, 0, balanced=True)
        self.assertEqual(ma, mb)
        np.testing.assert_array_equal(a.sensory(), b.sensory())
        with self.assertRaises(ValueError):
            worker_world(np.random.default_rng(19), 100, 0, stratified=True)
        with self.assertRaises(ValueError):
            worker_world(np.random.default_rng(19), 100, 4, balanced=True, stratified=True)
        with self.assertRaises(ValueError):
            balanced_world(np.random.default_rng(19), 100, fixed_kind='unknown')

    def test_missing_duplicate_or_unversioned_families_rejected(self):
        rng = np.random.default_rng(9340101)
        good = {'worlds': [worker_world(rng, 100+i, i, balanced=True, stratified=True)[1] for i in range(4)]}
        validate_family_coverage(good)
        validate_family_coverage(good, actors=16)
        with self.assertRaises(ValueError):
            validate_family_coverage(good, actors=4)
        for mutation in ('missing', 'duplicate', 'flag', 'version'):
            bad = copy.deepcopy(good)
            if mutation == 'missing':
                bad['worlds'].pop()
            elif mutation == 'duplicate':
                bad['worlds'][0]['kind'] = bad['worlds'][1]['kind']
            elif mutation == 'flag':
                bad['worlds'][0]['familyStratified'] = False
            else:
                bad['worlds'][0]['curriculum'] = 'old'
            with self.assertRaises(ValueError):
                validate_family_coverage(bad)

    def test_actual_worker_packets_keep_coverage_after_independent_resets(self):
        class FrozenBrain(torch.nn.Module):
            n = 166700
            fixed_hash = 'unit-fixed'
            interface = INTERFACE

            def checkpoint_hash(self):
                return 'unit-parameter'

            def weights(self):
                return torch.tensor(0.)

            def forward(self, observations, steps, state, weights):
                return torch.zeros((len(observations), 3)), state

        class LocalClient:
            def __init__(self):
                self.metadata = []

            def manifest(self):
                return {'finished': False, 'queued': 0, 'version': 0, 'runId': 'unit-run',
                        'fixedHash': 'unit-fixed', 'checkpoint': {'parameterHash': 'unit-parameter',
                            'sha256': hashlib.sha256(b'unit-checkpoint').hexdigest()}}

            def request(self, path):
                return b'unit-checkpoint'

            def upload(self, blob):
                self.metadata.append(unpack_rollout(blob)[3])
                return 'accepted'

        client = LocalClient()
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            args = SimpleNamespace(out=Path(root)/'run', root=Path(root), seed=9340101,
                                   worlds=4, balanced_curriculum=True, stratified_families=True,
                                   coordinator='unit', token_file=Path(root)/'token', cert=Path(root)/'cert',
                                   windows=2, max_seconds=60, contact_timeout=1,
                                   burn=16, gradient_frames=16, reset_seconds=.05)
            prefix = 'engine.layout_recovery_worker.'
            original_zeros, original_as_tensor = torch.zeros, torch.as_tensor
            stack.enter_context(patch(prefix+'Client', return_value=client))
            stack.enter_context(patch(prefix+'load_model', return_value=FrozenBrain()))
            stack.enter_context(patch(prefix+'torch.zeros', side_effect=lambda *a, **k:
                                      original_zeros(*a, **{**k, 'device': 'cpu'})))
            stack.enter_context(patch(prefix+'torch.as_tensor', side_effect=lambda *a, **k:
                                      original_as_tensor(*a, **{**k, 'device': 'cpu'})))
            optimizer = stack.enter_context(patch('torch.optim.Adam'))
            stack.enter_context(patch('builtins.print'))
            worker_main(args)
            self.assertEqual(len(client.metadata), 2)
            for metadata in client.metadata:
                validate_family_coverage(metadata)
                self.assertEqual(metadata['control'], 'learner-only')
                self.assertFalse(metadata['teacherActions'])
                self.assertFalse(metadata['productsAreVerification'])
                self.assertEqual([w['kind'] for w in metadata['worlds']], list(KINDS))
            self.assertNotEqual(client.metadata[0]['worlds'], client.metadata[1]['worlds'])
            self.assertEqual(json.loads((args.out/'finished.json').read_text())['accepted'], 2)
            optimizer.assert_not_called()
            for changes in ({'worlds': 3}, {'balanced_curriculum': False}):
                bad = SimpleNamespace(**{**vars(args), **changes, 'out': Path(root)/'bad'})
                with self.assertRaises(ValueError):
                    worker_main(bad)
                self.assertFalse(bad.out.exists())


if __name__ == '__main__':
    unittest.main()
