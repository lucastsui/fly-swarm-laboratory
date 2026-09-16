from types import SimpleNamespace
import unittest
import numpy as np
from .layout_service_balanced_sampling import ServiceBalancedDemonstrations, service_buckets, VERSION
from .layout_recovery_teacher import KINDS


class ServiceBalancedSamplingTests(unittest.TestCase):
    def test_every_batch_contains_all_four_contexts_and_original_actor_history(self):
        # Isolated sampling test; physical replay and version provenance have
        # independent integration tests in the cooldown dataset/sampling suite.
        categories = ('pickup-1', 'pickup-2', 'transfer-1', 'delivery-3',
                      'after-pickup-1', 'uniform-trajectory')
        groups, specs = {}, []
        for kind in KINDS:
            groups[kind] = {}
            for c in categories:
                groups[kind][c] = [len(specs)]
                specs.append({'kind': kind, 'category': c, 'episode': len(specs), 'start': 0, 'stop': 96})
        count = len(specs)
        sampler = object.__new__(ServiceBalancedDemonstrations)
        x = np.arange(count*96*4*297, dtype=np.float32).reshape(count,96,4,297)
        y = np.zeros((count,96,4,3), np.float32)
        states = np.arange(count*4*4, dtype=np.float32).reshape(count,4,4)
        sampler.bank = SimpleNamespace(groups=groups, manifest={'windows': specs}, x=x, y=y, states=states)
        sampler.revised = [v.copy() for v in y]
        sampler.focused = SimpleNamespace(choices=[[{'fly': 2, 'eventTick': 80}] for _ in specs])
        sampler.groups = groups
        sampler.choices = dict(enumerate(sampler.focused.choices))
        sampler.record = {'labelVersion': 'test-timing-version', 'labelRevisionManifestHash': 'test-hash'}
        sampler.buckets = service_buckets(groups)
        for seed in range(10):
            bx, by, bs, meta = sampler.sample(np.random.default_rng(seed), 'cpu')
            self.assertEqual(meta['sampling'], VERSION)
            self.assertEqual(len(meta['selections']), 16)
            for kind in KINDS:
                part = [s for s in meta['selections'] if s['kind']==kind]
                self.assertEqual({s['trainingContext'] for s in part},
                                 {'pickup','loaded-operation','after-pickup','exploration'})
            for col, s in enumerate(meta['selections']):
                i, fly = s['windowIndex'], s['fly']
                np.testing.assert_array_equal(bx[:,col], x[i,:,fly])
                np.testing.assert_array_equal(by[:,col], y[i,:,fly])
                np.testing.assert_array_equal(bs[:,col], states[i,:,fly])

    def test_reject_curriculum_without_loaded_operation(self):
        groups = {k:{'pickup-1':[0], 'after-pickup-1':[1], 'uniform-trajectory':[2]} for k in KINDS}
        with self.assertRaisesRegex(ValueError, 'every service'):
            service_buckets(groups)


if __name__ == '__main__':
    unittest.main()
