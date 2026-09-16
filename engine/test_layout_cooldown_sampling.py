import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .layout_cooldown_sampling import CooldownFocusedDemonstrations
from .layout_cooldown_dataset import build_dataset
from .layout_demonstration_train import DemonstrationCache
from .test_layout_demonstration_curriculum import physical_curriculum_fixture
from .test_layout_microfit_flow import TinyTrainBrain


class CooldownSamplingTests(unittest.TestCase):
    def test_versioned_overlay_keeps_original_actor_states_inputs_and_motion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root/'candidate.npz'
            parent.write_bytes(b'test-parent')
            model = TinyTrainBrain()
            dataset, cache = physical_curriculum_fixture(root, model, parent)
            with contextlib.redirect_stdout(io.StringIO()):
                build_dataset(dataset, root/'labels')
            bank = DemonstrationCache(cache, model.checkpoint_hash(), model.fixed_hash, 2, 2, model.n)
            x0, y0, s0 = (v.copy() for v in (bank.x, bank.y, bank.states))
            sampler = CooldownFocusedDemonstrations(bank, dataset, root/'labels')
            # Exercise a changed grip value after provenance validation without
            # altering physical fixture files. Real replay changes are tested
            # separately in test_layout_cooldown_dataset.
            for target in sampler.revised:
                target[..., 2] = 2.2
            for seed in range(3):
                x, y, state, meta = sampler.sample(np.random.default_rng(seed), 'cpu')
                self.assertEqual(x.shape[1], 16)
                self.assertGreater(meta['changedGripLabels'], 0)
                for col, selection in enumerate(meta['selections']):
                    i, fly = selection['windowIndex'], selection['fly']
                    np.testing.assert_array_equal(x[:, col], x0[i, :, fly])
                    np.testing.assert_array_equal(state[:, col], s0[i, :, fly])
                    np.testing.assert_array_equal(y[:, col, :2], y0[i, :, fly, :2])
                self.assertEqual(meta['labelVersion'], sampler.record['labelVersion'])
            indexes = sampler.diagnostic_indexes()[:2]
            x, y, state = sampler.batch(indexes, 'cpu')
            torch.testing.assert_close(x, bank.batch(indexes, 'cpu')[0], atol=0, rtol=0)
            self.assertTrue(torch.all(y[..., 2] == 2.2))
            for original, now in zip((x0, y0, s0), (bank.x, bank.y, bank.states)):
                np.testing.assert_array_equal(original, now)
            self.assertFalse(sampler.record['networkPacketLabelSemanticsChanged'])


if __name__ == '__main__':
    unittest.main()
