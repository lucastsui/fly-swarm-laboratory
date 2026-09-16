import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .layout_grip_calibration import CalibrationBank, matched_examples
from .layout_microfit import select_microfit_scenes, microfit_passes, discard_pending_warmup_replay


class MicrofitTests(unittest.TestCase):
    def bank(self):
        bank = object.__new__(CalibrationBank)
        bank.observations, bank.targets, bank.metadata = matched_examples(scenes=16, burn=4, frames=2)
        bank.burn, bank.parent_hash = 4, 'unchanged-parent'
        return bank

    def test_one_mixed_cargo_scene_per_box_without_input_changes(self):
        bank = self.bank()
        selected = select_microfit_scenes(bank.metadata, bank.targets)
        self.assertEqual(len(selected), 4)
        self.assertEqual([bank.metadata[i]['box'] for i in selected], [0, 1, 2, 3])
        subset = bank.subset(selected)
        x, y, ids = subset.all_examples('cpu')
        self.assertEqual(x.shape, (6, 16, 297))
        self.assertEqual(y.shape, (2, 16, 3))
        self.assertEqual(ids, selected)
        self.assertEqual(subset.parent_hash, bank.parent_hash)
        indices = (np.asarray(selected)[:, None]*4+np.arange(4)).ravel()
        np.testing.assert_array_equal(x.numpy(), bank.observations[:, indices])
        np.testing.assert_array_equal(y.numpy(), bank.targets[:, indices])
        for i in range(4):
            positive = y[:, 4*i:4*i+4, 2].numpy() > 1
            self.assertTrue(positive.any())
            self.assertTrue((~positive).any())

    def test_reject_missing_roles_or_invalid_subset(self):
        bank = self.bank()
        with self.assertRaises(ValueError):
            select_microfit_scenes(bank.metadata, np.zeros_like(bank.targets))
        for indices in ([], [-1], [16], [1, 1]):
            with self.assertRaises(ValueError):
                bank.subset(indices)

    def test_fit_gate_rejects_never_grip_always_grip_and_motion_regression(self):
        fit = {'gripRecall': .9, 'gripFalsePositiveRate': .1, 'parentMotionMSE': [.01, .05]}
        self.assertTrue(microfit_passes(fit))
        for changes in ({'gripRecall': 0.}, {'gripFalsePositiveRate': 1.},
                        {'parentMotionMSE': [.01, .0501]}, {'gripRecall': None},
                        {'parentMotionMSE': [float('nan'), 0.]}):
            self.assertFalse(microfit_passes({**fit, **changes}))

    def test_pending_retirement_preserves_every_raw_file(self):
        with tempfile.TemporaryDirectory() as root:
            files = [Path(root)/f'packet-{i}.npz' for i in range(2)]
            for path in files:
                path.write_bytes(b'preserved evidence')
            exchange = SimpleNamespace(queue=deque(files), lock=threading.Lock())
            report = discard_pending_warmup_replay(exchange)
            self.assertEqual(len(exchange.queue), 0)
            self.assertTrue(report['rawExperiencePreserved'])
            self.assertEqual(report['discardedPendingIds'], [p.name for p in files])
            for path in files:
                self.assertEqual(path.read_bytes(), b'preserved evidence')


if __name__ == '__main__':
    unittest.main()
