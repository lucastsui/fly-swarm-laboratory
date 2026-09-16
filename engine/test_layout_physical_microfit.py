import unittest
from unittest.mock import patch
import torch
from .test_layout_cargo_response_probe import CargoBrain
from .layout_physical_microfit import validate_bounds, fit_passes, run_fit


class PhysicalMicrofitTests(unittest.TestCase):
    def test_bounds_and_gate(self):
        validate_bounds(80, 20, .003, .00001)
        for values in ((81, 20, .003, .00001), (0, 1, .003, .00001), (1, 2, .003, .00001),
                       (4, 2, float('nan'), .00001), (4, 2, .004, .00001)):
            with self.assertRaises(ValueError):
                validate_bounds(*values)
        self.assertTrue(fit_passes({'gripRecall': .9, 'gripFalsePositiveRate': .1, 'parentMotionMSE': [.05, .01]}))
        self.assertFalse(fit_passes({'gripRecall': None, 'gripFalsePositiveRate': .1, 'parentMotionMSE': [0, 0]}))

    def test_real_update_and_exact_data_preservation(self):
        model = CargoBrain()
        before = model.checkpoint_hash()
        x = torch.ones((6, 4, 297))
        x[:, ::2, 126] = 0.
        labels = torch.full((6, 4, 3), .2)
        labels[:, 1::2, 2] = 2.2
        state = torch.full((4, 4), .001)
        originals = [v.clone() for v in (x, labels, state)]
        records = []
        with patch('builtins.print'):
            result = run_fit(model, x, labels, state, 3, 4, 2, .003, .00001,
                             lambda update, row, opt: records.append((update, dict(row), len(opt.state))))
        self.assertNotEqual(before, model.checkpoint_hash())
        self.assertEqual([r[0] for r in records], [0, 2, 4])
        self.assertEqual(records[0][2], 0)
        self.assertEqual(records[-1][2], 2)
        self.assertFalse(result['isServiceEvidence'])
        self.assertTrue(result['optimizerFreshByDesign'])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(len(result['history']), 4)
        for original, value in zip(originals, (x, labels, state)):
            torch.testing.assert_close(original, value, rtol=0, atol=0)
        self.assertNotIn('teacherMotionMSE', records[-1][1])


if __name__ == '__main__':
    unittest.main()
