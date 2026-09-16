import unittest
import contextlib
import io
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from .layout_operation_focus import focus_values, focused_heads
from .layout_operation_prefix import CONTEXTS, KINDS
from .layout_operation_focus import focus_fit
from .layout_decision_grip_train import train
from .layout_operation_sampling import OperationCache
from .test_layout_operation_training import OperationTrainingTests
from .test_layout_microfit_flow import TinyTrainBrain


class FocusLossTests(unittest.TestCase):
    def fixture(self):
        specs = [dict(kind=k, category=c, start=20, stop=116, lossStart=84, focusTick=100)
                 for k in KINDS for c in CONTEXTS]
        labels = torch.zeros((32, 16, 3))
        labels[..., 2] = .2
        for i, s in enumerate(specs):
            labels[16, i, 2] = .2 if s['category'] == CONTEXTS[2] else 2.2
        predictions = torch.zeros_like(labels)
        predictions[..., 2] = 1.
        return specs, labels, predictions.requires_grad_(True)

    def test_grip_loss_only_at_physical_focus_and_balanced_classes(self):
        specs, y, p = self.fixture()
        a = p[..., :2].detach().clone()
        focused_heads(p, y, a, specs)[2].backward()
        self.assertEqual(int(torch.count_nonzero(p.grad[:, :, 2])), 16)
        self.assertEqual(int(torch.count_nonzero(p.grad[:16])), 0)
        self.assertEqual(int(torch.count_nonzero(p.grad[17:])), 0)
        positive = y[16, :, 2] > 1
        self.assertTrue((p.grad[16, positive, 2] < 0).all())
        self.assertTrue((p.grad[16, ~positive, 2] > 0).all())
        torch.testing.assert_close(p.grad[16, positive, 2].abs().sum(), p.grad[16, ~positive, 2].abs().sum())

    def test_nearby_opposite_labels_do_not_reverse_focus_gradient(self):
        specs, y, p = self.fixture()
        first = focused_heads(p, y, p[..., :2], specs)
        y2 = y.clone(); y2[:16, :, 2] = 2.2; y2[17:, :, 2] = 2.2
        torch.testing.assert_close(first, focused_heads(p, y2, p[..., :2], specs))

    def test_reject_misaligned_contexts_labels_and_motion(self):
        specs, y, p = self.fixture()
        bad = [dict(s) for s in specs]; bad[0]['focusTick'] += 1
        with self.assertRaises(ValueError): focus_values(p, y, bad)
        altered = y.clone(); altered[16, 0, 2] = .2
        with self.assertRaises(ValueError): focus_values(p, altered, specs)
        with self.assertRaises(ValueError): focused_heads(p, y, p[..., :1], specs)

    def test_real_recurrent_steps_and_read_only_focus_fit(self):
        b = TinyTrainBrain().eval(); before = b.checkpoint_hash()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            episodes, data, dataset, manifest = OperationTrainingTests().make_cache(folder, b)
            with patch('engine.layout_operation_sampling.load_histories', return_value=(episodes, data)):
                bank = OperationCache(folder, dataset, before, b.fixed_hash, 4)
            fit = focus_fit(b, bank, True)
            self.assertEqual(before, b.checkpoint_hash())
            self.assertTrue(fit['prefixIsExactForCandidate']); self.assertFalse(fit['isServiceEvidence'])
            self.assertEqual(len(fit['contexts']), 16)
            original_labels = bank.y.copy(); original_x = bank.x.copy()
            opt = torch.optim.Adam([{'params': [b.log_gains], 'lr': .0003}, {'params': [b.tonic], 'lr': .000001}])
            saved = []
            with contextlib.redirect_stdout(io.StringIO()):
                history = train(b, bank, opt, np.random.default_rng(25), 4, lambda u, r: saved.append(u))
            self.assertEqual(saved, [64]); self.assertEqual([r['update'] for r in history], [61,62,63,64])
            self.assertNotEqual(before, b.checkpoint_hash())
            self.assertTrue(all(np.isfinite(r['lossBeforeUpdate']) for r in history))
            np.testing.assert_array_equal(bank.y, original_labels)
            np.testing.assert_array_equal(bank.x, original_x)
            self.assertFalse(history[-1]['trainingFit']['prefixIsExactForCandidate'])


if __name__ == '__main__': unittest.main()
