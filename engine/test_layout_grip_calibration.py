import unittest
from types import SimpleNamespace
import numpy as np
import torch
from .layout_grip_calibration import matched_examples, cache_parent_motion, cargo_contrast, calibration_progress
from .layout_recovery_train import calibration_loss


class TinyBrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 4
        self.tonic = torch.nn.Parameter(torch.ones((4, 1))*.001)
        for name, index in [('forward', 0), ('left', 1), ('right', 2), ('interact', 3)]:
            setattr(self, 'motor_'+name, torch.tensor([index]))

    def weights(self):
        return self.tonic

    def forward(self, x, steps, state=None, weights=None):
        if state is None:
            state = torch.zeros((4, len(x)))
        return None, .75*state + self.tonic*x[:, :4].T


class GripCalibrationTests(unittest.TestCase):
    def test_ranking_weight_changes_only_training_grip_head(self):
        model = TinyBrain()
        x = torch.ones((6, 8, 4))
        y = torch.zeros((2, 8, 3))
        y[..., 2] = torch.tensor([.2, 2.2, .2, 2.2]*2)
        ordinary = calibration_loss(model, x, y, 4, model.weights(), .1)
        weighted = calibration_loss(model, x, y, 4, model.weights(), .1, 100.)
        torch.testing.assert_close(ordinary[:2], weighted[:2])
        self.assertGreater(float(weighted[2].detach()), float(ordinary[2].detach()))
        for invalid in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                calibration_loss(model, x, y, 4, model.weights(), .1, invalid)

    def test_fit_logger_has_no_gradients_or_parameter_updates(self):
        model = TinyBrain()
        model.checkpoint_hash = lambda: str(model.tonic.detach().tolist())
        bank = SimpleNamespace(observations=np.ones((6, 8, 4), np.float32),
                               targets=np.zeros((2, 8, 3), np.float32), burn=4)
        bank.targets[:, ::2, 2] = 2.2
        result = calibration_progress(model, bank, actors=8)
        self.assertTrue(result['trainingSubsetOnly'])
        self.assertFalse(result['isServiceEvidence'])
        self.assertEqual(result['frames'], 2)
        self.assertEqual(result['actors'], 8)
        self.assertIsNone(model.tonic.grad)
        self.assertTrue(model.training)

    def test_reproducible_disjoint_training_geometry_and_shapes(self):
        x, y, meta = matched_examples(scenes=16, burn=4, frames=2)
        xx, yy, mm = matched_examples(scenes=16, burn=4, frames=2)
        self.assertEqual(x.shape, (6, 64, 297))
        self.assertEqual(y.shape, (2, 64, 3))
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(y, yy)
        self.assertEqual(meta, mm)
        self.assertTrue(all(9320000 <= m['seed'] < 9329000 for m in meta))
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(int((y[..., 2] > 1).sum()), 0)
        self.assertGreater(int((y[..., 2] < 1).sum()), 0)
        for box in range(4):
            self.assertEqual({m['near'] for m in meta if m['box'] == box}, {False, True})
            self.assertEqual({m['cargoTransition'] for m in meta if m['box'] == box}, {False, True})

    def test_counterfactuals_keep_all_boxes_visible_and_change_only_cargo_feedback(self):
        x, _, meta = matched_examples(scenes=16, burn=4, frames=2)
        for i, m in enumerate(meta):
            group = x[-1, i*4:i*4+4]
            # Geometry/color and capacity maps do not select a task target.
            for channels in (slice(0, 24), slice(30, 126), slice(129, 297)):
                np.testing.assert_array_equal(group[:, channels], np.repeat(group[:1, channels], 4, axis=0))
            np.testing.assert_array_equal(group[:, 126:129], np.vstack((np.zeros(3), np.eye(3))))
            if m['cargoTransition']:
                np.testing.assert_array_equal(x[0, i*4:i*4+4], np.repeat(x[0, i*4:i*4+1], 4, axis=0))

    def test_motion_cache_is_detached_and_never_changes_parent_or_grip_targets(self):
        model = TinyBrain()
        before = model.tonic.detach().clone()
        obs = np.ones((6, 8, 4), np.float32)
        targets = np.zeros((2, 8, 3), np.float32)
        targets[..., 2] = 2.2
        anchored = cache_parent_motion(model, obs, targets, 4, batch=4)
        self.assertIsNone(model.tonic.grad)
        torch.testing.assert_close(before, model.tonic)
        np.testing.assert_array_equal(anchored[..., 2], targets[..., 2])
        self.assertGreater(float(anchored[..., 0].mean()), 0)
        self.assertEqual(float(targets[..., 0].sum()), 0)

    def test_contrast_rejects_constant_grip_and_preserves_scene_grouping(self):
        y = torch.zeros((2, 8, 3))
        y[..., 2] = torch.tensor([.2, 2.2, .2, 2.2]*2)
        p = torch.ones_like(y, requires_grad=True)
        loss = cargo_contrast(p, y)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertGreater(float(p.grad[0, 0, 2]), 0)
        self.assertLess(float(p.grad[0, 1, 2]), 0)
        self.assertEqual(float(cargo_contrast(y, y)), 0)

    def test_recurrent_calibration_backpropagates_inside_existing_brain(self):
        model = TinyBrain()
        x = torch.ones((6, 8, 4))
        y = torch.zeros((2, 8, 3))
        y[..., 2] = torch.tensor([.2, 2.2, .2, 2.2]*2)
        heads = calibration_loss(model, x, y, 4, model.weights())
        self.assertEqual(heads.shape, (3,))
        self.assertTrue(torch.isfinite(heads).all())
        heads.sum().backward()
        self.assertTrue(torch.isfinite(model.tonic.grad).all())


if __name__ == '__main__':
    unittest.main()
