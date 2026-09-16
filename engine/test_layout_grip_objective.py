import unittest
import torch
from .layout_grip_objective import threshold_grip_loss, matched_grip_ranking
from .layout_recovery_train import motor_head_errors, calibration_loss
from .test_layout_grip_calibration import TinyBrain


class GripObjectiveTests(unittest.TestCase):
    def test_correct_actions_need_not_match_arbitrary_amplitudes(self):
        p = torch.tensor([1.2, .8, 2.3, .1], requires_grad=True)
        y = torch.tensor([2.2, .2, 2.2, .2])
        loss = threshold_grip_loss(p, y)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        torch.testing.assert_close(p.grad, torch.zeros_like(p))

    def test_wrong_actions_have_correct_gradients(self):
        p = torch.tensor([.5, 1.5], requires_grad=True)
        loss = threshold_grip_loss(p, torch.tensor([2.2, .2]))
        loss.backward()
        self.assertLess(float(p.grad[0]), 0)
        self.assertGreater(float(p.grad[1]), 0)

    def test_class_balance_and_missing_class(self):
        p, y = torch.tensor([.5, 1.5]), torch.tensor([2.2, .2])
        torch.testing.assert_close(threshold_grip_loss(p, y),
            threshold_grip_loss(torch.cat((p[:1], p[1:].repeat(100))),
                                torch.cat((y[:1], y[1:].repeat(100)))))
        self.assertTrue(torch.isfinite(threshold_grip_loss(p[:1], y[:1])))

    def test_ranking_cannot_be_satisfied_by_common_bias(self):
        p = torch.ones((2, 3, 4), requires_grad=True)
        y = torch.tensor([2.2, .2, 2.2, .2]).expand_as(p)
        loss = matched_grip_ranking(p, y)
        self.assertGreater(float(loss.detach()), 0)
        torch.testing.assert_close(loss, matched_grip_ranking(p+3, y))
        loss.backward()
        self.assertLess(float(p.grad[0, 0, 0]), 0)
        self.assertGreater(float(p.grad[0, 0, 1]), 0)
        self.assertAlmostEqual(float(p.grad.sum()), 0)

    def test_ranking_only_compares_same_view_and_accepts_no_pairs(self):
        p = torch.tensor([[1.2, .8, 1.2, .8], [2., 2., 2., 2.]])
        y = torch.tensor([[2.2, .2, 2.2, .2], [2.2, 2.2, 2.2, 2.2]])
        self.assertEqual(float(matched_grip_ranking(p, y)), 0)
        self.assertEqual(float(matched_grip_ranking(p[1:], y[1:])), 0)

    def test_invalid_margins_and_broken_groups_rejected(self):
        for margin in (0., -.1, .6, float('nan')):
            with self.assertRaises(ValueError):
                threshold_grip_loss(torch.ones(4), torch.ones(4), margin)
        with self.assertRaises(ValueError):
            matched_grip_ranking(torch.ones(3), torch.ones(3))

    def test_opt_in_keeps_motion_mse_and_recurrent_gradients(self):
        p = torch.tensor([[.5, .2, 1.3], [.8, -.2, .7]], requires_grad=True)
        y = torch.tensor([[.4, .3, 2.2], [.7, -.1, .2]])
        heads = motor_head_errors(p, y, grip_margin=.1)
        torch.testing.assert_close(heads[:2], (p-y).square().mean(0)[:2])
        self.assertEqual(float(heads[2].detach()), 0)
        model = TinyBrain()
        x = torch.ones((6, 8, 4))
        target = torch.zeros((2, 8, 3))
        target[..., 2] = torch.tensor([.2, 2.2, .2, 2.2]*2)
        loss = calibration_loss(model, x, target, 4, model.weights(), .1).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(model.tonic.grad).all())


if __name__ == '__main__':
    unittest.main()
