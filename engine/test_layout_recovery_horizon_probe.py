import unittest
import torch
from .test_layout_grip_calibration import TinyBrain
from .layout_recovery_horizon_probe import horizon_predictions
from .layout_recovery_train import recurrent_predictions


class HorizonProbeTests(unittest.TestCase):
    def test_same_forward_truncated_matches_existing_and_full_matches_finite_difference(self):
        model = TinyBrain().double()
        x = torch.ones((8, 4, 4), dtype=torch.float64)
        truncated = horizon_predictions(model, x, 4, 4)
        existing = recurrent_predictions(model, x, torch.zeros((4, 4), dtype=torch.float64), 4, model.weights())
        torch.testing.assert_close(truncated, existing, rtol=0, atol=0)
        full = horizon_predictions(model, x, 4, 0)
        torch.testing.assert_close(truncated, full, rtol=0, atol=0)
        production_full = recurrent_predictions(model, x, torch.zeros((4, 4), dtype=torch.float64),
                                                4, model.weights(), gradient_start=0)
        torch.testing.assert_close(full, production_full, rtol=0, atol=0)
        gt = torch.autograd.grad(truncated.sum(), model.tonic)[0]
        gf = torch.autograd.grad(full.sum(), model.tonic)[0]
        torch.testing.assert_close(gf, torch.autograd.grad(production_full.sum(), model.tonic)[0], rtol=0, atol=0)
        self.assertGreater(float((gf-gt).abs().max()), 1.)
        original = model.tonic.detach().clone()
        values = []
        for sign in (-1, 1):
            with torch.no_grad():
                model.tonic.copy_(original+sign*1e-6)
            values.append(float(horizon_predictions(model, x, 4, 0).sum().detach()))
        with torch.no_grad():
            model.tonic.copy_(original)
        self.assertAlmostEqual((values[1]-values[0])/2e-6, float(gf.sum()), places=5)

    def test_invalid_horizons_rejected(self):
        for output, gradient in ((4, -1), (4, 5), (8, 0)):
            with self.assertRaises(ValueError):
                horizon_predictions(TinyBrain(), torch.ones((8, 4, 4)), output, gradient)
            model = TinyBrain()
            with self.assertRaises(ValueError):
                recurrent_predictions(model, torch.ones((8, 4, 4)), torch.zeros((4, 4)),
                                      output, model.weights(), gradient_start=gradient)


if __name__ == '__main__':
    unittest.main()
