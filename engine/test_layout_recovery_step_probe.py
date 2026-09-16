import unittest
import torch
from .layout_recovery_step_probe import first_step_direction, temporary_step


class StepProbeTests(unittest.TestCase):
    def test_direction_is_bounded_finite_and_preserves_zero(self):
        g = torch.tensor([0., 1e-12, -2., 4.])
        step = first_step_direction(g, .01, 1e-14)
        self.assertTrue(torch.isfinite(step).all())
        self.assertLessEqual(float(step.abs().max()), .0100001)
        self.assertEqual(float(step[0]), 0.)
        self.assertTrue(torch.all(step[1:]*g[1:] < 0))
        with self.assertRaises(ValueError):
            first_step_direction(torch.tensor([float('nan')]), .01, 1e-14)

    def test_all_parameters_restore_even_after_failure(self):
        model = torch.nn.Module()
        model.log_gains = torch.nn.Parameter(torch.tensor([1.99, -1.99]))
        model.tonic = torch.nn.Parameter(torch.tensor([.099, -.099]))
        originals = [p.detach().clone() for p in (model.log_gains, model.tonic)]
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with temporary_step(model, originals, [torch.tensor([1., -1.]), torch.tensor([1., -1.])]):
                torch.testing.assert_close(model.log_gains, torch.tensor([2., -2.]))
                torch.testing.assert_close(model.tonic, torch.tensor([.1, -.1]))
                raise RuntimeError('deliberate')
        torch.testing.assert_close(model.log_gains, originals[0], rtol=0, atol=0)
        torch.testing.assert_close(model.tonic, originals[1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
