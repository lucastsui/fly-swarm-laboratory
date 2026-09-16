import unittest
import torch
from .layout_adam_step_probe import adam_direction, training_derivative


class AdamProbeTests(unittest.TestCase):
    def test_direction_matches_real_adam_and_leaves_moments_unchanged(self):
        p = torch.nn.Parameter(torch.tensor([.2, -.3, .1], dtype=torch.float64))
        optimizer = torch.optim.Adam([p], lr=.003, eps=1e-14)
        for gradient in ([.8, -.6, .2], [-1., .3, .4]):
            p.grad = torch.tensor(gradient, dtype=p.dtype)
            torch.nn.utils.clip_grad_norm_([p], 1.)
            optimizer.step()
        before = p.detach().clone()
        state = optimizer.state[p]
        moments = {k: v.clone() for k, v in state.items()}
        next_gradient = torch.tensor([2., .8, -.4], dtype=p.dtype)
        direction = adam_direction(next_gradient, state, optimizer.param_groups[0])
        for key in state:
            torch.testing.assert_close(state[key], moments[key], rtol=0, atol=0)
        p.grad = next_gradient.clone()
        torch.nn.utils.clip_grad_norm_([p], 1.)
        optimizer.step()
        torch.testing.assert_close(p.detach()-before, direction, rtol=1e-11, atol=1e-13)

    def test_derivative_mode_restores_on_exception(self):
        model = torch.nn.Linear(1, 1)
        model.eval()
        model.surrogate_training = False
        with self.assertRaises(RuntimeError):
            with training_derivative(model, True):
                self.assertTrue(model.training)
                self.assertTrue(model.surrogate_training)
                raise RuntimeError('deliberate')
        self.assertFalse(model.training)
        self.assertFalse(model.surrogate_training)


if __name__ == '__main__':
    unittest.main()
