import unittest
import numpy as np
import torch

from .evaluate_supervised import FrozenPolicy
from .fast_frozen_policy import FastFrozenPolicy, SplitFrozenPolicy
from .supervised_steering import TrainableConnectome


def tiny_model(device='cpu'):
    model = TrainableConnectome.__new__(TrainableConnectome)
    torch.nn.Module.__init__(model)
    model.n, model.input_channels = 4, 3
    values = {'crow': [0, 1, 2, 3, 4], 'col': [1, 2, 3, 0], 'post': [0, 1, 2, 3],
              'tcrow': [0, 1, 2, 3, 4], 'tcol': [3, 0, 1, 2], 'torder': [3, 0, 1, 2],
              'sensory_indices': [0, 1, 2], 'sensory_channels': [0, 1, 2],
              'motor_forward': [3], 'motor_left': [2], 'motor_right': [1], 'motor_interact': [0]}
    for name, value in values.items():
        model.register_buffer(name, torch.tensor(value, device=device))
    model.register_buffer('base', torch.tensor([.07, -.1, .05, .08], device=device))
    model.register_buffer('tonic', torch.zeros((4, 1), device=device))
    model.register_buffer('fast_mask', torch.ones((4, 1), device=device))
    model.log_gains = torch.nn.Parameter(torch.zeros(4, device=device))
    return model.eval().requires_grad_(False)


class FastPolicyTests(unittest.TestCase):
    def compare(self, policy, model):
        reference = FrozenPolicy(model, 4)
        rng = np.random.default_rng(917)
        for _ in range(40):
            obs = rng.random((4, 3), dtype=np.float32)
            expected, actual = reference.act(obs), policy.act(obs)
            torch.testing.assert_close(policy.state, reference.state, atol=1e-7, rtol=1e-5)
            np.testing.assert_allclose([[a['speed'], a['turn']] for a in actual],
                                       [[a['speed'], a['turn']] for a in expected], atol=1e-6, rtol=1e-5)
            self.assertEqual([a['interact'] for a in actual], [a['interact'] for a in expected])
        policy.reset()
        reference.reset()
        self.assertIsNone(policy.state)
        policy.act(obs)
        reference.act(obs)
        torch.testing.assert_close(policy.state, reference.state)
        with self.assertRaises(ValueError):
            policy.act(obs, explore=True)
        with self.assertRaises(ValueError):
            policy.act(obs[:2])

    def test_cpu_cached_original_equations_and_reset(self):
        model = tiny_model()
        self.compare(FastFrozenPolicy(model, 4, index32=True), model)

    def test_trainable_model_rejected(self):
        model = tiny_model().requires_grad_(True)
        with self.assertRaises(ValueError):
            FastFrozenPolicy(model, 4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_graph_and_parallel_states(self):
        model = tiny_model('cuda')
        self.compare(FastFrozenPolicy(model, 4, cuda_graph=True, index32=True), model)
        self.compare(SplitFrozenPolicy(model, 4), model)


if __name__ == '__main__':
    unittest.main()
