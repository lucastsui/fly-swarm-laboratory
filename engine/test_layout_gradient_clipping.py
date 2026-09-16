import unittest
import torch
from .layout_recovery_train import clip_brain_gradients


class ClipTests(unittest.TestCase):
    def model(self):
        model = torch.nn.Module()
        model.log_gains = torch.nn.Parameter(torch.zeros(2))
        model.tonic = torch.nn.Parameter(torch.zeros(2))
        model.log_gains.grad = torch.tensor([.1, .2])
        model.tonic.grad = torch.tensor([30., 40.])
        return model

    def test_global_default_reproduces_shared_clipping(self):
        model = self.model()
        norms = clip_brain_gradients(model)
        self.assertGreater(norms['global'], 50.)
        self.assertLess(float(model.log_gains.grad.norm()), .005)
        self.assertLessEqual(float(torch.cat([p.grad for p in model.parameters()]).norm()), 1.)

    def test_block_clip_preserves_weak_synaptic_signal_without_parameter_updates(self):
        model = self.model()
        norms = clip_brain_gradients(model, True)
        self.assertAlmostEqual(norms['tonic'], 50.)
        torch.testing.assert_close(model.log_gains.grad, torch.tensor([.1, .2]))
        torch.testing.assert_close(model.tonic.grad, torch.tensor([.6, .8]))
        for parameter in model.parameters():
            torch.testing.assert_close(parameter, torch.zeros(2))

    def test_each_large_block_remains_individually_bounded(self):
        model = self.model()
        model.log_gains.grad *= 100
        clip_brain_gradients(model, True)
        for parameter in model.parameters():
            self.assertLessEqual(float(parameter.grad.norm()), 1.)
            self.assertTrue((parameter.grad > 0).all())


if __name__ == '__main__':
    unittest.main()
