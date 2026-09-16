import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from .layout_demonstration_step_probe import run_probe, select_windows, moment_hash, frozen_predictions
from .layout_recovery_teacher import KINDS
from .test_layout_microfit_flow import TinyTrainBrain


class PhysicalStepProbeTests(unittest.TestCase):
    def model_and_optimizer(self):
        model = TinyTrainBrain().eval()
        model.surrogate_training = False
        optimizer = torch.optim.Adam([{'params': [model.log_gains], 'lr': .0003, 'eps': 1e-14},
                                      {'params': [model.tonic], 'lr': .000001, 'eps': 1e-10}])
        for p in model.parameters():
            p.grad = torch.full_like(p, .1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return model, optimizer

    def test_every_mode_restores_parameters_and_saved_moments(self):
        model, optimizer = self.model_and_optimizer()
        before, moments = model.checkpoint_hash(), moment_hash(optimizer)
        x = torch.ones((6, 4, 297))
        y = torch.full((6, 4, 3), .2)
        y[3:, ::2, 2] = 2.2
        state = torch.full((4, 4), .02)
        with patch('builtins.print'):
            result = run_probe(model, x, y, state, optimizer, 3, scales=(.1, 1.))
        self.assertEqual(len(result['trials']), 12)
        self.assertEqual(len(result['gradients']), 3)
        self.assertTrue(result['parametersRestoredExactly'])
        self.assertEqual(model.checkpoint_hash(), before)
        self.assertEqual(moment_hash(optimizer), moments)
        self.assertFalse(model.training)
        self.assertFalse(model.surrogate_training)
        self.assertTrue(all(not r['isServiceEvidence'] for r in result['trials']))

    def test_trial_exception_restores_model(self):
        model, optimizer = self.model_and_optimizer()
        before, moments = model.checkpoint_hash(), moment_hash(optimizer)
        count = 0
        def fail_after_baseline(*args):
            nonlocal count
            count += 1
            if count > 1:
                raise RuntimeError('deliberate trial failure')
            return frozen_predictions(*args)
        with patch('builtins.print'), patch('engine.layout_demonstration_step_probe.frozen_predictions',
                                           side_effect=fail_after_baseline):
            with self.assertRaises(RuntimeError):
                run_probe(model, torch.ones((6, 4, 297)), torch.ones((6, 4, 3)),
                          torch.zeros((4, 4)), optimizer, 3)
        self.assertEqual(model.checkpoint_hash(), before)
        self.assertEqual(moment_hash(optimizer), moments)

    def test_deterministic_one_physical_window_per_family(self):
        labels = np.full((8, 6, 4, 3), .2, np.float32)
        labels[4:, 4:, 0, 2] = 2.2
        bank = SimpleNamespace(groups={k: {'pickup-1': [i+4, i]} for i, k in enumerate(KINDS)},
                               manifest={'burn': 3}, y=labels)
        self.assertEqual(select_windows(bank), [4, 5, 6, 7])
        with self.assertRaises(ValueError):
            select_windows(bank, 'missing-category')


if __name__ == '__main__':
    unittest.main()
