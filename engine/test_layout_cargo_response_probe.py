import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from .layout_cargo_response_probe import perturb_cargo, run_probe, select_batch
from .layout_recovery_teacher import KINDS
from .test_layout_microfit_flow import TinyTrainBrain


class CargoBrain(TinyTrainBrain):
    def forward(self, x, steps, state=None, weights=None):
        if state is None:
            state = torch.zeros((4, len(x)))
        drive = x[:, [0, 1, 2, 126]].T
        return None, .75*state+(weights+self.tonic)*drive


class CargoResponseTests(unittest.TestCase):
    def test_selection_preserves_exact_event_actor_history_and_prefix(self):
        x = np.arange(12*6*4*297, dtype=np.float32).reshape(12, 6, 4, 297)
        y = np.full((12, 6, 4, 3), .2, np.float32)
        states = np.arange(12*5*4, dtype=np.float32).reshape(12, 5, 4)
        groups, windows, choices = {}, [], []
        for family, kind in enumerate(KINDS):
            groups[kind] = {}
            for stage in (1, 2, 3):
                index = family*3+stage-1
                groups[kind]['pickup-'+str(stage)] = [index]
                y[index, 4, stage, 2] = 2.2
                choices.append([{'fly': stage, 'eventTick': 4}])
                windows.append({'kind': kind, 'category': 'pickup-'+str(stage)})
        bank = SimpleNamespace(x=x, y=y, states=states, groups=groups,
                               manifest={'burn': 3, 'windows': windows})
        xx, yy, ss, selections = select_batch(SimpleNamespace(bank=bank, choices=choices), 'cpu')
        for index in range(12):
            fly = index % 3+1
            torch.testing.assert_close(xx[:, index], torch.from_numpy(x[index, :, fly]), rtol=0, atol=0)
            torch.testing.assert_close(yy[:, index], torch.from_numpy(y[index, :, fly]), rtol=0, atol=0)
            torch.testing.assert_close(ss[:, index], torch.from_numpy(states[index, :, fly]), rtol=0, atol=0)
            self.assertEqual(selections[index]['fly'], fly)
        y[0, :, 1, 2] = .2
        with self.assertRaises(ValueError):
            select_batch(SimpleNamespace(bank=bank, choices=choices), 'cpu')

    def test_interventions_only_change_cargo_and_preserve_original(self):
        x = torch.rand((6, 4, 297))
        before = x.clone()
        for mode in ('zero', 'cycle', 'scale-0.25', 'scale-4', 'scale-16'):
            changed = perturb_cargo(x, mode)
            torch.testing.assert_close(changed[..., :126], x[..., :126], rtol=0, atol=0)
            torch.testing.assert_close(changed[..., 129:], x[..., 129:], rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            perturb_cargo(x, 'unbounded')

    def test_frozen_probe_has_input_gradient_and_no_parameter_or_prefix_change(self):
        model = CargoBrain().eval().requires_grad_(False)
        before = model.checkpoint_hash()
        x = torch.ones((6, 4, 297))
        x[:, ::2, 126] = 0.
        y = torch.full((6, 4, 3), .2)
        y[:, 1::2, 2] = 2.2
        state = torch.full((4, 4), .001)
        with patch('builtins.print'):
            result = run_probe(model, x, y, state, 3)
        self.assertEqual(model.checkpoint_hash(), before)
        self.assertFalse(result['optimizerUsed'])
        self.assertFalse(result['isServiceEvidence'])
        self.assertEqual(len(result['inputInterventions']), 5)
        self.assertGreater(result['exactInputGradientOfGripGroupContrast']['cargo']['absoluteSum'], 0)
        self.assertEqual(result['exactInputGradientOfGripGroupContrast']['color']['absoluteSum'], 0)
        self.assertGreater(result['inputInterventions'][0]['meanAbsoluteMotorChange'][2], 0)
        model.train()
        with self.assertRaises(ValueError):
            run_probe(model, x, y, state, 3)


if __name__ == '__main__':
    unittest.main()
