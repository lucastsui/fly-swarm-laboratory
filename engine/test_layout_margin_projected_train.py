import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from .layout_margin_projected_train import attempt, margin_gate, project_coordinates
from .layout_contextual_operation_probe import measure, contextual_gate
from .layout_recovery_brain import save_model
from .layout_guarded_snapshot_train import verify_saved_snapshot
from .test_layout_contextual_operation_probe import measured_pair, fixture
from .layout_demonstration_step_probe import frozen_predictions


class MarginProjectedTests(unittest.TestCase):
    def bank(self):
        brain, bank = fixture()
        bank.manifest['behaviorParameterHash'] = brain.checkpoint_hash()
        brain.interface = 'fixture-only'
        brain.status_scale = torch.tensor(.15)
        return brain, bank

    def test_margin_protects_actions_not_identical_amplitudes(self):
        p,q,y,x,m,s = measured_pair()
        q[10,1,2] = 1.35
        a,b = [measure(z,y,x,m,s) for z in (p,q)]
        self.assertFalse(contextual_gate(a,b))
        self.assertTrue(margin_gate(a,b)['passed'])
        for value in (1.09,.999):
            q[10,1,2] = value
            self.assertFalse(margin_gate(a,measure(q,y,x,m,s))['passed'])

    def test_motion_and_contextual_improvement_requirements_are_retained(self):
        p,q,y,x,m,s = measured_pair(); a = measure(p,y,x,m,s)
        q[0,0,1] = 1.
        self.assertFalse(margin_gate(a,measure(q,y,x,m,s))['passed'])
        self.assertFalse(margin_gate(a,a)['passed'])
        with self.assertRaises(ValueError):
            margin_gate(a,a,float('nan'))
        q[0,0,1] = float('inf')
        with self.assertRaises(ValueError):
            measure(q,y,x,m,s)

    def test_coordinate_projection_keeps_small_updates_and_bounds_outlier(self):
        original = [torch.tensor([100.,.001,-100.]),torch.tensor([1.,.00001,-1.])]
        projected = project_coordinates(original)
        torch.testing.assert_close(projected[0],torch.tensor([.03,.001,-.03]))
        torch.testing.assert_close(projected[1],torch.tensor([.0001,.00001,-.0001]))
        self.assertEqual(float(original[0][0]),100.)
        with self.assertRaises(ValueError):
            project_coordinates([torch.tensor([float('nan')]),torch.ones(1)])

    def test_unmocked_recurrent_attempt_is_bounded_and_restores_rejection(self):
        brain,bank = self.bank(); before = brain.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()):
            row = attempt(brain,bank)
        self.assertFalse(row['accepted'])
        self.assertEqual(before,brain.checkpoint_hash())
        self.assertEqual(len(row['linearProposal']['requestedOutputChange']),18)
        self.assertEqual(len(row['trials']),6)
        self.assertFalse(row['globalControl']['permanentUpdate'])
        self.assertTrue(all(p.grad is None for p in brain.parameters()))

    def test_selected_batch_cannot_override_full_bank_rejection(self):
        brain,bank = self.bank(); before = brain.checkpoint_hash()
        gates = [{'passed':True}] + [{'passed':True},{'passed':False}]*6
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'engine.layout_margin_projected_train.margin_gate',side_effect=gates):
            row = attempt(brain,bank)
        self.assertFalse(row['accepted'])
        self.assertTrue(all(t['fullBank'] is not None for t in row['trials']))
        self.assertEqual(before,brain.checkpoint_hash())

    def test_exact_accepted_snapshot_is_saved_without_second_update(self):
        brain,bank = self.bank(); before = brain.checkpoint_hash()
        labels = bank.batch(None,None)[1].clone()
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'engine.layout_margin_projected_train.margin_gate',return_value={'passed':True}):
            row = attempt(brain,bank)
        self.assertTrue(row['accepted']); self.assertEqual(len(row['trials']),1)
        self.assertNotEqual(before,brain.checkpoint_hash())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'candidate.npz'
            save_model(path,brain); verify_saved_snapshot(path,brain)
            self.assertEqual(row['afterHash'],brain.checkpoint_hash())
            with self.assertRaises(FileExistsError):
                save_model(path,brain)
        torch.testing.assert_close(labels,bank.batch(None,None)[1],atol=0,rtol=0)

    def test_exception_after_changed_forward_rolls_back(self):
        brain,bank = self.bank(); brain.eval(); before = brain.checkpoint_hash()
        def fail(*args):
            if brain.checkpoint_hash() != before:
                raise RuntimeError('Injected changed-forward failure')
            return frozen_predictions(*args)
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'engine.layout_margin_projected_train.frozen_predictions',side_effect=fail):
            with self.assertRaises(RuntimeError):
                attempt(brain,bank)
        self.assertEqual(before,brain.checkpoint_hash()); self.assertFalse(brain.training)


if __name__ == '__main__':
    unittest.main()
