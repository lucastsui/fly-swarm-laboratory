import contextlib
import copy
import io
import unittest
from unittest.mock import patch
import torch
from .layout_contextual_operation_probe import measure, contextual_gate, probe
from .layout_operation_prefix import CONTEXTS, KINDS
from .layout_demonstration_step_probe import frozen_predictions
from .test_layout_pickup_preservation_probe import fixture as parent_fixture


def measured_pair():
    specs = [dict(kind=k, category=c, start=20, stop=116, lossStart=84, focusTick=100)
             for k in KINDS for c in CONTEXTS]
    p = torch.zeros(32, 16, 3)
    x = torch.zeros(32, 16, 297)
    y = torch.ones(32, 16, 3)*.2
    for i, spec in enumerate(specs):
        pickup = spec['category'] == CONTEXTS[0]
        p[:, i, 2] = .8 if pickup else 1.4
        x[:, i, 126] = 0 if pickup else 1
        y[16, i, 2] = .2 if spec['category'] == CONTEXTS[2] else 2.2
    y[10, 1, 2] = 2.2  # Protected valid loaded operation outside the focus frame.
    motion = p[..., :2].clone()
    q = p.clone()
    for i, spec in enumerate(specs):
        if spec['category'] == CONTEXTS[0]:
            q[16, i, 2] += .01
        elif spec['category'] == CONTEXTS[2]:
            q[16, i, 2] -= .02
    return p, q, y, x, motion, specs


def fixture():
    brain, bank = parent_fixture()
    x, y, state, motion = bank.batch(None, None)
    with torch.no_grad():
        brain.tonic[3] = .008  # Actual negative-return margin deficit in toy dynamics.
        for i, spec in enumerate(bank.manifest['windows']):
            if spec['category'] == CONTEXTS[0]:
                x[:, i, 3] = .5
        motion.copy_(frozen_predictions(brain, x, state, 64)[..., :2])
    bank.manifest['parameterHash'] = brain.checkpoint_hash()
    return brain, bank


class ContextualOperationTests(unittest.TestCase):
    def test_correct_return_and_pickup_changes_pass_without_preserving_wrong_returns(self):
        p, q, y, x, motion, specs = measured_pair()
        a, b = [measure(z, y, x, motion, specs) for z in (p, q)]
        self.assertTrue(contextual_gate(a, b))
        self.assertEqual(len(a['protectedLoadedGrip']), 9)
        self.assertEqual(a['protectedLoadedGrip'], b['protectedLoadedGrip'])
        self.assertLess(b['opposedReturnHinge'], a['opposedReturnHinge'])

    def test_uniform_shift_does_not_count_as_contextual_learning(self):
        p, _, y, x, motion, specs = measured_pair()
        a = measure(p, y, x, motion, specs)
        for shift in (-.01, .01):
            q = p.clone(); q[..., 2] += shift
            self.assertFalse(contextual_gate(a, measure(q, y, x, motion, specs)))

    def test_supported_loaded_frames_and_motor_drift_are_protected(self):
        p, q, y, x, motion, specs = measured_pair()
        a = measure(p, y, x, motion, specs)
        for index, value in [((10, 1, 2), 1.404), ((16, 1, 2), .999), ((0, 0, 1), 1.)]:
            bad = q.clone(); bad[index] = value
            self.assertFalse(contextual_gate(a, measure(bad, y, x, motion, specs)))

    def test_nonfinite_and_changed_physical_masks_rejected(self):
        p, q, y, x, motion, specs = measured_pair()
        a, b = [measure(z, y, x, motion, specs) for z in (p, q)]
        bad = copy.deepcopy(b); bad['protectedLoadedGrip'][0] = float('nan')
        with self.assertRaises(ValueError):
            contextual_gate(a, bad)
        bad = copy.deepcopy(b); bad['protectedLoadedMask'][0][0] = True
        with self.assertRaises(ValueError):
            contextual_gate(a, bad)
        broken = x.clone(); broken[16, 0, 126] = 1.
        with self.assertRaises(ValueError):
            measure(p, y, broken, motion, specs)

    def test_unmocked_recurrent_individual_proposal_restores_every_parameter(self):
        brain, bank = fixture(); before = brain.checkpoint_hash()
        labels = bank.batch(None, None)[1].clone()
        with contextlib.redirect_stdout(io.StringIO()):
            report = probe(brain, bank)
        self.assertEqual(before, brain.checkpoint_hash())
        self.assertEqual(len(report['trials']), 6)
        residual = report['linearProposal']['requestedOutputChange']
        self.assertEqual(len(residual), 18)
        for i, spec in enumerate(report['selections']):
            if spec['category'] == CONTEXTS[2]:
                self.assertLess(residual[i], 0.)
            elif spec['category'] in (CONTEXTS[1], CONTEXTS[3]):
                self.assertEqual(residual[i], 0.)
        self.assertFalse(report['candidateSaved'])
        self.assertTrue(report['parametersRestored'])
        torch.testing.assert_close(labels, bank.batch(None, None)[1], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in brain.parameters()))

    def test_forward_failure_after_perturbation_restores_model_and_mode(self):
        brain, bank = fixture(); brain.eval(); before = brain.checkpoint_hash()
        def fail(*args):
            if brain.checkpoint_hash() != before:
                raise RuntimeError('Injected changed-model replay failure')
            return frozen_predictions(*args)
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'engine.layout_contextual_operation_probe.frozen_predictions', side_effect=fail):
            with self.assertRaises(RuntimeError):
                probe(brain, bank)
        self.assertEqual(before, brain.checkpoint_hash())
        self.assertFalse(brain.training)

    def test_selected_batch_cannot_bypass_full_bank_guard(self):
        brain, bank = fixture(); before = brain.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'engine.layout_contextual_operation_probe.contextual_gate', side_effect=[True, False]*6):
            report = probe(brain, bank)
        self.assertTrue(all(row['batchGuardPassed'] for row in report['trials']))
        self.assertTrue(all(row['fullBank'] is not None for row in report['trials']))
        self.assertTrue(all(not row['allTrainingGuards'] for row in report['trials']))
        self.assertEqual(before, brain.checkpoint_hash())


if __name__ == '__main__':
    unittest.main()
