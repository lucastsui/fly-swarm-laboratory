import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .layout_box_solver import solve_box, bounded_direction
from .layout_box_constrained_train import attempt
from . import test_layout_margin_projected_train as margin_tests
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import save_model
from .layout_guarded_snapshot_train import verify_saved_snapshot


class BoxSolverTests(unittest.TestCase):
    def test_matches_independent_bounded_least_squares_reference(self):
        try:
            from scipy.optimize import lsq_linear
        except ImportError:
            self.skipTest('Independent SciPy reference is tested on laptop; not a training runtime dependency')
        for seed in range(4):
            rng = np.random.default_rng(seed)
            a = rng.normal(size=(7, 12)); a /= np.linalg.norm(a, axis=1)[:, None]
            b = rng.normal(size=7)*1.5
            lo, hi = np.full(12, -.2), np.full(12, .2)
            augmented = np.concatenate((a, np.sqrt(.001)*np.eye(12)))
            target = np.concatenate((b, np.zeros(12)))
            reference = lsq_linear(augmented, target, bounds=(lo, hi), tol=1e-12)
            self.assertTrue(reference.success)
            u, report = solve_box(*[torch.tensor(z) for z in (a, b, lo, hi)], tolerance=1e-9)
            self.assertLessEqual(abs(report['boundedObjective']-reference.cost), 1e-8)
            self.assertTrue(np.all(u.numpy() >= lo) and np.all(u.numpy() <= hi))

    def test_unsaturated_solution_matches_closed_form(self):
        a = torch.tensor([[1., .2, .3], [.1, 1., .5]], dtype=torch.float64)
        b = torch.tensor([.1, -.2], dtype=torch.float64)
        limits = torch.ones(3, dtype=torch.float64)
        result, report = solve_box(a, b, -limits, limits, tolerance=1e-10)
        expected = a.T@torch.linalg.solve(a@a.T+.001*torch.eye(2, dtype=a.dtype), b)
        torch.testing.assert_close(result, expected, atol=1e-9, rtol=1e-7)
        self.assertTrue(report['converged'])

    def test_resolves_clipping_error_inside_constraints(self):
        a = torch.tensor([[10., 1.]], dtype=torch.float64)
        b = torch.tensor([20.], dtype=torch.float64)
        limits = torch.ones(2, dtype=torch.float64)
        result, report = solve_box(a, b, -limits, limits)
        torch.testing.assert_close(result, limits)
        self.assertLess(report['boundedObjective'], report['clippedUnconstrainedObjective'])
        self.assertTrue(report['converged'])

    def test_asymmetric_bounds_and_iteration_limit(self):
        a = torch.eye(2)
        result, report = solve_box(a, torch.tensor([-1., 1.]), torch.tensor([0., -1.]), torch.tensor([1., 0.]), iterations=1)
        torch.testing.assert_close(result, torch.zeros(2))
        self.assertEqual(report['iterations'], 1)

    def test_invalid_inputs_fail_closed(self):
        a, b, low, high = torch.eye(2), torch.ones(2), -torch.ones(2), torch.ones(2)
        for options in ({'iterations': 0}, {'iterations': 5000}, {'damping': float('nan')}, {'tolerance': 0}):
            with self.assertRaises(ValueError):
                solve_box(a, b, low, high, **options)
        with self.assertRaises(ValueError):
            solve_box(a, b, torch.ones(2), high)
        with self.assertRaises(ValueError):
            solve_box(a, torch.tensor([float('nan'), 1.]), low, high)

    def test_active_columns_and_global_parameter_limits(self):
        base = [torch.tensor([1.999, 0., 0.]), torch.tensor([-.09999, 0.])]
        rows = [[torch.tensor([2., 0., 1.]), torch.tensor([-10., 0.])],
                [torch.tensor([0., 0., 1.]), torch.tensor([1., 0.])]]
        delta, report = bounded_direction(rows, [1., -1.], base, iterations=128)
        self.assertEqual(report['activeJacobianCoordinates'], [2, 1])
        self.assertEqual(float(delta[0][1]), 0.)
        self.assertEqual(float(delta[1][1]), 0.)
        for original, change, cap, limits in zip(base, delta, (.03, .0001), ((-2., 2.), (-.1, .1))):
            self.assertTrue(torch.all(change.abs() <= cap+1e-7))
            self.assertTrue(torch.all((original+change >= limits[0]) & (original+change <= limits[1])))
        actual = [sum(float((g*d).sum()) for g,d in zip(row,delta)) for row in rows]
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(report['predictedBoundedOutputChange']))
        self.assertLessEqual(report['boundedObjective'], report['zeroObjective'])

    def test_zero_gradient_rows_rejected(self):
        base = [torch.zeros(2), torch.zeros(1)]
        with self.assertRaises(ValueError):
            bounded_direction([[torch.zeros(2), torch.zeros(1)]], [1.], base)


class BoxTrainingTests(unittest.TestCase):
    def bank(self):
        return margin_tests.MarginProjectedTests().bank()

    def test_unmocked_recurrence_is_bounded_and_rejected_parent_restored(self):
        model, bank = self.bank(); before = model.checkpoint_hash()
        labels = bank.batch(None, None)[1].clone()
        with contextlib.redirect_stdout(io.StringIO()):
            row = attempt(model, bank, solver_iterations=64)
        self.assertFalse(row['accepted'])
        self.assertEqual(before, model.checkpoint_hash())
        self.assertEqual(len(row['trials']), 6)
        self.assertEqual(len(row['boundedLinearSolve']['requestedOutputChange']), 18)
        torch.testing.assert_close(labels, bank.batch(None, None)[1], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_full_bank_rejection_cannot_be_bypassed(self):
        model, bank = self.bank(); before = model.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'engine.layout_box_constrained_train.margin_gate', side_effect=[{'passed':True}, {'passed':False}]*6):
            row = attempt(model, bank, solver_iterations=64)
        self.assertFalse(row['accepted']); self.assertEqual(before, model.checkpoint_hash())
        self.assertTrue(all(t['fullBank'] is not None for t in row['trials']))

    def test_accepted_snapshot_saved_exactly_without_second_step(self):
        model, bank = self.bank(); before = model.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'engine.layout_box_constrained_train.margin_gate', return_value={'passed':True}):
            row = attempt(model, bank, solver_iterations=64)
        self.assertTrue(row['accepted']); self.assertNotEqual(before, model.checkpoint_hash())
        self.assertEqual(len(row['trials']), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'candidate.npz'
            save_model(path, model); verify_saved_snapshot(path, model)
            self.assertEqual(row['afterHash'], model.checkpoint_hash())
            with self.assertRaises(FileExistsError):
                save_model(path, model)

    def test_changed_forward_failure_rolls_back_weights_and_mode(self):
        model, bank = self.bank(); model.eval(); before = model.checkpoint_hash()
        def fail(*args):
            if model.checkpoint_hash() != before:
                raise RuntimeError('Injected changed-forward failure')
            return frozen_predictions(*args)
        with contextlib.redirect_stdout(io.StringIO()), patch('engine.layout_box_constrained_train.frozen_predictions', side_effect=fail):
            with self.assertRaises(RuntimeError):
                attempt(model, bank, solver_iterations=64)
        self.assertEqual(before, model.checkpoint_hash()); self.assertFalse(model.training)


if __name__ == '__main__':
    unittest.main()
