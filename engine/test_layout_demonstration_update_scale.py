import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from .layout_demonstration_update_scale import validate_scale, resolve_update_scale, scaled_learning_rates
from .layout_recovery_demonstrations import file_hash


class UpdateScaleTests(unittest.TestCase):
    def test_adam_step_scales_displacement_but_not_saved_moments_or_controls(self):
        p = torch.nn.Parameter(torch.tensor([.1, -.2], dtype=torch.float64))
        source = torch.optim.Adam([p], lr=.0003, eps=1e-14)
        p.grad = torch.tensor([.8, -.4], dtype=torch.float64)
        source.step()
        original = p.detach().clone()
        results = []
        for scale in (1., 4.):
            q = torch.nn.Parameter(original.clone())
            optimizer = torch.optim.Adam([q], lr=.0003, eps=1e-14)
            # Adam state tensors are cloned to avoid shared moments between trials.
            import copy
            optimizer.load_state_dict(copy.deepcopy(source.state_dict()))
            q.grad = torch.tensor([-.2, .5], dtype=torch.float64)
            with scaled_learning_rates(optimizer, scale):
                self.assertEqual(optimizer.param_groups[0]['lr'], .0003*scale)
                optimizer.step()
            self.assertEqual(optimizer.param_groups[0]['lr'], .0003)
            results.append((q.detach()-original, optimizer.state[q]))
        torch.testing.assert_close(results[1][0], 4*results[0][0], rtol=1e-10, atol=1e-15)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(results[0][1][key], results[1][1][key], rtol=0, atol=0)

    def test_rates_restore_on_exception_and_bounds_fail_closed(self):
        p = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([p], lr=.001)
        with self.assertRaises(RuntimeError):
            with scaled_learning_rates(optimizer, 4.):
                raise RuntimeError('deliberate failure')
        self.assertEqual(optimizer.param_groups[0]['lr'], .001)
        for bad in (0., -.1, .099, 4.01, float('nan'), float('inf'), True, '4'):
            with self.assertRaises(ValueError):
                validate_scale(bad)

    def test_explicit_revision_inherits_on_later_resume_and_verifies_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder/'manifest.json'
            path.write_text(json.dumps({}))
            continuation = {'record': {'sourceRun': str(folder),
                                      'sourceFileSHA256': {'manifest.json': file_hash(path)}}}
            args = SimpleNamespace(revise_step_scale=4.)
            scale, record = resolve_update_scale(args, continuation)
            self.assertEqual(scale, 4.)
            self.assertTrue(record['changed'])
            self.assertTrue(record['savedAdamMomentsPreserved'])
            path.write_text(json.dumps({'effectiveStepScale': 4.}))
            with self.assertRaises(ValueError):
                resolve_update_scale(args, continuation)
            continuation['record']['sourceFileSHA256']['manifest.json'] = file_hash(path)
            scale, record = resolve_update_scale(SimpleNamespace(), continuation)
            self.assertEqual(scale, 4.)
            self.assertFalse(record['changed'])
            self.assertFalse(record['explicitRevision'])
            self.assertEqual(resolve_update_scale(SimpleNamespace(), None)[0], 1.)


if __name__ == '__main__':
    unittest.main()
