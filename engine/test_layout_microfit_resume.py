import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .layout_microfit_resume import restore_optimizer, file_hash


class ResumeOptimizerTests(unittest.TestCase):
    def test_resumed_adam_next_update_matches_uninterrupted(self):
        def model_optimizer():
            model = torch.nn.Linear(2, 1, dtype=torch.float64)
            optimizer = torch.optim.Adam([{'params': [model.weight], 'lr': .003, 'eps': 1e-14},
                                          {'params': [model.bias], 'lr': .00001}], eps=1e-10)
            return model, optimizer
        def step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones((4, 2), dtype=torch.float64)).square().mean().backward()
            optimizer.step()
        original, optimizer = model_optimizer()
        for _ in range(3):
            step(original, optimizer)
        restored, restored_optimizer = model_optimizer()
        restored.load_state_dict(original.state_dict())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'optimizer-3.pt'
            rng = np.random.default_rng(99).bit_generator.state
            torch.save({'updates': 3, 'rng': rng, 'optimizer': optimizer.state_dict()}, path)
            data = {'start': 3, 'optimizerPath': path,
                    'record': {'sourceFileSHA256': {path.name: file_hash(path)}}}
            self.assertEqual(restore_optimizer(restored_optimizer, restored, data), rng)
            step(original, optimizer)
            step(restored, restored_optimizer)
            for a, b in zip(original.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                restore_optimizer(restored_optimizer, restored, {**data, 'start': 4})
            with self.assertRaisesRegex(ValueError, 'file changed'):
                restore_optimizer(restored_optimizer, restored, {**data,
                                  'record': {'sourceFileSHA256': {path.name: 'wrong'}}})


if __name__ == '__main__':
    unittest.main()
