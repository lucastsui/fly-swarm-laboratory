import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .layout_recovery_protocol import Exchange, pack_rollout, unpack_rollout
from .layout_recovery_world import CHANNELS, INTERFACE, PHYSICS
from .layout_recovery_train import recurrent_loss
from .layout_recovery_eval import wilson


class ProtocolTests(unittest.TestCase):
    def packet(self, **changes):
        metadata = {'id': 'a'*32, 'runId': 'run', 'worker': 'spark2', 'version': 0,
                    'parameterHash': 'parameter', 'fixedHash': 'fixed',
                    'interface': INTERFACE, 'physics': PHYSICS, 'control': 'learner-only'}
        metadata.update(changes)
        return pack_rollout(np.zeros((32, 4, CHANNELS), np.float32), np.zeros((32, 4, 3), np.float32),
                            np.zeros((166700, 4), np.float32), metadata)

    def test_roundtrip_and_metadata(self):
        x, y, state, meta = unpack_rollout(self.packet())
        self.assertEqual(x.shape, (32, 4, CHANNELS))
        self.assertEqual(state.shape, (166700, 4))
        self.assertEqual(meta['control'], 'learner-only')

    def test_teacher_actions_and_interface_mismatch_rejected(self):
        for changes in ({'control': 'teacher'}, {'interface': 'old'}, {'physics': 'old'}, {'id': '../escape'}):
            with self.assertRaises(ValueError):
                unpack_rollout(self.packet(**changes))

    def test_queue_identity_deduplication_staleness_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'experience').mkdir()
            (path/'candidate-0.npz').write_bytes(b'fixture')
            exchange = Exchange(path, 'test-token', 'fixed', 'run', max_queue=1)
            exchange.publish(0, 'candidate-0.npz', 'parameter')
            self.assertEqual(exchange.accept(self.packet()), 'accepted')
            self.assertEqual(exchange.accept(self.packet()), 'duplicate')
            self.assertEqual(exchange.accept(self.packet(id='b'*32)), 'full')
            self.assertIsNotNone(exchange.pop())
            self.assertEqual(exchange.consumed, 1)
            with self.assertRaises(ValueError):
                exchange.accept(self.packet(runId='other'))
            with self.assertRaises(ValueError):
                exchange.accept(self.packet(parameterHash='other'))
            exchange.update = 81
            self.assertEqual(exchange.accept(self.packet(id='c'*32)), 'stale')
            self.assertEqual(exchange.manifest()['checkpoint']['sha256'], hashlib.sha256(b'fixture').hexdigest())

    def test_longer_gradient_window_with_fake_existing_motor_cells(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.1))
                for name, index in [('forward', 0), ('left', 1), ('right', 2), ('interact', 3)]:
                    setattr(self, 'motor_'+name, torch.tensor([index]))

            def forward(self, x, substeps, state, weights):
                state = .75*state + self.weight*x.T
                return None, state
        model = TinyModel()
        heads = recurrent_loss(model, torch.ones((8, 2, 4)), torch.zeros((8, 2, 3)),
                               torch.zeros((4, 2)), 4, None)
        heads.sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad))
        self.assertGreater(float(model.weight.grad), 0)

    def test_small_sample_uncertainty_is_not_false_certainty(self):
        low, high = wilson(8, 8)
        self.assertLess(low, .95)
        self.assertAlmostEqual(high, 1.)


if __name__ == '__main__':
    unittest.main()
