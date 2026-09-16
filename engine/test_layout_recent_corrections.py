import tempfile
import unittest
from pathlib import Path
import numpy as np
from .layout_recent_corrections import pop_recent_correction
from .layout_recovery_protocol import Exchange, pack_rollout
from .layout_recovery_world import INTERFACE, PHYSICS


class RecentCorrectionTests(unittest.TestCase):
    def prepare(self, root):
        root = Path(root)
        (root/'experience').mkdir()
        (root/'candidate-0.npz').write_bytes(b'fixture')
        exchange = Exchange(root, 'test-token', 'test-fixed', run_id='f'*32)
        exchange.publish(0, 'candidate-0.npz', 'test-parameter')
        for i in range(3):
            meta = {'id': f'{i:032x}', 'runId': 'f'*32, 'fixedHash': 'test-fixed',
                    'version': 0, 'parameterHash': 'test-parameter', 'interface': INTERFACE,
                    'physics': PHYSICS, 'control': 'learner-only', 'teacherActions': False}
            blob = pack_rollout(np.zeros((32, 1, 297), np.float32), np.zeros((32, 1, 3), np.float32),
                                np.zeros((166700, 1), np.float32), meta)
            self.assertEqual(exchange.accept(blob), 'accepted')
        return exchange

    def test_selects_latest_and_preserves_old_order_and_all_raw_files(self):
        with tempfile.TemporaryDirectory() as root:
            exchange = self.prepare(root)
            files = {p: p.read_bytes() for p in (Path(root)/'experience').iterdir()}
            newest = pop_recent_correction(exchange)
            self.assertEqual(newest[3]['id'], f'{2:032x}')
            self.assertEqual(exchange.consumed, 1)
            self.assertEqual(exchange.received, 3)
            self.assertEqual(exchange.pop()[3]['id'], f'{0:032x}')
            self.assertEqual(exchange.pop()[3]['id'], f'{1:032x}')
            self.assertIsNone(pop_recent_correction(exchange))
            for path, data in files.items():
                self.assertEqual(path.read_bytes(), data)

    def test_staleness_checks_remain_in_force(self):
        with tempfile.TemporaryDirectory() as root:
            exchange = self.prepare(root)
            exchange.update = 81
            self.assertIsNone(pop_recent_correction(exchange))
            self.assertEqual(exchange.consumed, 0)
            self.assertEqual(len(list((Path(root)/'experience').iterdir())), 3)


if __name__ == '__main__':
    unittest.main()
