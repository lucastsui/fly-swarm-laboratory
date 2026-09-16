import unittest
import numpy as np
import torch
from .layout_demonstration_cache import window_specs, prefix_states
from .test_layout_microfit_flow import TinyTrainBrain


class PrefixTests(unittest.TestCase):
    def test_prefix_matches_full_history_not_zero_window_reset(self):
        model = TinyTrainBrain().eval().requires_grad_(False)
        x = np.random.default_rng(1).normal(size=(15, 4, 297)).astype(np.float32)
        saved = {i: s.clone() for i, s in prefix_states(model, x, [0, 4, 8, 14])}
        state = torch.zeros(4, 4)
        with torch.no_grad():
            for i in range(15):
                if i in saved:
                    torch.testing.assert_close(saved[i], state, rtol=0, atol=0)
                _, state = model(torch.tensor(x[i]), 4, state, model.weights())
        self.assertEqual(saved[0].abs().sum(), 0.)
        self.assertGreater(saved[8].abs().sum(), 0.)
        self.assertTrue(all(s.grad_fn is None for s in saved.values()))

    def test_rejects_training_model_and_bad_positions(self):
        model = TinyTrainBrain()
        x = np.ones((5, 4, 297), np.float32)
        with self.assertRaises(ValueError):
            list(prefix_states(model, x, [0]))
        model.eval().requires_grad_(False)
        for positions in ([], [-1], [5]):
            with self.assertRaises(ValueError):
                list(prefix_states(model, x, positions))

    def test_event_window_places_contact_in_loss_and_records_release(self):
        meta = {'seed': 9360000, 'kind': 'compact', 'frames': 1000, 'dt': .05,
                'interactionEvents': [{'event': 'pickup', 'cargoAfter': 2,
                                       'cargoBefore': 0, 'time': 10.} ]}
        specs = window_specs(meta, 64, 32)
        contact = next(s for s in specs if s['category'] == 'pickup-2')
        release = next(s for s in specs if s['category'] == 'after-pickup-2')
        self.assertLessEqual(contact['lossStart'], 199)
        self.assertGreater(contact['stop'], 199)
        self.assertGreater(release['lossStart'], 199)
        self.assertEqual(len({s['start'] for s in specs}), len(specs))
        self.assertEqual(specs, window_specs(meta, 64, 32))


if __name__ == '__main__':
    unittest.main()
