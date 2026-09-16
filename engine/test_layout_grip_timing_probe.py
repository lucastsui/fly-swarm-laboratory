import unittest
import numpy as np
import torch
from .layout_grip_timing_probe import analyze, predict_all
from .test_layout_cargo_response_probe import CargoBrain


class TimingProbeTests(unittest.TestCase):
    def test_event_alignment_and_delayed_motor_threshold(self):
        x = np.zeros((20, 1, 297), np.float32)
        y = np.full((20, 1, 3), .2, np.float32)
        p = y.copy()
        x[10:, 0, 27] = 1.
        x[11:, 0, 126] = 1.
        y[10, 0, 2] = 2.2
        p[13:, 0, 2] = 1.1
        rows = analyze(x, y, p, [{'start': 90, 'eventTick': 100}])
        self.assertEqual(rows[0]['teacherPulseSecondsFromEvent'], .05)
        self.assertEqual(rows[0]['tasteOnsetSecondsBeforeEvent'], 0.)
        self.assertAlmostEqual(rows[0]['firstGripThresholdSecondsAfterEvent'], .15)
        self.assertEqual(rows[0]['timeline'][8]['secondsFromPickupLabel'], 0.)
        p[:, :, 2] = .2
        self.assertIsNone(analyze(x, y, p, [{'start': 90, 'eventTick': 100}])[0]['firstGripThresholdSecondsAfterEvent'])

    def test_prediction_is_frozen_and_does_not_modify_prefix(self):
        model = CargoBrain().eval().requires_grad_(False)
        before = model.checkpoint_hash()
        x = torch.ones((6, 4, 297))
        state = torch.zeros((4, 4))
        prediction = predict_all(model, x, state)
        self.assertEqual(prediction.shape, (6, 4, 3))
        self.assertEqual(model.checkpoint_hash(), before)
        self.assertTrue(torch.equal(state, torch.zeros_like(state)))
        model.train()
        with self.assertRaises(ValueError):
            predict_all(model, x, state)


if __name__ == '__main__':
    unittest.main()
