import unittest
import numpy as np
from .layout_cooldown_labels import extend_labels, verify_equivalent_replay
from .layout_recovery_demonstrations import collect_episode


class CooldownLabelTests(unittest.TestCase):
    def test_extended_pulses_replay_identical_real_physics(self):
        arrays, meta = collect_episode(9360123, 'compact', 500)
        original = arrays['labels'].copy()
        revised = extend_labels(original, meta)
        audit = verify_equivalent_replay(arrays, meta, revised)
        self.assertGreater(audit['changedGripLabels'], 0)
        self.assertTrue(audit['everyObservationBodyAndEventExactlyMatched'])
        np.testing.assert_array_equal(original, arrays['labels'])
        np.testing.assert_array_equal(original[..., :2], revised[..., :2])
        self.assertFalse(audit['isServiceEvidence'])

    def test_reject_non_equivalent_actions_and_unbounded_extension(self):
        arrays, meta = collect_episode(9360123, 'compact', 100)
        revised = arrays['labels'].copy()
        revised[0, 0, 0] += .1
        with self.assertRaises(ValueError):
            verify_equivalent_replay(arrays, meta, revised)
        revised = arrays['labels'].copy()
        fly = int(np.flatnonzero(revised[0, :, 2] <= 1.)[0])
        revised[0, fly, 2] = 2.2
        with self.assertRaisesRegex(ValueError, 'cooldown'):
            verify_equivalent_replay(arrays, meta, revised)
        with self.assertRaises(ValueError):
            extend_labels(arrays['labels'], meta, 20)


if __name__ == '__main__':
    unittest.main()
