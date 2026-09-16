import unittest
import numpy as np
from .layout_forward_repeatability import compare_outputs


class RepeatabilityTests(unittest.TestCase):
    def test_exact(self):
        x = np.zeros((4, 2, 3), np.float32)
        self.assertTrue(compare_outputs(x, x.copy())['bitwiseEqual'])

    def test_threshold_and_location(self):
        a = np.zeros((4, 2, 3), np.float32)
        b = a.copy(); b[2, 1, 2] = .026
        r = compare_outputs(a, b)
        self.assertEqual(r['firstDifferentFrameActorHead'], [2, 1, 2])
        self.assertEqual(r['gripDecisionDifferences'], 1)
        self.assertEqual(r['differentValues'], 1)

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            compare_outputs(np.array([np.nan]), np.array([np.nan]))


if __name__ == '__main__':
    unittest.main()
