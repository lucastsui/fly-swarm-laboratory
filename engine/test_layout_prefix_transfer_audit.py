import copy
import unittest
from .layout_prefix_transfer_audit import prefix_difference


class PrefixDifferenceTests(unittest.TestCase):
    def record(self):
        return {'rawGrip': [.9, 1., 1.1], 'protectedLoadedGrip': [.8, 1.2],
                'positiveFocus': [True, False, True], 'protectedLoadedMask': [[True, False, True]]}

    def test_identity(self):
        a = self.record()
        for row in prefix_difference(a, copy.deepcopy(a)).values():
            self.assertEqual(row, {'maxAbs': 0., 'rms': 0., 'thresholdFlips': 0})

    def test_strict_threshold_and_magnitude(self):
        a, b = self.record(), self.record()
        b['rawGrip'] = [1.1, 1.01, 1.]
        report = prefix_difference(a, b)
        self.assertEqual(report['rawGrip']['thresholdFlips'], 3)
        self.assertAlmostEqual(report['rawGrip']['maxAbs'], .2)
        self.assertEqual(report['protectedLoadedGrip']['thresholdFlips'], 0)

    def test_changed_labels_or_masks_rejected(self):
        a = self.record()
        for field in ('positiveFocus', 'protectedLoadedMask'):
            b = self.record(); b[field] = []
            with self.assertRaises(ValueError):
                prefix_difference(a, b)

    def test_invalid_outputs_rejected(self):
        for field in ('rawGrip', 'protectedLoadedGrip'):
            for value in ([], [1.], [float('nan')]*3, [float('inf')]*2):
                a, b = self.record(), self.record(); b[field] = value
                with self.assertRaises(ValueError):
                    prefix_difference(a, b)


if __name__ == '__main__':
    unittest.main()
