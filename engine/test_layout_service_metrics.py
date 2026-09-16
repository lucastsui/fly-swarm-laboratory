import unittest
from .layout_recovery_eval import service_metrics


class ServiceMetricsTests(unittest.TestCase):
    def test_early_deliveries_cannot_pass_moved_gate(self):
        result = service_metrics([10., 100., 500., 600., 700.], 1200., 600.)
        self.assertTrue(result['sustainedSuccess'])
        self.assertEqual(result['postMoveProducts'], 1)
        self.assertFalse(result['postMoveSuccess'])

    def test_three_new_deliveries_pass_moved_gate(self):
        result = service_metrics([601., 800., 1199.], 1200., 600.)
        self.assertEqual(result['postMoveProducts'], 3)
        self.assertTrue(result['postMoveSuccess'])

    def test_ordinary_gate_and_invalid_horizon(self):
        self.assertEqual(service_metrics([10., 20., 40.], 100.), {'sustainedSuccess': False})
        self.assertEqual(service_metrics([10., 20., 70.], 100.), {'sustainedSuccess': True})
        self.assertNotIn('postMoveSuccess', service_metrics([10., 20., 70.], 100.))
        with self.assertRaises(ValueError):
            service_metrics([], 600., 600.)


if __name__ == '__main__':
    unittest.main()
