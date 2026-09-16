import unittest
import numpy as np
from .layout_connectivity_audit import active_reverse_graph, ancestor_distances


class ConnectivityAuditTests(unittest.TestCase):
    def test_direction_and_disabled_presynaptic_cell(self):
        # Biological directed edges: 0->1->2, 3->2, and 2->4 (a descendant).
        crow = np.array([0, 0, 1, 3, 3, 4])
        col = np.array([0, 1, 3, 2])
        values = np.array([1., -1., 1., 1.])
        graph = active_reverse_graph(crow, col, values, np.ones(5))
        np.testing.assert_equal(ancestor_distances(graph, [2]), [2., 1., 0., 1., np.inf])
        mask = np.ones(5)
        mask[1] = 0
        blocked = active_reverse_graph(crow, col, values, mask)
        np.testing.assert_equal(ancestor_distances(blocked, [2]), [np.inf, np.inf, 0., 1., np.inf])
        self.assertEqual(graph.nnz, 4)
        self.assertEqual(blocked.nnz, 3)

    def test_zero_weight_and_multiple_motors(self):
        graph = active_reverse_graph(np.array([0, 0, 1, 2]), np.array([0, 1]),
                                     np.array([0., -1.]), np.ones(3))
        np.testing.assert_equal(ancestor_distances(graph, [0, 2]), [0., 1., 0.])
        with self.assertRaises(ValueError):
            active_reverse_graph(np.array([0, 0]), np.array([], dtype=int), np.array([]), np.ones(2))


if __name__ == '__main__':
    unittest.main()
