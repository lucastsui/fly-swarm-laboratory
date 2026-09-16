import unittest
import torch
from .layout_recovery_gradient_probe import vector_comparison


class GradientComparisonTests(unittest.TestCase):
    def test_alignment_opposition_and_clipping(self):
        x = torch.tensor([1., -2., 1e-12, 0.])
        same = vector_comparison(x, x)
        self.assertAlmostEqual(same['cosine'], 1., places=6)
        self.assertEqual(same['surrogateNonzero'], 3)
        self.assertEqual(same['surrogateClippedBelowAdamEps'], 1)
        self.assertAlmostEqual(vector_comparison(x, -x)['cosine'], -1., places=6)
        self.assertEqual(vector_comparison(x, x, 1e-12)['surrogateClippedBelowAdamEps'], 3)

    def test_zero_vector_has_no_defined_angle(self):
        result = vector_comparison(torch.zeros(3), torch.ones(3))
        self.assertIsNone(result['cosine'])
        self.assertEqual(result['exactNonzero'], 0)


if __name__ == '__main__':
    unittest.main()
