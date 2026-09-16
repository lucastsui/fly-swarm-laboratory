import unittest
import torch
from .layout_box_solver import bounded_direction as original_direction
from .layout_weighted_box_solver import bounded_direction


class WeightedBoxTests(unittest.TestCase):
    def fixture(self):
        return [torch.zeros(1),torch.zeros(1)], [[torch.ones(1),torch.zeros(1)] for _ in range(2)]

    def test_unit_weights_reproduce_original_solver_and_unweighted_predictions(self):
        base,rows = self.fixture()
        old,old_report = original_direction(rows,[.01,0.],base,iterations=128)
        new,report = bounded_direction(rows,[.01,0.],base,[1.,1.],iterations=128)
        for a,b in zip(old,new):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(report['predictedBoundedOutputChange'],old_report['predictedBoundedOutputChange'])
        self.assertEqual(report['boundedObjective'],old_report['boundedObjective'])

    def test_stronger_zero_motion_penalty_reduces_motion_in_conflicting_rows(self):
        base,rows = self.fixture()
        old,_ = bounded_direction(rows,[.01,0.],base,[1.,1.],iterations=256)
        new,report = bounded_direction(rows,[.01,0.],base,[1.,16.],iterations=256)
        self.assertLess(float(new[0].abs()),float(old[0].abs())/4)
        self.assertGreater(float(new[0]),0.)
        actual = [sum(float((g*d).sum()) for g,d in zip(row,new)) for row in rows]
        torch.testing.assert_close(torch.tensor(actual),torch.tensor(report['predictedBoundedOutputChange']))
        weighted = torch.tensor(report['predictedWeightedNormalizedOutputChange'])
        unweighted = torch.tensor(report['predictedNormalizedOutputChange'])
        torch.testing.assert_close(weighted,unweighted*torch.tensor([1.,4.]))
        self.assertLessEqual(report['boundedObjective'],report['zeroObjective'])

    def test_bounds_and_inactive_coordinates_are_unchanged(self):
        base = [torch.tensor([1.999,0.]),torch.tensor([-.09999,0.])]
        rows = [[torch.tensor([2.,0.]),torch.tensor([-10.,0.])],
                [torch.tensor([1.,0.]),torch.tensor([1.,0.])]]
        delta,report = bounded_direction(rows,[1.,0.],base,[1.,16.],iterations=128)
        for b,d,cap,limits in zip(base,delta,(.03,.0001),((-2.,2.),(-.1,.1))):
            self.assertEqual(float(d[1]),0.)
            self.assertTrue(torch.all(d.abs() <= cap+1e-7))
            self.assertTrue(torch.all((b+d >= limits[0]) & (b+d <= limits[1])))
        self.assertEqual(report['activeJacobianCoordinates'],[1,1])

    def test_invalid_weights_and_zero_gradients_fail_closed(self):
        base,rows = self.fixture()
        for weights in ([1.],[1.,0.],[1.,-1.],[1.,float('nan')],[1.,float('inf')]):
            with self.assertRaises(ValueError):bounded_direction(rows,[.01,0.],base,weights)
        with self.assertRaises(ValueError):bounded_direction([[torch.zeros(1),torch.zeros(1)]],[0.],base,[16.])


if __name__ == '__main__':unittest.main()
