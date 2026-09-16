import unittest
from .fixed_layout_review import trial_metrics, review


class ReviewTests(unittest.TestCase):
    def trial(self):
        return {'seed': 1, 'products': 3, 'returns': 2, 'deliveryTimes': [100., 200., 400.],
                'agentContributions': [{'transfers': x} for x in (2, 1, 0, 1)],
                'interactionEvents': [
                    {'event': 'pickup', 'fly': 0, 'nearestBox': 0, 'time': 5.},
                    {'event': 'pickup', 'fly': 1, 'nearestBox': 1, 'time': 5.},
                    {'event': 'return', 'fly': 0, 'nearestBox': 0, 'time': 6.},
                    {'event': 'return', 'fly': 1, 'nearestBox': 1, 'time': 9.}],
                'interactionEventsTruncated': False}

    def test_contribution_gaps_and_immediate_returns(self):
        got = trial_metrics(self.trial(), 600.)
        self.assertTrue(got['sustained'])
        self.assertEqual(got['productiveFlies'], 3)
        self.assertFalse(got['allFourMadeTransfers'])
        self.assertEqual(got['recordedImmediatePickupReturns'], 1)
        self.assertEqual(got['longestNoDeliverySeconds'], 200.)
        self.assertEqual(got['terminalNoDeliverySeconds'], 200.)
        self.assertEqual(got['secondHalfProducts'], 1)

    def test_no_deliveries_and_truncated_events_are_explicit(self):
        trial = self.trial(); trial.update(products=0, deliveryTimes=[], interactionEventsTruncated=True)
        got = trial_metrics(trial, 600.)
        self.assertFalse(got['sustained'])
        self.assertFalse(got['eventRecordingComplete'])
        self.assertEqual(got['terminalNoDeliverySeconds'], 600.)

    def test_rejects_assisted_or_mismatched_evidence(self):
        for report in ({}, {'teacher': True}):
            with self.assertRaises(ValueError): review(report)
        trial = self.trial(); trial['products'] = 4
        with self.assertRaises(ValueError): trial_metrics(trial, 600.)


if __name__ == '__main__': unittest.main()
