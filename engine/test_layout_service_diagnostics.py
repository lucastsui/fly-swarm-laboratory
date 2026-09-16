import copy
import unittest
from .layout_service_diagnostics import diagnose_report, diagnose_trial


def event(time, kind, before, after, fly=0, box=0, distance=.6):
    return dict(time=time, event=kind, cargoBefore=before, cargoAfter=after,
                fly=fly, nearestBox=box, boxDistance=distance)


def trial():
    return {'seed': 1, 'kind': 'wide', 'products': 0, 'sustainedSuccess': False,
            'interactionEventsTruncated': False,
            'interactionEvents': [event(1, 'pickup', 0, 1), event(2, 'rejected', 1, 1, distance=2),
                                  event(3, 'return', 1, 0), event(5, 'pickup', 0, 1),
                                  event(14, 'transfer', 1, 0, box=1)],
            'interactionCounts': {'pickup': 2, 'return': 1, 'transfer': 1, 'rejected': 1}}


class ServiceDiagnosticTests(unittest.TestCase):
    def test_distinguishes_short_returns_from_forward_transfers(self):
        result = diagnose_trial(trial())
        self.assertEqual(result['returnsWithin2Seconds'], 1)
        self.assertEqual(result['medianForwardCarrySeconds'], 9)
        self.assertEqual(result['forwardTransfersIncludingDelivery'], 1)
        self.assertEqual(result['rejected']['outsideGripRange'], 1)
        self.assertEqual(result['stillCarryingAtEnd'], 0)

    def test_rejects_incomplete_or_inconsistent_physical_history(self):
        original = trial()
        for corrupt in ('truncated', 'counts', 'cargo', 'unmatched'):
            value = copy.deepcopy(original)
            if corrupt == 'truncated':
                value['interactionEventsTruncated'] = True
            elif corrupt == 'counts':
                value['interactionCounts']['return'] = 2
            elif corrupt == 'cargo':
                value['interactionEvents'][2]['cargoBefore'] = 2
            else:
                value['interactionEvents'][2]['fly'] = 1
            with self.assertRaises(ValueError):
                diagnose_trial(value)

    def test_rejects_assistance_and_does_not_mutate_input(self):
        report = dict(teacher=False, learning=False, noise=False, injectedCargo=False,
                      deliveryResets=False, checkpointHash='fixed-candidate', fixedHash='fixed-model',
                      trials=[trial()])
        original = copy.deepcopy(report)
        self.assertTrue(diagnose_report(report)['shortReturnsAreNotNecessarilyWrong'])
        self.assertEqual(report, original)
        for flag in ('teacher', 'learning', 'noise', 'injectedCargo', 'deliveryResets'):
            with self.assertRaises(ValueError):
                diagnose_report({**report, flag: True})


if __name__ == '__main__':
    unittest.main()
