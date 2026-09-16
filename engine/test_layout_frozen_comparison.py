import copy
import unittest
from .layout_frozen_comparison import attest, compare
from .layout_recovery_eval import wilson
from .layout_recovery_world import INTERFACE, PHYSICS
from .test_layout_service_diagnostics import trial


def fixture():
    t = trial()
    t.update(pickups=2, transfers=1, returns=1, deliveryTimes=[],
             initialPositions=[[i, i] for i in range(4)], finalPositions=[[i, i] for i in range(4)],
             initialAvatars=[[i, i, 0] for i in range(4)], trace=[{'time': 20, 'products': 0}])
    m = dict(checkpointHash='candidate', fixedHash='fixed', seconds=20, random_starts=False,
             move_at=0, original_senses=False, migrate=False, interaction_events=True,
             kinds='wide', seed=1, cases=1, sourceHash='source')
    r = dict(checkpointHash='candidate', fixedHash='fixed', seconds=20, randomStarts=False,
             moveAtSeconds=0, interface=INTERFACE, physics=PHYSICS, batchActors=4, device='gpu',
             teacher=False, learning=False, noise=False, injectedCargo=False, deliveryResets=False,
             interactionEventRecording=True, trials=[t], summary={'wide': dict(cases=1, products=0,
             worldsWithProduct=0, worldsWithThree=0, sustainedSuccesses=0, wilson95=wilson(0, 1))})
    return r, m


class FrozenComparisonTests(unittest.TestCase):
    def test_complete_matched_comparison_does_not_mutate(self):
        r, m = fixture(); saved = copy.deepcopy(r)
        result = compare(r, m, copy.deepcopy(r), copy.deepcopy(m), 'candidate')
        self.assertTrue(result['matchedControlsAndFullEventAccountingPassed'])
        self.assertEqual(result['gainedWorlds'], 0)
        self.assertEqual(r, saved)

    def test_rejects_assistance_early_stop_and_fabricated_score(self):
        for key in ('teacher', 'learning', 'noise', 'injectedCargo', 'deliveryResets', 'horizon', 'score', 'event', 'worlds'):
            r, m = fixture()
            if key == 'horizon': r['trials'][0]['trace'][-1]['time'] = 15
            elif key == 'score': r['summary']['wide']['sustainedSuccesses'] = 1
            elif key == 'event': r['trials'][0]['transfers'] = 2
            elif key == 'worlds': r['trials'].append(copy.deepcopy(r['trials'][0]))
            else: r[key] = True
            with self.assertRaises(ValueError, msg=key): attest(r, m, 'candidate')

    def test_rejects_different_device_source_or_initial_world(self):
        for key in ('device', 'source', 'position', 'avatar', 'hash'):
            r, m = fixture(); b, bm = fixture()
            if key == 'device': b['device'] = 'other'
            elif key == 'source': bm['sourceHash'] = 'other'
            elif key == 'position':
                b['trials'][0]['initialPositions'][0][0] = 7
                b['trials'][0]['finalPositions'][0][0] = 7
            elif key == 'avatar': b['trials'][0]['initialAvatars'][0][0] = 7
            else: r['checkpointHash'] = 'other'
            with self.assertRaises(ValueError, msg=key): compare(r, m, b, bm, 'candidate')


if __name__ == '__main__': unittest.main()
