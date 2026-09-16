import copy
import unittest
import numpy as np
import torch
from .layout_operation_prefix import operation_specs, parent_motion, CONTEXTS, KINDS
from .test_layout_correction_prefix import fixtures
from .test_layout_microfit_flow import TinyTrainBrain
from .supervised_joint import raw_readout


def operations():
    episodes = fixtures()
    for a, m in episodes:
        a['appliedActions'][80:100, :, 2] = 0
        m['interactionEvents'] = []
        for fly in range(4):
            for tick, kind, on in ((110, 'transfer', True), (150, 'return', False), (170, 'return', True)):
                a['labels'][tick, fly, 2] = 2.2 if on else .2
                a['bodies'][tick+1, fly, 5] = 0
                m['interactionEvents'].append(dict(time=(tick+1)*.05, event=kind, fly=fly,
                                                   cargoBefore=1, cargoAfter=0,
                                                   nearestBox=1 if kind == 'transfer' else 0, boxDistance=.6))
    return episodes


class OperationPrefixTests(unittest.TestCase):
    def test_exact_event_selection_preserves_original_arrays(self):
        episodes = operations(); original = copy.deepcopy(episodes)
        specs = operation_specs(episodes)
        self.assertEqual(specs, operation_specs(episodes))
        self.assertEqual({(s['kind'], s['category']) for s in specs}, {(k,c) for k in KINDS for c in CONTEXTS})
        for s in specs:
            a, m = episodes[s['episode']]; t, f = s['focusTick'], s['fly']
            self.assertEqual(s['stop']-s['start'], 96)
            self.assertTrue(s['lossStart'] <= t < s['stop'])
            if s['category'] == CONTEXTS[0]:
                self.assertEqual(a['bodies'][t, f, 5], 0)
                self.assertEqual(a['appliedActions'][t, f, 2], 0)
            else:
                self.assertTrue(any(e['fly']==f and round(e['time']/.05)-1==t for e in m['interactionEvents']))
            self.assertEqual(bool(a['labels'][t, f, 2]>1), s['category'] != CONTEXTS[2])
        for (a,m), (b,n) in zip(episodes, original):
            self.assertEqual(m,n)
            for key in a: np.testing.assert_array_equal(a[key], b[key])

    def test_rejects_fabricated_returns_validation_seed_and_missing_context(self):
        for kind in ('cargo', 'box', 'source', 'missing'):
            e = operations()
            if kind == 'cargo': e[0][1]['interactionEvents'][1]['cargoBefore'] = 2
            elif kind == 'box': e[0][1]['interactionEvents'][1]['nearestBox'] = 2
            elif kind == 'source': e[0][1]['seed'] = 10100000
            else: e[0][1]['interactionEvents'] = []
            with self.assertRaises(ValueError): operation_specs(e)

    def test_parent_motion_has_independent_actor_columns_and_no_mutation(self):
        b = TinyTrainBrain().eval().requires_grad_(False)
        x = np.ones((3,96,1,297), np.float32); x[1] *= 2; x[2] *= 3
        states = np.zeros((3,4,1), np.float32); original=x.copy(); before=b.checkpoint_hash()
        result = parent_motion(b, x, states)
        self.assertEqual(result.shape, (3,32,1,2))
        for actor in range(3):
            state=torch.as_tensor(states[actor]); expected=[]
            for t in range(96):
                _,state=b(torch.as_tensor(x[actor,t]),4,state,b.weights())
                if t>=64: expected.append(raw_readout(b,state)[:,:2].numpy())
            np.testing.assert_array_equal(result[actor], np.asarray(expected))
        np.testing.assert_array_equal(x,original); self.assertEqual(b.checkpoint_hash(),before)


if __name__ == '__main__': unittest.main()
