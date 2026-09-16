import copy
import unittest
import numpy as np
from .layout_correction_prefix import correction_specs, CONTEXTS, KINDS


def fixtures():
    episodes=[]
    for i,kind in enumerate(KINDS):
        n=200
        a={'observations':np.zeros((n,4,297),np.float32),
           'labels':np.full((n,4,3),.2,np.float32),
           'bodies':np.zeros((n+1,4,6),np.float32),
           'cooldowns':np.zeros((n+1,4),np.float64),
           'appliedActions':np.ones((n,4,3),np.float32)}
        a['labels'][80:140,:,2]=2.2
        a['bodies'][100:,:,5]=1
        a['observations'][:,:,126]=a['bodies'][:-1,:,5]
        episodes.append((a,{'seed':9370100+1000*i,'kind':kind,'frames':n}))
    return episodes


class CorrectionPrefixTests(unittest.TestCase):
    def test_real_condition_selection_no_input_or_label_change(self):
        episodes=fixtures(); original=copy.deepcopy(episodes)
        specs=correction_specs(episodes)
        self.assertEqual(len(specs),128)
        self.assertEqual(specs,correction_specs(episodes))
        self.assertEqual({(s['kind'],s['category']) for s in specs},
                         {(k,c) for k in KINDS for c in CONTEXTS})
        for s in specs:
            self.assertEqual(s['stop']-s['start'],96)
            self.assertTrue(s['lossStart']<=s['focusTick']<s['stop'])
            a=episodes[s['episode']][0]; t,f=s['focusTick'],s['fly']
            self.assertLessEqual(a['cooldowns'][t,f],.05)
            if s['category']=='pickup-needed':
                self.assertEqual(a['bodies'][t,f,5],0)
                self.assertGreater(a['labels'][t,f,2],1)
            if s['category']=='loaded-grip-needed':
                self.assertGreater(a['bodies'][t,f,5],0)
                self.assertGreater(a['labels'][t,f,2],1)
            if s['category']=='loaded-release':
                self.assertGreater(a['bodies'][t,f,5],0)
                self.assertLess(a['labels'][t,f,2],1)
                self.assertGreater(a['appliedActions'][t,f,2],.5)
        for (a,_),(b,_) in zip(episodes,original):
            for key in a: np.testing.assert_array_equal(a[key],b[key])

    def test_missing_context_cooldown_validation_seed_rejected(self):
        for count in (0,5,True):
            with self.assertRaises(ValueError): correction_specs(fixtures(),count)
        e=fixtures(); e[0][1]['seed']=10100000
        with self.assertRaises(ValueError): correction_specs(e)
        e=fixtures(); e[0][0]['cooldowns'][:]=1
        with self.assertRaises(ValueError): correction_specs(e)
        e=fixtures(); e[0][0]['appliedActions'][:,:,2]=0
        with self.assertRaises(ValueError): correction_specs(e)


if __name__=='__main__': unittest.main()
