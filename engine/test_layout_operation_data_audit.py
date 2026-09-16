import unittest
import numpy as np
from .layout_operation_data_audit import eligible_counts
from .layout_operation_prefix import CONTEXTS


class OperationAuditTests(unittest.TestCase):
    def fixture(self):
        n=120
        a={'bodies':np.zeros((n+1,4,6)),'labels':np.zeros((n,4,3)),
           'appliedActions':np.zeros((n,4,3)),'cooldowns':np.zeros((n+1,4))}
        return a,{'frames':n,'interactionEvents':[]}

    def test_original_pickup_boundary_cargo_actions_and_cooldown(self):
        a,m=self.fixture();a['labels'][[79,80,103,104],0,2]=2.2
        self.assertEqual(eligible_counts(a,m)[CONTEXTS[0]],2)
        a['cooldowns'][80,0]=.06;a['bodies'][103,0,5]=1
        self.assertEqual(eligible_counts(a,m)[CONTEXTS[0]],0)

    def test_actual_event_types_and_labels_not_assumed_success(self):
        a,m=self.fixture();a['bodies'][...,5]=1
        for tick,event,on in [(79,'return',True),(80,'return',True),(81,'return',False),(82,'transfer',True),(83,'delivery',True),(84,'transfer',False)]:
            m['interactionEvents'].append({'time':(tick+1)*.05,'fly':1,'event':event})
            a['labels'][tick,1,2]=2.2 if on else .2
        self.assertEqual(eligible_counts(a,m),dict(zip(CONTEXTS,[0,2,1,1])))
        m['interactionEvents'].append({'time':0,'fly':1,'event':'return'})
        with self.assertRaises(ValueError):eligible_counts(a,m)


if __name__=='__main__':unittest.main()
