import copy
import contextlib
import io
import unittest
from unittest.mock import patch
import torch
from .layout_decision_margin_probe import decision_gate,compare_rejected
from .layout_operation_prefix import CONTEXTS
from .test_layout_selective_jacobian_train import fixture
from .layout_context_step_probe import scores
from .layout_demonstration_step_probe import frozen_predictions


class DecisionMarginTests(unittest.TestCase):
    def test_full_recurrent_trials_and_error_restore(self):
        b,bank,fit=fixture();before=b.checkpoint_hash();ids=list(range(16));selections=bank.manifest['windows']
        x,y,state,motion=bank.batch(ids,None)
        initial=scores(frozen_predictions(b,x,state,64),y[64:],motion,selections)
        rejected={'beforeHash':before,'accepted':False,'windowIndexes':ids,'selections':selections,
                  'initialScores':initial,'referenceParentMotionMSE':[0.,0.],
                  'linearProposal':{'predictedUnboundedOutputChange':[0.]*6}}
        proposal=([torch.ones_like(b.log_gains)*.0001,torch.ones_like(b.tonic)*.00001],
                  {'predictedUnboundedOutputChange':[0.]*6})
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_decision_margin_probe.direction',return_value=proposal):
            result=compare_rejected(b,bank,bank,rejected,fit,fit)
        self.assertEqual(len(result['trials']),6);self.assertEqual(before,b.checkpoint_hash())
        self.assertTrue(result['parametersRestored']);self.assertTrue(all(p.grad is None for p in b.parameters()))
        def fail(*args):
            self.assertNotEqual(before,b.checkpoint_hash());raise RuntimeError('Injected temporary-forward failure')
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_decision_margin_probe.direction',return_value=proposal),\
             patch('engine.layout_decision_margin_probe.frozen_predictions',side_effect=fail):
            with self.assertRaises(RuntimeError):compare_rejected(b,bank,bank,rejected,fit,fit)
        self.assertEqual(before,b.checkpoint_hash())

    def fixture(self):
        selections=[{'category':c} for c in CONTEXTS]
        before={'rawGrip':[.88,1.032,1.038,1.013],'heads':[0.,.0046,.02],'recall':2/3,'falsePositiveRate':1.}
        after=copy.deepcopy(before);after['rawGrip']=[.89,1.031,1.037,1.012];after['heads'][2]=.019
        return before,after,selections

    def test_small_positive_margin_decrease_without_decision_change(self):
        b,a,s=self.fixture();self.assertTrue(decision_gate(b,a,[0.,.0046],s))

    def test_every_original_correct_decision_and_wrong_direction_protected(self):
        b,a,s=self.fixture()
        for index,bad_value in [(0,.879),(1,1.0),(1,1.009),(2,1.039),(3,1.009)]:
            altered=copy.deepcopy(a);altered['rawGrip'][index]=bad_value
            self.assertFalse(decision_gate(b,altered,[0.,.0046],s))
        b['rawGrip'][2]=.98;a['rawGrip'][2]=.989
        self.assertTrue(decision_gate(b,a,[0.,.0046],s))
        a['rawGrip'][2]=.991;self.assertFalse(decision_gate(b,a,[0.,.0046],s))

    def test_existing_low_confidence_and_numeric_validation(self):
        b,a,s=self.fixture();b['rawGrip'][1]=1.003;a['rawGrip'][1]=1.002
        self.assertFalse(decision_gate(b,a,[0.,.0046],s))
        a['rawGrip'][1]=float('nan')
        with self.assertRaises(ValueError):decision_gate(b,a,[0.,.0046],s)


if __name__=='__main__':unittest.main()
