import contextlib
import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .layout_selective_jacobian_train import context_gate,diagnostic_gate,guarded_update,train
from .layout_operation_prefix import KINDS,CONTEXTS
from .test_layout_microfit_flow import TinyTrainBrain
from .layout_demonstration_step_probe import frozen_predictions
from .layout_operation_focus import focus_fit


def fixture():
    b=TinyTrainBrain().train();b.surrogate_training=False;b.fingerprint=lambda:b.fixed_hash
    specs=[dict(kind=k,category=c,start=20,stop=116,lossStart=84,focusTick=100) for k in KINDS for c in CONTEXTS]
    x=torch.ones((96,16,297));y=torch.zeros((96,16,3))
    motion=frozen_predictions(b,x,torch.zeros((4,16)),64)[...,:2].clone()
    for i,s in enumerate(specs):y[80,i,2]=.2 if s['category']==CONTEXTS[2] else 2.2
    bank=SimpleNamespace(manifest={'parameterHash':b.checkpoint_hash(),'windows':specs},
        groups={(s['kind'],s['category']):[i] for i,s in enumerate(specs)},
        batch=lambda ids,device:(x,y,torch.zeros((4,16)),motion),diagnostic_indexes=lambda:list(range(16)))
    fit=focus_fit(b,bank)
    return b,bank,fit


class SelectiveTrainingTests(unittest.TestCase):
    def test_context_gate_rejects_global_shift_and_motion_drift(self):
        old={'heads':[0.,.01,.02],'recall':.5,'falsePositiveRate':.75,'meanRawGripByContext':[.89,1.02,1.01,1.00]}
        good=copy.deepcopy(old);good['heads'][2]=.019;good['meanRawGripByContext']=[.90,1.025,1.00,1.005]
        self.assertTrue(context_gate(old,good,[0.,.01]))
        for failure in ('unwanted_return','pickup','motion','recall','fpr'):
            bad=copy.deepcopy(good)
            if failure=='unwanted_return':bad['meanRawGripByContext'][2]=1.011
            elif failure=='pickup':bad['meanRawGripByContext'][0]=.88
            elif failure=='motion':bad['heads'][1]=.0111
            elif failure=='recall':bad['recall']=.4
            else:bad['falsePositiveRate']=1.
            self.assertFalse(context_gate(old,bad,[0.,.01]),failure)

    def test_global_diagnostic_limits_cumulative_drift(self):
        initial={'objectiveHeads':[0.,0.,.02],'parentMotionMSE':[0.,0.],'focusRecall':.5,'focusFalsePositiveRate':.75}
        current=copy.deepcopy(initial);current['objectiveHeads'][2]=.015
        good=copy.deepcopy(current);good['objectiveHeads'][2]=.014;good['parentMotionMSE'][1]=.0009
        self.assertTrue(diagnostic_gate(initial,current,good))
        good['parentMotionMSE'][1]=.0011
        self.assertFalse(diagnostic_gate(initial,current,good))
        good['parentMotionMSE'][1]=0.;good['objectiveHeads'][2]=.016
        self.assertFalse(diagnostic_gate(initial,current,good))

    def test_real_graph_rejection_and_error_restore_parameters(self):
        b,bank,fit=fixture();before=b.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_selective_jacobian_train.context_gate',return_value=False):
            row=guarded_update(b,bank,bank,list(range(16)),fit,fit)
        self.assertFalse(row['accepted']);self.assertEqual(len(row['trials']),6)
        self.assertEqual(before,b.checkpoint_hash());self.assertTrue(all(p.grad is None for p in b.parameters()))
        def fail(*args):
            self.assertNotEqual(before,b.checkpoint_hash());raise RuntimeError('Injected trial failure')
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_selective_jacobian_train.frozen_predictions',side_effect=fail):
            with self.assertRaises(RuntimeError):guarded_update(b,bank,bank,list(range(16)),fit,fit)
        self.assertEqual(before,b.checkpoint_hash())

    def test_real_accepted_update_changes_only_parameters(self):
        b,bank,fit=fixture();before=b.checkpoint_hash();labels=bank.batch(None,None)[1].clone()
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_selective_jacobian_train.context_gate',return_value=True),\
             patch('engine.layout_selective_jacobian_train.focus_fit',return_value=fit):
            row=guarded_update(b,bank,bank,list(range(16)),fit,fit)
        self.assertTrue(row['accepted']);self.assertNotEqual(before,b.checkpoint_hash())
        torch.testing.assert_close(labels,bank.batch(None,None)[1],atol=0,rtol=0)
        self.assertEqual(row['beforeHash'],before);self.assertEqual(row['afterHash'],b.checkpoint_hash())
        self.assertTrue(all(p.grad is None for p in b.parameters()))

    def test_unmocked_proposal_and_gates_reduce_training_loss(self):
        b,bank,fit=fixture();before=b.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()):
            row=guarded_update(b,bank,bank,list(range(16)),fit,fit)
        self.assertTrue(row['accepted'])
        self.assertNotEqual(before,b.checkpoint_hash())
        self.assertLess(row['trainingFit']['objectiveHeads'][2],fit['objectiveHeads'][2])
        self.assertTrue(row['trials'][-1]['contextGatePassed'])

    def test_hard_bound_and_stop_after_first_rejection(self):
        b,bank,fit=fixture();seen=[];started=[]
        for bad in (0,9,1.5):
            with self.assertRaises(ValueError):train(b,bank,bank,np.random.default_rng(1),bad,lambda *v:None)
        with patch('engine.layout_selective_jacobian_train.focus_fit',return_value=fit),\
             patch('engine.layout_selective_jacobian_train.guarded_update',return_value={'accepted':False}):
            _,rows=train(b,bank,bank,np.random.default_rng(1),8,lambda i,r:seen.append(i),started.append)
        self.assertEqual(seen,[1]);self.assertEqual(len(rows),1);self.assertEqual(started,[fit])


if __name__=='__main__':unittest.main()
