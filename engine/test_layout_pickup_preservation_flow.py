"""Unmocked acceptance on an identifiable small recurrent test circuit."""
import contextlib
import io
import unittest
from types import SimpleNamespace
import torch
from .test_layout_microfit_flow import TinyTrainBrain
from .layout_operation_prefix import KINDS,CONTEXTS
from .layout_demonstration_step_probe import frozen_predictions
from .layout_pickup_preservation_probe import probe,LOADED_GRIP_BUDGET


class AdditiveFixtureBrain(TinyTrainBrain):
    def __init__(self):
        super().__init__();self.surrogate_training=False
        with torch.no_grad():self.tonic.fill_(.00001)

    def fingerprint(self):return self.fixed_hash

    def forward(self,x,steps,state=None,weights=None):
        if state is None:state=torch.zeros((4,len(x)))
        return None,.75*state+weights*x[:,:4].T+100*self.tonic


def fixture():
    b=AdditiveFixtureBrain().train()
    specs=[dict(kind=k,category=c,start=20,stop=116,lossStart=84,focusTick=100) for k in KINDS for c in CONTEXTS]
    x=torch.ones((96,16,297));y=torch.zeros((96,16,3));x[...,126:129]=0.
    for i,s in enumerate(specs):
        y[80,i,2]=.2 if s['category']==CONTEXTS[2] else 2.2
        if s['category']!=CONTEXTS[0]:x[:,i,126]=1.;x[:,i,3]=2.
    state=torch.zeros((4,16));motion=frozen_predictions(b,x,state,64)[...,:2].clone()
    arrays=(x,y,state,motion)
    bank=SimpleNamespace(manifest={'parameterHash':b.checkpoint_hash(),'windows':specs},
        groups={(s['kind'],s['category']):[i] for i,s in enumerate(specs)},
        batch=lambda ids,device:tuple(a[:,ids] for a in arrays),diagnostic_indexes=lambda:list(range(16)))
    return b,bank


class PickupFlowTests(unittest.TestCase):
    def test_real_proposal_and_guards_improve_pickup_without_loaded_regression(self):
        b,bank=fixture();before=b.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()):r=probe(b,bank)
        passing=[t for t in r['trials'] if t['allTrainingGuards']]
        self.assertTrue(passing);self.assertEqual(before,b.checkpoint_hash())
        t=passing[0]
        self.assertLess(t['measured']['pickupHinge'],.99*r['initial']['pickupHinge'])
        delta=max(abs(a-z) for a,z in zip(r['initial']['protectedLoadedGrip'],t['measured']['protectedLoadedGrip']))
        self.assertLessEqual(delta,LOADED_GRIP_BUDGET)
        self.assertNotEqual(t['temporaryParameterHash'],before)
        self.assertTrue(all(p.grad is None for p in b.parameters()))


if __name__=='__main__':unittest.main()
