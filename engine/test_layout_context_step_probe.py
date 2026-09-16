import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .layout_context_step_probe import direction,trust_scale,rollback_probe
from .layout_operation_prefix import CONTEXTS,KINDS
from .test_layout_microfit_flow import TinyTrainBrain


class StepProbeTests(unittest.TestCase):
    def test_independent_linear_output_changes(self):
        rows=[[torch.tensor([1.,0.]),torch.tensor([0.])],[torch.tensor([0.,1.]),torch.tensor([0.])]]
        delta,m=direction(rows,[.2,-.1],(1.,1.),1e-6)
        np.testing.assert_allclose(delta[0].numpy(),[.2,-.1],rtol=2e-6)
        np.testing.assert_allclose(m['predictedUnboundedOutputChange'],delta[0].numpy(),rtol=1e-6)
        scale=trust_scale(delta,(.01,.001))
        self.assertLessEqual(float((scale*delta[0]).abs().max()),.010001)

    def test_identical_outputs_cannot_take_opposite_steps(self):
        rows=[[torch.ones(2),torch.ones(1)]]*2
        delta,_=direction(rows,[.1,-.1],(1.,1.))
        self.assertLess(max(float(d.abs().max()) for d in delta),1e-5)
        with self.assertRaises(ValueError): direction(rows,[.1],(1.,1.))
        with self.assertRaises(ValueError): trust_scale(delta,(0.,1.))

    def test_full_recurrent_trials_and_exception_restore(self):
        b=TinyTrainBrain().train(); b.surrogate_training=False; b.fingerprint=lambda:b.fixed_hash
        before=b.checkpoint_hash()
        specs=[dict(kind=k,category=c,start=20,stop=116,lossStart=84,focusTick=100) for k in KINDS for c in CONTEXTS]
        x=torch.ones((96,16,297));y=torch.zeros((96,16,3))
        for i,s in enumerate(specs): y[80,i,2]=.2 if s['category']==CONTEXTS[2] else 2.2
        bank=SimpleNamespace(manifest={'parameterHash':before,'windows':specs},
            groups={(s['kind'],s['category']):[i] for i,s in enumerate(specs)},
            batch=lambda ids,device:(x,y,torch.zeros((4,16)),torch.zeros((32,16,2))))
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_context_step_probe.focus_fit',return_value={}):
            result=rollback_probe(b,bank)
        self.assertEqual(len(result['trials']),9)
        self.assertEqual(before,b.checkpoint_hash());self.assertTrue(b.training)
        self.assertTrue(all(p.grad is None for p in b.parameters()))
        def fail_after_change(*args):
            self.assertNotEqual(before,b.checkpoint_hash())
            raise RuntimeError('Injected probe error')
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_context_step_probe.frozen_predictions',side_effect=fail_after_change):
            with self.assertRaises(RuntimeError): rollback_probe(b,bank)
        self.assertEqual(before,b.checkpoint_hash());self.assertTrue(b.training)


if __name__=='__main__':unittest.main()
