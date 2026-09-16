import contextlib
import copy
import io
import unittest
from unittest.mock import patch
import torch
from .layout_pickup_preservation_probe import measure,preservation_gate,probe
from .layout_operation_prefix import CONTEXTS
from .layout_demonstration_step_probe import frozen_predictions
from .test_layout_selective_jacobian_train import fixture as original_fixture


def fixture():
    brain,bank,_=original_fixture()
    x,_,_,_=bank.batch(None,None)
    x[...,126:129]=0.
    for i,s in enumerate(bank.manifest['windows']):
        if s['category']!=CONTEXTS[0]:x[:,i,126]=1.
    return brain,bank


class PickupPreservationTests(unittest.TestCase):
    def scores(self):
        before={'pickupRawGrip':[.8,1.05],'protectedLoadedGrip':[.9995,1.4],
                'parentMotionMSE':[0.,0.],'rawGrip':[.8,1.05,.9995,1.4],
                'loadedMask':[[False,False,True,True]],'pickupHinge':.04625}
        after=copy.deepcopy(before);after['pickupRawGrip']=[.81,1.051]
        after['pickupHinge']=.04325;after['protectedLoadedGrip']=[.9996,1.4001]
        return before,after

    def test_guard_accepts_pickups_with_protected_handling(self):
        before,after=self.scores();self.assertTrue(preservation_gate(before,after))

    def test_guard_rejects_threshold_crossing_even_with_tiny_loaded_change(self):
        before,after=self.scores();after['protectedLoadedGrip'][0]=1.0001
        self.assertFalse(preservation_gate(before,after))

    def test_guard_rejects_drift_regression_changed_mask_and_nonfinite_values(self):
        before,after=self.scores()
        for key,value in [('protectedLoadedGrip',[.9996,1.403]),('parentMotionMSE',[0.,.000251]),
                          ('pickupRawGrip',[.79,1.06]),('pickupHinge',.0463)]:
            bad=copy.deepcopy(after);bad[key]=value;self.assertFalse(preservation_gate(before,bad))
        after['loadedMask']=[[True]]
        with self.assertRaises(ValueError):preservation_gate(before,after)
        _,after=self.scores();after['protectedLoadedGrip'][0]=float('nan')
        with self.assertRaises(ValueError):preservation_gate(before,after)

    def test_original_cargo_selects_all_loaded_frames_not_just_focus(self):
        b,bank=fixture();x,y,state,motion=bank.batch(None,None);spec=bank.manifest['windows']
        p=frozen_predictions(b,x,state,64);m=measure(p,y[64:],x[64:],motion,spec)
        self.assertEqual(len(m['protectedLoadedGrip']),32*12)
        self.assertEqual(len(m['pickupRawGrip']),4)
        broken=x[64:].clone();broken[16,0,126]=1.
        with self.assertRaises(ValueError):measure(p,y[64:],broken,motion,spec)
        broken=x[64:].clone();broken[0,0,126]=.1
        with self.assertRaises(ValueError):measure(p,y[64:],broken,motion,spec)

    def test_unmocked18_output_full_recurrent_probe_always_restores(self):
        b,bank=fixture();before=b.checkpoint_hash();labels=bank.batch(None,None)[1].clone()
        with contextlib.redirect_stdout(io.StringIO()):r=probe(b,bank)
        self.assertEqual(before,b.checkpoint_hash());self.assertEqual(len(r['trials']),6)
        self.assertEqual(len(r['linearProposal']['requestedOutputChange']),18)
        for i,s in enumerate(r['selections']):
            if s['category']!=CONTEXTS[0]:self.assertEqual(r['linearProposal']['requestedOutputChange'][i],0.)
        self.assertTrue(r['parametersRestored']);self.assertFalse(r['candidateSaved'])
        torch.testing.assert_close(labels,bank.batch(None,None)[1],atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in b.parameters()))

    def test_exception_after_actual_perturbation_restores_parameters_and_mode(self):
        b,bank=fixture();b.eval();before=b.checkpoint_hash()
        original=frozen_predictions
        def fail_after_change(*args):
            if b.checkpoint_hash()!=before:raise RuntimeError('Injected temporary-forward failure')
            return original(*args)
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_pickup_preservation_probe.frozen_predictions',side_effect=fail_after_change):
            with self.assertRaises(RuntimeError):probe(b,bank)
        self.assertEqual(before,b.checkpoint_hash());self.assertFalse(b.training)


if __name__=='__main__':unittest.main()
