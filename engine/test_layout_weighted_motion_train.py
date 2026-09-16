import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from .layout_weighted_motion_train import window_outputs, active_rows, attempt, guarded_attempt, validate_rejection
from .layout_operation_prefix import KINDS, CONTEXTS
from .layout_contextual_operation_probe import measure
from .layout_margin_projected_train import margin_gate
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import save_model
from .layout_guarded_snapshot_train import verify_saved_snapshot
from .test_layout_margin_projected_train import MarginProjectedTests
from .test_layout_contextual_operation_probe import measured_pair
from .test_layout_history_guarded_train import Model

MODULE = 'engine.layout_weighted_motion_train.'


class WindowOutputTests(unittest.TestCase):
    def inputs(self):
        p = torch.arange(32*16*3,dtype=torch.float32).reshape(32,16,3).requires_grad_()
        selections = [{'kind':k,'category':c} for k in KINDS for c in CONTEXTS]
        return p,p[16,:,2],selections

    def test_rows_are_ordered_and_never_average_across_actors(self):
        p,grip,s = self.inputs()
        outputs,residual,rows = window_outputs(p,grip,s)
        self.assertEqual(len(outputs),48); self.assertEqual(residual[16:],[0.]*32)
        for i in range(16):
            self.assertEqual(outputs[i],grip[i])
            for head in range(2):
                j = 16+2*i+head
                self.assertEqual(outputs[j],p[:,i,head].mean())
                self.assertEqual(rows[j]['selectedWindow'],i); self.assertEqual(rows[j]['head'],head)
                g, = torch.autograd.grad(outputs[j],p,retain_graph=True)
                expected = torch.zeros_like(p); expected[:,i,head] = 1/32
                torch.testing.assert_close(g,expected,atol=0,rtol=0)

    def test_opposite_actor_changes_do_not_cancel_in_new_rows(self):
        p,grip,s = self.inputs(); q = p.detach().clone()
        q[:,0,1] += 2; q[:,1,1] -= 2
        a,_,_ = window_outputs(p,grip,s); b,_,_ = window_outputs(q,q[16,:,2],s)
        self.assertEqual(p[...,1].mean(),q[...,1].mean())
        self.assertEqual(float((b[17]-a[17]).detach()),2.); self.assertEqual(float((b[19]-a[19]).detach()),-2.)

    def test_only_exact_zero_motion_rows_with_zero_target_omitted(self):
        zero = [torch.zeros(2),torch.zeros(1)]
        tiny = [torch.tensor([1e-12,0.]),torch.zeros(1)]
        gradients,residual,report = active_rows([tiny,zero],[1.,0.],[{'kind':'grip-focus'},{'kind':'window-motion-mean'}])
        self.assertEqual(len(gradients),1); self.assertEqual(residual,[1.])
        self.assertEqual(report['activeRowIndexes'],[0]); self.assertEqual(report['omittedExactZeroMotionRows'],[1])
        for target,kind in ((1.,'window-motion-mean'),(0.,'grip-focus'),(1.,'grip-focus')):
            with self.assertRaises(ValueError):active_rows([zero],[target],[{'kind':kind}])

    def test_invalid_shapes_order_or_nonfinite_values_fail_closed(self):
        p,g,s = self.inputs()
        with self.assertRaises(ValueError):window_outputs(p[:31],g,s)
        with self.assertRaises(ValueError):window_outputs(p,g,s[::-1])
        bad = p.detach().clone(); bad[0,0,1] = float('nan')
        with self.assertRaises(ValueError):window_outputs(bad,g,s)
        with self.assertRaises(ValueError):active_rows([[torch.tensor([float('nan')]),torch.zeros(1)]],[0.],[{'kind':'window-motion-mean'}])


class WindowTrainingTests(unittest.TestCase):
    def test_unmocked_recurrence_has_48_rows_and_exact_rejection_rollback(self):
        model,bank = MarginProjectedTests().bank(); before = model.checkpoint_hash()
        labels = bank.batch(None,None)[1].clone()
        with contextlib.redirect_stdout(io.StringIO()):row = attempt(model,bank,solver_iterations=64)
        self.assertFalse(row['accepted']); self.assertEqual(model.checkpoint_hash(),before)
        constraints = row['outputConstraints']
        self.assertEqual(len(constraints['rows']),48)
        self.assertEqual(constraints['omittedExactZeroMotionRows'],[])
        self.assertEqual(len(row['boundedLinearSolve']['requestedOutputChange']),48)
        self.assertEqual(row['boundedLinearSolve']['squaredResidualWeights'],[1.]*16+[16.]*32)
        self.assertEqual(len(row['trials']),6); self.assertTrue(all(p.grad is None for p in model.parameters()))
        torch.testing.assert_close(labels,bank.batch(None,None)[1],atol=0,rtol=0)

    def test_full_bank_rejection_cannot_be_bypassed(self):
        model,bank = MarginProjectedTests().bank(); before = model.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()), patch(MODULE+'margin_gate',side_effect=[{'passed':True},{'passed':False}]*6):
            row = attempt(model,bank,solver_iterations=64)
        self.assertFalse(row['accepted']); self.assertEqual(model.checkpoint_hash(),before)
        self.assertTrue(all(t['fullBank'] is not None for t in row['trials']))

    def test_accepted_snapshot_is_one_exact_tested_update(self):
        model,bank = MarginProjectedTests().bank(); before = model.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()), patch(MODULE+'margin_gate',return_value={'passed':True}):
            row = attempt(model,bank,solver_iterations=64)
        self.assertTrue(row['accepted']); self.assertEqual(len(row['trials']),1)
        self.assertNotEqual(model.checkpoint_hash(),before); self.assertEqual(model.checkpoint_hash(),row['afterHash'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'candidate.npz'; save_model(path,model); verify_saved_snapshot(path,model)
            with self.assertRaises(FileExistsError):save_model(path,model)

    def test_exception_after_changed_forward_restores_weights_and_mode(self):
        model,bank = MarginProjectedTests().bank(); model.eval(); before = model.checkpoint_hash()
        def fail(*args):
            if model.checkpoint_hash() != before:raise RuntimeError('injected changed-forward failure')
            return frozen_predictions(*args)
        with contextlib.redirect_stdout(io.StringIO()), patch(MODULE+'frozen_predictions',side_effect=fail):
            with self.assertRaises(RuntimeError):attempt(model,bank,solver_iterations=64)
        self.assertEqual(model.checkpoint_hash(),before); self.assertFalse(model.training)


class FullHistoryTests(unittest.TestCase):
    def proposal(self,model,*args,**kwargs):
        with torch.no_grad():model.log_gains.add_(.001); model.tonic.add_(.00001)
        return {'accepted':True,'initialFullBank':{'baseline':True}}

    def test_history_rejection_rolls_back_both_parameter_blocks(self):
        model = Model(); before = model.checkpoint_hash()
        with patch(MODULE+'attempt',side_effect=self.proposal), patch(MODULE+'candidate_history_guard',return_value={'gate':{'passed':False}}):
            row = guarded_attempt(model,None,[])
        self.assertFalse(row['accepted']); self.assertEqual(model.checkpoint_hash(),before); self.assertEqual(row['afterHash'],before)

    def test_history_acceptance_retains_only_one_proposal(self):
        model = Model(); before = model.checkpoint_hash()
        with patch(MODULE+'attempt',side_effect=self.proposal) as proposal, patch(MODULE+'candidate_history_guard',return_value={'gate':{'passed':True}}) as guard:
            row = guarded_attempt(model,None,[])
        self.assertTrue(row['accepted']); self.assertEqual(row['afterHash'],model.checkpoint_hash())
        self.assertNotEqual(before,model.checkpoint_hash()); self.assertEqual(proposal.call_count,1); self.assertEqual(guard.call_count,1)

    def test_replay_and_callback_exceptions_both_roll_back(self):
        for callback in (False,True):
            model = Model(); before = model.checkpoint_hash()
            def fail(value):raise RuntimeError('injected callback failure')
            with patch(MODULE+'attempt',side_effect=self.proposal), patch(MODULE+'candidate_history_guard',side_effect=RuntimeError('injected replay failure')):
                with self.assertRaises(RuntimeError):guarded_attempt(model,None,[],on_provisional=fail if callback else lambda v:None)
            self.assertEqual(model.checkpoint_hash(),before)

    def test_rejected_proposal_never_enters_history_check(self):
        model = Model()
        with patch(MODULE+'attempt',return_value={'accepted':False}), patch(MODULE+'candidate_history_guard') as guard:
            row = guarded_attempt(model,None,[])
        self.assertFalse(row['accepted']); guard.assert_not_called()


class PriorEvidenceTests(unittest.TestCase):
    def docs(self):
        p,q,y,x,m,s = measured_pair()
        before,after = [measure(z,y,x,m,s) for z in (p,q)]
        q = q.clone(); q[0,0,1] = 1.
        own = measure(q,y,x,m,s)
        trial = {'measured':after,'fullBank':after,'selectedGate':margin_gate(before,after),
                 'fullGate':margin_gate(before,after,0.),'accepted':True,'parameterHash':'proposed'}
        proposal = {'beforeHash':'parent','afterHash':'proposed','accepted':True,'initial':before,
                    'initialFullBank':before,'trials':[trial]}
        history = {'candidateParameterHash':'proposed','parentParameterHash':'parent',
                   'candidateFullHistoryPrefix':True,'originalMotionTargetsRetained':True,
                   'originalGuardThresholdsUnchanged':True,'optimizerUsed':False,'isServiceEvidence':False,
                   'measurement':own,'gate':margin_gate(before,own,0.)}
        return {'manifest':{'parentParameterHash':'parent'},'initial-all-window-measure':before,'provisional-step':proposal,
                'step-1':{'proposal':proposal,'accepted':False,'beforeHash':'parent','afterHash':'parent','candidateHistory':history},
                'result':{'algorithm':'one-pooled-window-motion-update-with-full-history-gate-v1','acceptedUpdates':0,
                          'parametersRestored':True,'candidateSaved':False,'sourceFilesUnchanged':True,
                          'isServiceEvidence':False,'finalParameterHash':'parent'},
                'status':{'finished':True,'acceptedUpdates':0,'parameterHash':'parent'}}

    def test_consistent_history_motion_rejection_is_required(self):
        docs = self.docs(); validate_rejection(docs,'parent')
        mutations = [('status','finished',False),('result','candidateSaved',True),('result','parametersRestored',False),
                     ('step-1','afterHash','proposed'),('result','acceptedUpdates',1)]
        for name,key,value in mutations:
            bad = copy.deepcopy(docs); bad[name][key] = value
            with self.assertRaises(ValueError):validate_rejection(bad,'parent')

    def test_prior_measurement_or_gate_tampering_rejected(self):
        for name in ('gate','measurement'):
            docs = self.docs(); history = docs['step-1']['candidateHistory']
            if name == 'gate':history['gate']['parentMotionWithinBudget'] = True
            else:history['measurement']['parentMotionMSE'] = [0.,0.]
            with self.assertRaises(ValueError):validate_rejection(docs,'parent')


if __name__ == '__main__':unittest.main()
