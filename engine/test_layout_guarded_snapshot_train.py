"""Exact saved proposal tests; original numerical/physical gates stay required."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .test_layout_loaded_transition_train import case
from .layout_transition_reproduction_audit import observe_attempt
from .layout_guarded_snapshot_train import apply_snapshot_step,verify_saved_snapshot,validate_audit
from .layout_recovery_brain import save_model


class GuardedSnapshotTests(unittest.TestCase):
    def test_same_pass_snapshot_commits_and_serializes_exactly(self):
        b,bank,old,pub=case();before=b.checkpoint_hash()
        b.interface='tiny-test-interface';b.status_scale=torch.tensor(.15)
        with contextlib.redirect_stdout(io.StringIO()):r=apply_snapshot_step(b,bank,pub,old)
        self.assertNotEqual(before,b.checkpoint_hash());self.assertEqual(r['afterHash'],b.checkpoint_hash())
        self.assertTrue(r['allOriginalTrainingGuardsPassed']);self.assertFalse(r['crossRunByteIdentityRequired'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'candidate.npz';save_model(path,b);verify_saved_snapshot(path,b)
            with np.load(path,allow_pickle=False) as z:np.testing.assert_array_equal(z['gains'],b.log_gains.detach().numpy())

    def test_identity_only_rejection_does_not_discard_guarded_snapshot(self):
        b,bank,old,pub=case();before=b.checkpoint_hash()
        next(t for t in pub['trials'] if t['allTrainingGuards'])['temporaryParameterHash']='different-prior-test-identity'
        with contextlib.redirect_stdout(io.StringIO()):r=apply_snapshot_step(b,bank,pub,old)
        self.assertNotEqual(before,b.checkpoint_hash());self.assertTrue(r['snapshotBytesReverified'])
        self.assertFalse(r['originalAttemptOutcome']['returned'])

    def test_failed_snapshot_remeasurement_restores_weights(self):
        b,bank,old,pub=case();before=b.checkpoint_hash();b.eval()
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_guarded_snapshot_train.measure_bank',side_effect=RuntimeError('Injected readback error')):
            with self.assertRaises(RuntimeError):apply_snapshot_step(b,bank,pub,old)
        self.assertEqual(before,b.checkpoint_hash());self.assertFalse(b.training)

    def test_changed_snapshot_bytes_and_failed_original_guards_are_rejected(self):
        b,bank,old,pub=case();before=b.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()):report,arrays=observe_attempt(b,bank,pub,old)
        corrupt=[a.copy() for a in arrays];corrupt[0].flat[0]+=.00001
        with patch('engine.layout_guarded_snapshot_train.observe_attempt',return_value=(report,corrupt)):
            with self.assertRaises(ValueError):apply_snapshot_step(b,bank,pub,old)
        bad=copy.deepcopy(report);bad['allOriginalGuardsPassed']=False
        with patch('engine.layout_guarded_snapshot_train.observe_attempt',return_value=(bad,arrays)):
            with self.assertRaises(ValueError):apply_snapshot_step(b,bank,pub,old)
        self.assertEqual(before,b.checkpoint_hash())

    def test_changed_saved_array_is_rejected(self):
        b,bank,old,pub=case();b.interface='tiny-test-interface';b.status_scale=torch.tensor(.15)
        with contextlib.redirect_stdout(io.StringIO()):apply_snapshot_step(b,bank,pub,old)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'candidate.npz';save_model(path,b)
            with torch.no_grad():b.tonic.add_(.00001)
            with self.assertRaises(ValueError):verify_saved_snapshot(path,b)

    def test_audit_requires_two_unchanged_guarded_identity_failures(self):
        b,bank,old,pub=case()
        next(t for t in pub['trials'] if t['allTrainingGuards'])['temporaryParameterHash']='prior-test-hash'
        with contextlib.redirect_stdout(io.StringIO()):row,arrays=observe_attempt(b,bank,pub,old)
        rows=[copy.deepcopy(row),copy.deepcopy(row)]
        for i,r in enumerate(rows):r['attempt']=i+1
        rows[1]['comparedWithFirstAttempt']=[{'differentEntries':0,'maximumAbsoluteDifference':0.,'rmsDifference':0.}]*2
        m={'parentParameterHash':b.checkpoint_hash(),'fixedHash':b.fixed_hash,'originalGuardsAndReturnValuesUnchanged':True}
        result={'finished':True,'attempts':2,'parametersRestored':True,'sourcesAndInputsUnchanged':True,
                'candidateSaved':False,'optimizerUsed':False,'isServiceEvidence':False}
        validate_audit(m,result,rows,pub)
        bad=copy.deepcopy(rows);bad[1]['originalGuardChecks'][0]['passed']=False
        with self.assertRaises(ValueError):validate_audit(m,result,bad,pub)
        with self.assertRaises(ValueError):validate_audit(m,result,rows[:1],pub)


if __name__=='__main__':unittest.main()
