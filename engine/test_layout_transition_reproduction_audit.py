"""Real tiny recurrent flow; the observer cannot bypass the original gate."""
import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from .test_layout_loaded_transition_train import case
from .layout_transition_reproduction_audit import observe_attempt
from . import layout_loaded_transition_train as training


class ReproductionAuditTests(unittest.TestCase):
    def test_valid_original_commit_is_observed_then_restored(self):
        b,bank,old,pub=case();before=b.checkpoint_hash();gate=training.preservation_gate
        with contextlib.redirect_stdout(io.StringIO()):r,arrays=observe_attempt(b,bank,pub,old)
        self.assertTrue(r['originalCommitReturned']);self.assertTrue(r['allOriginalGuardsPassed'])
        self.assertTrue(r['exactPublishedHashMatched']);self.assertEqual(b.checkpoint_hash(),before)
        self.assertIs(training.preservation_gate,gate);self.assertEqual(len(arrays),2)
        self.assertTrue(any(np.any(a!=p.detach().numpy()) for a,p in zip(arrays,(b.log_gains,b.tonic))))
        self.assertFalse(r['candidateSaved']);self.assertFalse(r['isServiceEvidence'])

    def test_hash_only_failure_keeps_both_original_guards_and_rollback(self):
        b,bank,old,pub=case();before=b.checkpoint_hash()
        next(t for t in pub['trials'] if t['allTrainingGuards'])['temporaryParameterHash']='deliberately-wrong-test-hash'
        with contextlib.redirect_stdout(io.StringIO()):r,arrays=observe_attempt(b,bank,pub,old)
        self.assertFalse(r['originalCommitReturned']);self.assertFalse(r['exactPublishedHashMatched'])
        self.assertTrue(r['allOriginalGuardsPassed']);self.assertIsNotNone(arrays)
        self.assertEqual(r['originalError']['type'],'ValueError');self.assertEqual(b.checkpoint_hash(),before)

    def test_original_guard_failure_still_blocks_commit(self):
        b,bank,old,pub=case();before=b.checkpoint_hash()
        # Test only: force a failing underlying gate; observer must return False.
        with contextlib.redirect_stdout(io.StringIO()),patch.object(training,'preservation_gate',return_value=False):
            r,arrays=observe_attempt(b,bank,pub,old)
        self.assertFalse(r['allOriginalGuardsPassed']);self.assertFalse(r['originalCommitReturned'])
        self.assertEqual(len(r['originalGuardChecks']),1);self.assertEqual(b.checkpoint_hash(),before)

    def test_exception_before_observation_restores_functions_and_model(self):
        b,bank,old,pub=case();before=b.checkpoint_hash();b.eval();pub['activeLoadedFrames']=[]
        gate=training.preservation_gate
        with contextlib.redirect_stdout(io.StringIO()):r,arrays=observe_attempt(b,bank,pub,old)
        self.assertIsNone(arrays);self.assertFalse(r['allOriginalGuardsPassed'])
        self.assertEqual(b.checkpoint_hash(),before);self.assertFalse(b.training);self.assertIs(training.preservation_gate,gate)


if __name__=='__main__':unittest.main()
