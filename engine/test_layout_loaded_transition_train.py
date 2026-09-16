"""Tiny recurrent20-row commit/rollback flow, with synthetic rejection metadata."""
import contextlib
import copy
import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .test_layout_pickup_preservation_flow import fixture
from .layout_pickup_preservation_probe import probe as pickup_probe
from .layout_loaded_transition_probe import probe as transition_probe
from .layout_loaded_transition_train import apply_verified_step
from .layout_demonstration_step_probe import frozen_predictions


def case():
    b,bank=fixture()
    with contextlib.redirect_stdout(io.StringIO()):old=pickup_probe(b,bank)
    # Synthetic rejected trial selects two real loaded positions of this tiny
    # test circuit. Production checks the immutable original full-brain report.
    for row in old['trials']:row['allTrainingGuards']=False
    old['trials'][0]['measured']['protectedLoadedGrip']=list(old['initial']['protectedLoadedGrip'])
    for i in (0,1):
        old['trials'][0]['measured']['protectedLoadedGrip'][i]=old['initial']['protectedLoadedGrip'][i]+.006
    with contextlib.redirect_stdout(io.StringIO()):published=transition_probe(b,bank,old)
    return b,bank,old,published


class LoadedTransitionTrainingTests(unittest.TestCase):
    def test_unmocked20_row_update_reproduces_guarded_candidate(self):
        b,bank,old,pub=case();before=b.checkpoint_hash()
        selected=next(t for t in pub['trials'] if t['allTrainingGuards'])
        labels=bank.batch(list(range(16)),None)[1].clone()
        with contextlib.redirect_stdout(io.StringIO()):r=apply_verified_step(b,bank,pub,old)
        self.assertNotEqual(before,b.checkpoint_hash());self.assertEqual(b.checkpoint_hash(),selected['temporaryParameterHash'])
        self.assertEqual(len(r['activeLoadedFrames']),2);self.assertEqual(len(r['linearProposal']['requestedOutputChange']),20)
        self.assertTrue(r['selectedProposalReproducedExactly']);self.assertFalse(r['isServiceEvidence'])
        torch.testing.assert_close(labels,bank.batch(list(range(16)),None)[1],atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in b.parameters()))

    def test_changed_transition_rows_and_absent_pass_never_commit(self):
        b,bank,old,pub=case();before=b.checkpoint_hash();bad=copy.deepcopy(pub)
        bad['activeLoadedFrames'][0]['scoredFrame']+=1
        with self.assertRaises(ValueError):apply_verified_step(b,bank,bad,old)
        for t in pub['trials']:t['allTrainingGuards']=False
        with self.assertRaises(ValueError):apply_verified_step(b,bank,pub,old)
        self.assertEqual(before,b.checkpoint_hash())

    def test_wrong_published_candidate_rolls_back_real_temporary_change(self):
        b,bank,old,pub=case();before=b.checkpoint_hash()
        next(t for t in pub['trials'] if t['allTrainingGuards'])['temporaryParameterHash']='altered'
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):apply_verified_step(b,bank,pub,old)
        self.assertEqual(before,b.checkpoint_hash())

    def test_failure_after_perturbation_restores_original_mode_and_weights(self):
        b,bank,old,pub=case();b.eval();before=b.checkpoint_hash()
        def fail(*args):
            if b.checkpoint_hash()!=before:raise RuntimeError('Injected temporary-forward failure')
            return frozen_predictions(*args)
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_loaded_transition_train.frozen_predictions',side_effect=fail):
            with self.assertRaises(RuntimeError):apply_verified_step(b,bank,pub,old)
        self.assertEqual(before,b.checkpoint_hash());self.assertFalse(b.training)


if __name__=='__main__':unittest.main()
