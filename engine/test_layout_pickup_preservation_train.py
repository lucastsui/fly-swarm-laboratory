import contextlib
import copy
import io
import unittest
from unittest.mock import patch
import torch
from .test_layout_pickup_preservation_flow import fixture
from .layout_pickup_preservation_probe import probe
from .layout_pickup_preservation_train import apply_verified_step
from .layout_demonstration_step_probe import frozen_predictions


class SinglePickupStepTests(unittest.TestCase):
    def setup_case(self):
        b,bank=fixture()
        with contextlib.redirect_stdout(io.StringIO()):r=probe(b,bank)
        return b,bank,r

    def test_real_accepted_step_exactly_reproduces_original_guarded_candidate(self):
        b,bank,r=self.setup_case();before=b.checkpoint_hash();labels=bank.batch(list(range(16)),None)[1].clone()
        chosen=next(t for t in r['trials'] if t['allTrainingGuards'])
        with contextlib.redirect_stdout(io.StringIO()):row=apply_verified_step(b,bank,r)
        self.assertNotEqual(before,b.checkpoint_hash());self.assertEqual(b.checkpoint_hash(),chosen['temporaryParameterHash'])
        self.assertTrue(row['selectedProposalReproducedExactly']);self.assertEqual(row['beforeHash'],before)
        self.assertFalse(row['isServiceEvidence']);self.assertTrue(all(p.grad is None for p in b.parameters()))
        torch.testing.assert_close(labels,bank.batch(list(range(16)),None)[1],atol=0,rtol=0)

    def test_no_passing_probe_never_updates(self):
        b,bank,r=self.setup_case();before=b.checkpoint_hash()
        for t in r['trials']:t['allTrainingGuards']=False
        with self.assertRaises(ValueError):apply_verified_step(b,bank,r)
        self.assertEqual(before,b.checkpoint_hash())

    def test_wrong_published_candidate_restores_actual_temporary_change(self):
        b,bank,r=self.setup_case();before=b.checkpoint_hash()
        next(t for t in r['trials'] if t['allTrainingGuards'])['temporaryParameterHash']='altered'
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):apply_verified_step(b,bank,r)
        self.assertEqual(before,b.checkpoint_hash())

    def test_exception_after_perturbation_restores_model_and_mode(self):
        b,bank,r=self.setup_case();b.eval();before=b.checkpoint_hash()
        def fail(*args):
            if b.checkpoint_hash()!=before:raise RuntimeError('Injected temporary-forward failure')
            return frozen_predictions(*args)
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_pickup_preservation_train.frozen_predictions',side_effect=fail):
            with self.assertRaises(RuntimeError):apply_verified_step(b,bank,r)
        self.assertEqual(before,b.checkpoint_hash());self.assertFalse(b.training)

    def test_changed_initial_scores_or_selections_are_rejected(self):
        b,bank,r=self.setup_case();before=b.checkpoint_hash()
        altered=copy.deepcopy(r);altered['initial']['pickupRawGrip'][0]+=1.
        with self.assertRaises(AssertionError):apply_verified_step(b,bank,altered)
        self.assertEqual(before,b.checkpoint_hash())
        r['windowIndexes'][0]=r['windowIndexes'][1]
        with self.assertRaises(ValueError):apply_verified_step(b,bank,r)
        self.assertEqual(before,b.checkpoint_hash())


if __name__=='__main__':unittest.main()
