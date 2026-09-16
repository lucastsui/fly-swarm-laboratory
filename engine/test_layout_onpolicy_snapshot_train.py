import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from .test_layout_pickup_preservation_flow import fixture
from .layout_pickup_preservation_probe import probe
from .layout_onpolicy_snapshot_train import apply_step, validate_probe
from .layout_guarded_snapshot_train import verify_saved_snapshot
from .layout_recovery_brain import save_model
from .layout_demonstration_step_probe import frozen_predictions


def case():
    brain,bank=fixture()
    bank.manifest['behaviorParameterHash']=brain.checkpoint_hash()
    with contextlib.redirect_stdout(io.StringIO()):published=probe(brain,bank)
    return brain,bank,published


class OnPolicySnapshotTests(unittest.TestCase):
    def test_real_recurrent_update_and_exact_save(self):
        brain,bank,published=case();before=brain.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()):r=apply_step(brain,bank,published)
        self.assertNotEqual(brain.checkpoint_hash(),before)
        self.assertEqual(brain.checkpoint_hash(),r['afterHash'])
        self.assertTrue(r['allOriginalTrainingGuardsPassed'])
        brain.interface='tiny-test';brain.status_scale=torch.tensor(.15)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'candidate.npz';save_model(path,brain);verify_saved_snapshot(path,brain)

    def test_previous_hash_need_not_be_identical(self):
        brain,bank,pub=case()
        next(t for t in pub['trials'] if t['allTrainingGuards'])['temporaryParameterHash']='prior-floating-point-run'
        with contextlib.redirect_stdout(io.StringIO()):r=apply_step(brain,bank,pub)
        self.assertFalse(r['crossRunByteIdentityRequired'])

    def test_wrong_behavior_or_no_guarded_proposal_rejected(self):
        brain,bank,pub=case();before=brain.checkpoint_hash()
        bank.manifest['behaviorParameterHash']='another-policy'
        with self.assertRaises(ValueError):apply_step(brain,bank,pub)
        bank.manifest['behaviorParameterHash']=before
        for t in pub['trials']:t['allTrainingGuards']=False
        with self.assertRaises(ValueError):apply_step(brain,bank,pub)
        self.assertEqual(brain.checkpoint_hash(),before)

    def test_forward_failure_after_change_rolls_back(self):
        brain,bank,pub=case();before=brain.checkpoint_hash();brain.eval()
        def fail(*args):
            if brain.checkpoint_hash()!=before:raise RuntimeError('injected failure after update')
            return frozen_predictions(*args)
        with patch('engine.layout_onpolicy_snapshot_train.frozen_predictions',side_effect=fail),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):apply_step(brain,bank,pub)
        self.assertEqual(brain.checkpoint_hash(),before);self.assertFalse(brain.training)

    def test_manifest_validation_without_optional_derivative_key(self):
        m={'inputFileHashes':{},'sourceHashes':{}}
        r={**m,'parametersRestored':True,'candidateSaved':False,'optimizerUsed':False,
           'sourceFilesUnchanged':True,'prefixExactAtInitialization':True,'gradientFrames':96}
        validate_probe(m,r)
        for key,value in [('parametersRestored',False),('candidateSaved',True),('optimizerUsed',True),
                          ('sourceFilesUnchanged',False),('gradientFrames',32),('exactReLUDerivative',False)]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(ValueError):validate_probe(m,bad)


if __name__=='__main__':unittest.main()
