import copy
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .layout_candidate_history_guard import candidate_history_guard, validate_histories
from .layout_contextual_operation_probe import measure_bank
from .layout_operation_sampling import OperationCache
from .test_layout_microfit_flow import TinyTrainBrain
from .test_layout_operation_training import OperationTrainingTests


class Model(torch.nn.Module):
    n = 3
    def checkpoint_hash(self): return 'candidate'
    def fingerprint(self): return 'fixed'


class HistoryGuardTests(unittest.TestCase):
    def test_real_prefix_and_measurement_on_trainable_recurrent_model(self):
        class Brain(TinyTrainBrain):
            def fingerprint(self):return self.fixed_hash
        brain=Brain().train()
        with tempfile.TemporaryDirectory() as folder:
            episodes,data,dataset,meta=OperationTrainingTests().make_cache(Path(folder),brain)
            with patch('engine.layout_operation_sampling.load_histories',return_value=(episodes,data)):
                bank=OperationCache(Path(folder),dataset,brain.checkpoint_hash(),brain.fixed_hash,4)
            original=measure_bank(brain,bank,list(range(len(bank.manifest['windows']))))
            with torch.no_grad():brain.log_gains.add_(.00001)
            before=brain.checkpoint_hash()
            with contextlib.redirect_stdout(io.StringIO()):
                report=candidate_history_guard(brain,bank,episodes,original)
            self.assertTrue(report['candidateFullHistoryPrefix'])
            self.assertEqual(brain.checkpoint_hash(),before)
            self.assertTrue(brain.training)
            self.assertTrue(all(p.requires_grad for p in brain.parameters()))

    def fixture(self):
        x = np.zeros((96,4,297),np.float32)
        y = np.zeros((96,4,3),np.float32)
        w = {'episode':0,'seed':100,'kind':'wide','fly':2,'start':0,'stop':96}
        bank=SimpleNamespace(manifest={'parameterHash':'parent','windows':[w]},
             x=x[:,2:3][None].copy(),y=y[:,2:3][None].copy(),states=np.zeros((1,3,1),np.float32),
             motion=np.ones((1,32,1,2),np.float32))
        return Model(),bank,[({'observations':x,'labels':y},{'seed':100,'kind':'wide'})]

    def test_uses_candidate_states_and_original_motion_without_mutating_bank(self):
        model,bank,episodes=self.fixture(); model.train()
        before=copy.deepcopy(bank); state=np.arange(12,dtype=np.float32).reshape(1,3,4)
        def measure(m,b,ids):
            np.testing.assert_array_equal(b.states, state[:,:,2:3])
            self.assertIs(b.motion, bank.motion)
            self.assertIs(b.x, bank.x); self.assertFalse(m.training)
            return {'measured':True}
        with patch('engine.layout_candidate_history_guard.fill_states',return_value=state), patch(
                'engine.layout_candidate_history_guard.measure_bank',side_effect=measure), patch(
                'engine.layout_candidate_history_guard.margin_gate',return_value={'passed':False}) as gate:
            report=candidate_history_guard(model,bank,episodes,{'original':True})
        gate.assert_called_once_with({'original':True},{'measured':True},0.)
        self.assertFalse(report['gate']['passed']); self.assertFalse(report['isServiceEvidence'])
        np.testing.assert_array_equal(bank.states,before.states)
        self.assertTrue(model.training)

    def test_changed_observations_or_labels_rejected(self):
        for name in ('observations','labels'):
            _,bank,episodes=self.fixture(); episodes[0][0][name][10,2,0]=1
            with self.assertRaises(ValueError):validate_histories(bank,episodes)

    def test_wrong_identity_rejected(self):
        _,bank,episodes=self.fixture(); episodes[0][1]['seed']=101
        with self.assertRaises(ValueError):validate_histories(bank,episodes)

    def test_failed_replay_restores_model_mode(self):
        model,bank,episodes=self.fixture(); model.train()
        with patch('engine.layout_candidate_history_guard.fill_states',side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):candidate_history_guard(model,bank,episodes,{})
        self.assertTrue(model.training)

    def test_unchanged_parent_or_invalid_prefix_rejected(self):
        model,bank,episodes=self.fixture(); bank.manifest['parameterHash']='candidate'
        with self.assertRaises(ValueError):candidate_history_guard(model,bank,episodes,{})
        bank.manifest['parameterHash']='parent'
        with patch('engine.layout_candidate_history_guard.fill_states',return_value=np.full((1,3,4),np.nan,np.float32)):
            with self.assertRaises(ValueError):candidate_history_guard(model,bank,episodes,{})


if __name__ == '__main__':unittest.main()
