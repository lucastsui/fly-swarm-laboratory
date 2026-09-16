import unittest
from unittest.mock import patch
import torch
from .layout_history_guarded_train import guarded_attempt


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.log_gains=torch.nn.Parameter(torch.zeros(2));self.tonic=torch.nn.Parameter(torch.zeros(2))
    def checkpoint_hash(self):return repr([p.detach().tolist() for p in self.parameters()])


class GuardedTrainingTests(unittest.TestCase):
    def proposal(self,m,*args,**kwargs):
        with torch.no_grad():m.log_gains.add_(.001);m.tonic.add_(.00001)
        return {'accepted':True,'initialFullBank':{'baseline':True}}
    def test_additional_rejection_restores_provisional_update(self):
        m=Model();old=m.checkpoint_hash()
        with patch('engine.layout_history_guarded_train.attempt',side_effect=self.proposal), patch(
                'engine.layout_history_guarded_train.candidate_history_guard',return_value={'gate':{'passed':False}}):
            row=guarded_attempt(m,None,[])
        self.assertFalse(row['accepted']);self.assertEqual(m.checkpoint_hash(),old);self.assertEqual(row['afterHash'],old)
    def test_full_history_acceptance_retains_exact_single_proposal(self):
        m=Model();old=m.checkpoint_hash()
        with patch('engine.layout_history_guarded_train.attempt',side_effect=self.proposal) as proposal, patch(
                'engine.layout_history_guarded_train.candidate_history_guard',return_value={'gate':{'passed':True}}) as guard:
            row=guarded_attempt(m,None,[])
        self.assertTrue(row['accepted']);self.assertNotEqual(m.checkpoint_hash(),old)
        self.assertEqual(row['afterHash'],m.checkpoint_hash());self.assertEqual(proposal.call_count,1);self.assertEqual(guard.call_count,1)
    def test_replay_or_callback_exception_rolls_back(self):
        for callback in (False,True):
            m=Model();old=m.checkpoint_hash()
            def fail(value):raise RuntimeError('injected callback failure')
            with patch('engine.layout_history_guarded_train.attempt',side_effect=self.proposal), patch(
                    'engine.layout_history_guarded_train.candidate_history_guard',side_effect=RuntimeError('injected replay failure')):
                with self.assertRaises(RuntimeError):guarded_attempt(m,None,[],on_provisional=fail if callback else lambda v:None)
            self.assertEqual(m.checkpoint_hash(),old)
    def test_no_replay_after_original_gate_rejects(self):
        m=Model()
        with patch('engine.layout_history_guarded_train.attempt',return_value={'accepted':False}), patch(
                'engine.layout_history_guarded_train.candidate_history_guard') as guard:
            row=guarded_attempt(m,None,[])
        self.assertFalse(row['accepted']);guard.assert_not_called()


if __name__=='__main__':unittest.main()
