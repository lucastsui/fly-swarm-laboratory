"""Regression guards for the preserved pickup result; no training is performed."""
from pathlib import Path
import unittest
import numpy as np
import torch
from .plastic_brain import PlasticBrain,digest
from .conditioning_experiment import evaluate

ROOT=Path(__file__).resolve().parents[1]/'.runtime/dopamine-haul'
CANDIDATE=ROOT.parent/'dopamine-experiments/selected-pickup.npz'

class RuleTests(unittest.TestCase):
    def test_signed_gain_reinforces_depolarization_for_both_signs(self):
        base=np.array([.1,-.1]); gain=.05*np.sign(base)
        self.assertTrue(np.all(base*np.exp(gain)>base))
        np.testing.assert_array_equal(np.sign(base*np.exp(gain)),np.sign(base))

@unittest.skipUnless(torch.cuda.is_available() and CANDIDATE.exists(),'Requires preserved CUDA experiment')
class PreservedPickupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.b=PlasticBrain(ROOT,16,303)
        cls.gains=np.load(CANDIDATE,allow_pickle=False)['gains']
        cls.b.load_gains(cls.gains)
    def test_only_existing_interaction_afferents_changed(self):
        mask=np.isin(self.b.spec['post'],self.b.spec['motor_interact'])
        self.assertEqual(int(mask.sum()),415)
        self.assertEqual(int(np.count_nonzero(self.gains)),415)
        self.assertFalse(np.any(self.gains[~mask]))
    def test_retained_frozen_pickup_and_sensory_ablation(self):
        before=digest(self.b.gains)
        normal=evaluate(self.b,40,start=1500000,cases=32)
        blank=evaluate(self.b,40,start=1500000,cases=32,blank=True)
        self.assertEqual(normal['nearPickupRate'],1.)
        self.assertEqual(normal['farAttemptRate'],0.)
        self.assertEqual(blank['nearPickupRate'],0.)
        self.assertEqual(before,digest(self.b.gains))

if __name__=='__main__': unittest.main()
