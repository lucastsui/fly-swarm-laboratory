"""Tests of physical task rules, real plasticity, and shared update safety."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .haul_world import HaulWorld
from .haul_protocol import encode,decode
from .haul_server import Learner
from .plastic_brain import PlasticBrain,SCHEMA

ROOT=Path(__file__).resolve().parents[1]/'.runtime/dopamine-haul'

class WorldTests(unittest.TestCase):
    def test_zero_output_and_no_auto_interaction(self):
        w=HaulWorld(); start=(w.x,w.y,w.heading)
        for _ in range(20): w.advance({'speed':0.,'turn':0.,'interact':False})
        self.assertEqual(start,(w.x,w.y,w.heading)); self.assertEqual(w.pickups,0)
    def test_all_three_material_transfers_required(self):
        w=HaulWorld(); motor={'speed':0.,'turn':0.,'interact':True}
        for index in range(3):
            w.x,w.y=w.stations[index]['x'],7.; w.cooldown=0
            w.advance(motor); self.assertEqual(w.cargo,index+1)
            w.x=w.stations[index+1]['x']; w.cooldown=0; w.advance(motor)
            for _ in range(40): w.advance({**motor,'interact':False})
        self.assertEqual(w.transfers,3); self.assertEqual(w.deliveries,1)
    def test_wrong_station_cannot_deliver(self):
        w=HaulWorld(); w.cargo=1; w.x=w.stations[3]['x']; w.y=7
        w.advance({'speed':0.,'turn':0.,'interact':True})
        self.assertEqual(w.cargo,1); self.assertEqual(w.deliveries,0)
    def test_observation_has_only_fixed_thirty_channels(self):
        w=HaulWorld(); obs=w.sensory()
        self.assertEqual(obs.shape,(30,)); self.assertTrue(np.isfinite(obs).all())

class ProtocolTests(unittest.TestCase):
    def test_round_trip(self):
        meta,arr=decode(encode({'a':3},delta=np.ones(3,np.float32)))
        self.assertEqual(meta,{'a':3}); np.testing.assert_array_equal(arr['delta'],1)
    def test_duplicate_stale_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'brain-info.json').write_text(json.dumps({'plasticSynapses':3,'plasticMaskHash':'mask'}))
            l=Learner(root); meta={'schema':SCHEMA,'mask':'mask','worker':'laptop','updateId':'one','version':0}
            arr={'delta':np.ones(3,np.float32)*.01}
            self.assertTrue(l.submit(meta,arr)['accepted']); first=l.gains.copy()
            self.assertTrue(l.submit(meta,arr)['duplicate']); np.testing.assert_array_equal(first,l.gains)
            self.assertFalse(l.submit({**meta,'updateId':'stale','version':-100},arr)['accepted'])
            with self.assertRaises(ValueError): l.submit({**meta,'updateId':'nan'},{'delta':np.full(3,np.nan,np.float32)})
            l.save(); restored=Learner(root); np.testing.assert_array_equal(l.gains,restored.gains)

@unittest.skipUnless(torch.cuda.is_available() and ROOT.exists(),'Requires prepared CUDA graph')
class PlasticityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.b=PlasticBrain(ROOT,batch=2)
    def test_reward_and_eligibility_are_both_required(self):
        b=self.b; b.reset(); b.proposal.zero_()
        b.reinforce([1.,-1.]); self.assertEqual(float(b.proposal.abs().sum()),0.)
        b.reset()
        for _ in range(8): b.act([HaulWorld(i).sensory() for i in range(2)])
        b.reinforce([0.,0.]); self.assertEqual(float(b.proposal.abs().sum()),0.)
        b.reinforce([1.,.5]); self.assertGreater(float(b.proposal.abs().sum()),0.)
    def test_only_selected_real_synapses_change(self):
        b=self.b; gains=np.linspace(-.01,.01,len(b.gains),dtype=np.float32)
        b.load_gains(gains); result=b.audit()
        self.assertGreater(result['changedSynapses'],0); self.assertTrue(result['frozenSynapsesUnchanged'])
        self.assertTrue(result['signsPreserved']); self.assertTrue(result['sharedWeightsMatch'])
        self.assertFalse(result['backpropagation']); self.assertFalse(b.wiring.requires_grad)

if __name__=='__main__': unittest.main()
