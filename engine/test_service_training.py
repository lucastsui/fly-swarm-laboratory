import unittest
from unittest.mock import patch

import numpy as np
import torch

from .haul_world import HaulWorld
from .service_training import SequenceBatch,service_label,service_target,training_world,mask_reset_state,service_evaluation,development_rank


class ServiceTests(unittest.TestCase):
    def test_selection_uses_whole_line_counts_not_latest_update(self):
        def result(cleared,products,early):
            return {'checkpoints':{'40':{'episodesWithProduct':early},'120':{'episodesAllThree':cleared,'products':products}}}
        self.assertGreater(development_rank(result(7,37,12)),development_rank(result(4,28,13)))
        self.assertGreater(development_rank(result(7,38,11)),development_rank(result(7,37,12)))

    def test_natural_recovery_never_starts_preloaded_final_cargo(self):
        batch=SequenceBatch(16,5109917,natural_recovery=True)
        self.assertEqual(batch.kinds.count('line'),12);self.assertNotIn('final',batch.kinds)
        self.assertTrue(all(w.cargo==0 for w,k in zip(batch.worlds,batch.kinds) if k=='line'))

    def test_snapshot_heading_alias_requires_history(self):
        a=HaulWorld(5109918);b=HaulWorld(5109918)
        a.x=b.x=10.;a.y=b.y=7.;a.heading=0.;b.heading=np.pi;a.cargo=b.cargo=3
        np.testing.assert_allclose(a.sensory(),b.sensory(),atol=1e-7)
        self.assertGreater(float(np.linalg.norm(service_label(a)-service_label(b))),1.)

    def test_final_supervisor_and_inputs(self):
        w=training_world(5109911,'final');before=w.sensory().copy()
        self.assertIs(service_target(w),w.stations[3]);self.assertEqual(w.cargo,3)
        self.assertEqual(len(w.landmarks),4);self.assertEqual(len(before),30)
        label=service_label(w);np.testing.assert_array_equal(before,w.sensory())
        self.assertTrue(np.isfinite(label).all())

    def test_waits_for_nearby_processing(self):
        w=HaulWorld(5109912);w.x=w.stations[1]['x'];w.y=7.;w.stations[1]['timer']=1.
        self.assertIs(service_target(w),w.stations[1])

    def test_student_not_teacher_controls_training_world(self):
        batch=SequenceBatch(16,5109913);positions=[(w.x,w.y) for w in batch.worlds]
        batch.observations_and_labels();batch.advance(np.zeros((16,3),np.float32))
        self.assertEqual(positions,[(w.x,w.y) for w in batch.worlds])
        self.assertTrue(all(w.pickups==w.transfers==0 for w in batch.worlds))

    def test_reset_erases_state_and_gradient_only_for_reset_fly(self):
        state=torch.ones((5,3),requires_grad=True);masked=mask_reset_state(state,[False,True,False])
        self.assertTrue((masked[:,1]==0).all());masked.sum().backward()
        self.assertTrue((state.grad[:,1]==0).all());self.assertTrue((state.grad[:,0]==1).all())

    def test_frozen_service_has_no_teacher_or_automatic_material_handling(self):
        class Model:
            log_gains=torch.zeros(3)
        class Stationary:
            def __init__(self,*args):self.gains=np.zeros(3,np.float32)
            def reset(self):pass
            def act(self,observations,explore=False):
                return [{'speed':0.,'turn':0.,'interact':False} for _ in observations]
        with patch('engine.service_training.FrozenPolicy',Stationary),patch('engine.service_training.service_target',side_effect=AssertionError('Teacher during evaluation')):
            r=service_evaluation(Model(),[5109914,5109915],2,800,replenish=True)
        self.assertEqual(r['checkpoints']['40']['products'],0);self.assertEqual(r['checkpoints']['40']['pickups'],0)


if __name__=='__main__':unittest.main()
