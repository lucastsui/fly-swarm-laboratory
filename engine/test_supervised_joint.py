"""Teacher separation, mixed-task coverage and actual-world curriculum tests."""
import unittest
from unittest.mock import patch
import numpy as np
import torch

from .supervised_joint import curriculum_world,dataset,teacher_label,raw_readout,mirror_observations,transport_evaluation,TASKS


class JointTests(unittest.TestCase):
    def test_stationary_actor_cannot_pass_transport(self):
        class Stationary:
            def __init__(self,*args):pass
            def reset(self):pass
            def act(self,observations,explore=False):
                return [{'speed':0.,'turn':0.,'interact':True} for _ in observations]
        with patch('engine.supervised_joint.FrozenPolicy',Stationary),patch('engine.supervised_joint.teacher_label',side_effect=AssertionError('Teacher at evaluation')):
            report=transport_evaluation(None,[4810018,4810019],'approach',2,horizon=10)
        self.assertEqual(report['successes'],0);self.assertEqual(report['meanTravel'],0.)

    def test_mirror_matches_reflected_world(self):
        for kind in ('approach','carry','line'):
            w,target=curriculum_world(4810012,kind);x=w.sensory();label=teacher_label(w,target)
            w.y=14-w.y;w.heading=-w.heading;w.turn=-w.turn
            np.testing.assert_allclose(w.sensory(),mirror_observations(x[None])[0],atol=1e-7)
            expected=label.copy();expected[1]*=-1
            np.testing.assert_allclose(teacher_label(w,target),expected,atol=1e-7)

    def test_four_tasks_original_senses(self):
        x,y,t=dataset(4810001,32)
        self.assertEqual(x.shape,(128,30));self.assertEqual(y.shape,(128,3))
        self.assertTrue(np.isfinite(x).all());self.assertTrue(np.isfinite(y).all())
        np.testing.assert_array_equal(np.bincount(t),[32]*4)
        self.assertTrue((y[:,0]>0).any());self.assertTrue((y[:,1]<0).any());self.assertTrue((y[:,1]>0).any())
        self.assertTrue((y[:,2]>1).any());self.assertTrue((y[:,2]<1).any())

    def test_labels_do_not_mutate_observation(self):
        for kind in TASKS:
            w,target=curriculum_world(4810002,kind);before=w.sensory().copy()
            teacher_label(w,target)
            np.testing.assert_array_equal(before,w.sensory())
            self.assertEqual(len(before),30)

    def test_actual_pickup_requires_decoded_interaction(self):
        w,_=curriculum_world(4810003,'approach',bearing=0,distance=.5)
        w.advance({'speed':0.,'turn':0.,'interact':False});self.assertEqual(w.pickups,0)
        w.advance({'speed':0.,'turn':0.,'interact':True});self.assertEqual(w.pickups,1)

    def test_carry_requires_actual_drop(self):
        w,_=curriculum_world(4810004,'carry',bearing=0,distance=.5)
        self.assertEqual(w.transfers,0);self.assertEqual(w.cargo,1)
        w.advance({'speed':0.,'turn':0.,'interact':True});self.assertEqual(w.transfers,1)

    def test_loss_reaches_all_three_motor_heads(self):
        class Model:
            motor_forward=[0];motor_left=[1];motor_right=[2];motor_interact=[3]
        state=torch.full((4,4),.03,requires_grad=True)
        readout=raw_readout(Model(),state)
        target=torch.tensor([[1.,.5,2.2]]*4)
        (readout-target).square().sum().backward()
        self.assertTrue((state.grad.abs().sum(1)>0).all())


if __name__=='__main__':unittest.main()
