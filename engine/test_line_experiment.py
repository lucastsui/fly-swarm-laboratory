"""CPU tests: curriculum is initialization, not an action oracle."""
import unittest
import torch
import numpy as np
from .line_experiment import line_world,advance_reward,odd_perturbation,presynaptic_contrast,RewardBaseline,SignedBrain


class LineCurriculumTests(unittest.TestCase):
    def test_reward_error_is_causal_signed_and_replica_local(self):
        baseline=RewardBaseline(2,alpha=.5)
        reward=np.array([1.,-1.],np.float32);before=reward.copy()
        np.testing.assert_array_equal(baseline.advance(reward),[1.,-1.])
        np.testing.assert_array_equal(reward,before)
        np.testing.assert_array_equal(baseline.advance(reward),[.5,-.5])
        np.testing.assert_array_equal(baseline.advance([0.,0.]),[-.75,.75])
        baseline.reset([0]);np.testing.assert_array_equal(baseline.mean,[0.,-.375])
        baseline.reset();self.assertFalse(baseline.mean.any())
        with self.assertRaises(ValueError):baseline.advance([np.nan,0.])

    def test_coherent_exploration_holds_ten_steps_and_preserves_fixed_weights(self):
        # Exercise the actual sampler on CPU, without loading the full graph.
        brain=object.__new__(SignedBrain);brain.state=torch.zeros((5,3))
        brain.exploration_sigma=torch.ones((5,1))
        brain.generator=torch.Generator().manual_seed(1)
        brain.held_noise=torch.zeros_like(brain.state);brain.exploration_age=0
        brain.t={'motor_forward':torch.tensor([0]),'motor_interact':torch.tensor([1])}
        first=brain.exploration('differential').clone()
        for _ in range(9):torch.testing.assert_close(brain.exploration('differential'),first)
        self.assertFalse(torch.equal(brain.exploration('differential'),first))
        torch.testing.assert_close(brain.state,torch.zeros_like(brain.state))
        torch.testing.assert_close(brain.exploration_sigma,torch.ones_like(brain.exploration_sigma))
    def test_presynaptic_contrast_is_training_only_centered_and_bounded(self):
        activity=torch.tensor([[1.,2.,3.],[5.,5.,5.]])
        before=activity.clone();contrast=presynaptic_contrast(activity)
        torch.testing.assert_close(activity,before)
        torch.testing.assert_close(contrast.mean(1),torch.zeros(2))
        self.assertEqual(float(contrast[1].abs().sum()),0.)
        self.assertLessEqual(float(contrast.abs().max()),3.)
        with self.assertRaises(ValueError):presynaptic_contrast(activity[:,:1])
    def test_antithetic_response_removes_rectification_mean(self):
        current=torch.tensor([-.01,0.,.01,.5])
        noise=torch.tensor([.02,.02,.02,.02])
        positive=odd_perturbation(current,noise)
        negative=odd_perturbation(current,-noise)
        torch.testing.assert_close(positive,-negative)
        self.assertGreater(float(positive[1]),0.)
        # The older clean-baseline estimator is not zero-mean at the threshold.
        old=(torch.tanh(torch.relu(noise[1]))+torch.tanh(torch.relu(-noise[1])))/2
        self.assertGreater(float(old),0.)
    def test_stationary_action_does_not_haul_or_receive_progress_reward(self):
        for stage in ('aligned','mixed','wide','full'):
            world=line_world(3400000,400,stage)
            before=(world.x,world.y,world.cargo)
            reward,_=advance_reward(world,{'speed':0.,'turn':0.,'interact':False})
            self.assertEqual((world.x,world.y,world.cargo),before)
            self.assertEqual((world.pickups,world.transfers,world.deliveries),(0,0,0))
            self.assertAlmostEqual(reward,-.0002)

    def test_pickup_gets_event_reward_not_goal_switch_reward(self):
        world=line_world(3400001,400,'aligned')
        reward,_=advance_reward(world,{'speed':0.,'turn':0.,'interact':True})
        self.assertEqual(world.pickups,1)
        self.assertEqual(world.cargo,1)
        self.assertAlmostEqual(reward,.1-.0002)

    def test_mixed_initial_states_cover_every_leg_without_precredited_events(self):
        cargos=set()
        for seed in range(3400000,3400100):
            world=line_world(seed,400,'mixed');cargos.add(world.cargo)
            self.assertEqual((world.pickups,world.transfers,world.deliveries),(0,0,0))
            self.assertEqual(len(world.sensory()),30)
        self.assertEqual(cargos,{0,1,2,3})


if __name__=='__main__':unittest.main()
