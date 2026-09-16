"""Bounded tests of the untrained, local-only embodiment contract."""
import unittest
import numpy as np
import torch
from .plane import Plane, EmbodiedBrain, GRAPH


class PlaneTests(unittest.TestCase):
    def test_zero_motor_output_does_not_move_avatar(self):
        world = Plane()
        before = (world.x,world.y,world.heading)
        for _ in range(40):
            world.advance({'speed':0.,'turn':0.,'interact':False})
        np.testing.assert_allclose(before,(world.x,world.y,world.heading),atol=1e-14)
        self.assertEqual(world.distance,0.)

    def test_remote_landmarks_supply_no_local_signal(self):
        world = Plane()
        world.landmarks = [{'x':100,'y':100,'radius':1}]
        obs = world.sensory()
        np.testing.assert_allclose(obs[:24],.025)
        self.assertEqual(len(obs),30)

    def test_interaction_requires_proximity(self):
        world = Plane()
        world.advance({'speed':0.,'turn':0.,'interact':True})
        self.assertFalse(world.cargo)
        world.x,world.y = world.material['x'],world.material['y']
        world.cooldown = 0
        world.advance({'speed':0.,'turn':0.,'interact':True})
        self.assertTrue(world.cargo)


@unittest.skipUnless(torch.cuda.is_available() and all((GRAPH/name).is_file() for name in
    ('metadata.json', 'graph.npz', 'annotations.feather', 'neurotransmitters.feather')),
    'Requires local CUDA and prepared source tables for full-graph validation')
class FullGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = EmbodiedBrain()

    def test_no_training_and_reproducible_initial_trajectory(self):
        brain = self.brain
        paths = []
        for _ in range(2):
            brain.state.zero_()
            world = Plane()
            for step in range(100):
                world.advance(brain.forward(world.sensory()))
            paths.append((world.x,world.y,world.heading))
        np.testing.assert_allclose(paths[0],paths[1],atol=1e-6)
        self.assertEqual(brain.n,166700)
        self.assertEqual(brain.wiring._nnz(),25582938)
        self.assertTrue(brain.audit()['unchanged'])
        self.assertFalse(brain.state.requires_grad)
        self.assertFalse(brain.wiring.requires_grad)
        self.assertEqual(brain.updates,0)
        self.assertEqual(brain.reward_pulses,0)

    def test_reward_cannot_inject_dopamine(self):
        drive = self.brain.reinforcement_drive(100)
        self.assertEqual(float(drive.abs().sum()),0.)
        self.assertEqual(self.brain.reward_pulses,0)

    def test_sensory_input_changes_neural_output(self):
        brain = self.brain
        outputs=[]
        for observation in (np.zeros(30,np.float32),np.ones(30,np.float32)):
            brain.state.zero_()
            for _ in range(50):
                result=brain.forward(observation)
            outputs.append(result['rates'])
        self.assertTrue(any(abs(outputs[0][key]-outputs[1][key])>1e-7 for key in outputs[0]))


if __name__ == '__main__':
    unittest.main()
