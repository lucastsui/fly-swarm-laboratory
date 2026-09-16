import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from .haul_world import HaulWorld
from .swarm_world import SwarmWorld
from .service_viewer import ServiceViewer,next_tick_deadline,viewer_interface


class ContinuousTests(unittest.TestCase):
    def test_experimental_interface_cannot_silently_load_as_legacy(self):
        self.assertEqual(viewer_interface({}),'original-30')
        self.assertEqual(viewer_interface({'sensoryInterface':'local-color-cargo-v3'}),'local-color-cargo-v3')
        for value in ('annotated-color-cargo-v1','unknown',None):
            with self.assertRaises(ValueError):viewer_interface({'sensoryInterface':value})

    def test_threefold_pacing_changes_wall_time_not_physics(self):
        deadline=0.
        for _ in range(60):deadline=next_tick_deadline(deadline,deadline+.004,3.)
        self.assertAlmostEqual(deadline,1.)
        self.assertEqual(next_tick_deadline(0.,.1,3.),.1)
        self.assertAlmostEqual(next_tick_deadline(0.,0.,1.),.05)
        with self.assertRaises(ValueError):next_tick_deadline(0.,0.,0.)

    def test_completion_and_time_limit_do_not_end_continuous_world(self):
        world=HaulWorld(1,horizon=2,continuous=True)
        world.x=world.stations[3]['x'];world.y=7.;world.cargo=3;world.deliveries=2
        _,done=world.advance({'speed':0.,'turn':0.,'interact':True})
        self.assertFalse(done);self.assertEqual(world.deliveries,3)
        world.stations[1]['timer']=.8
        position=(world.x,world.y,world.heading)
        for _ in range(3):
            _,done=world.advance({'speed':0.,'turn':0.,'interact':False})
            self.assertFalse(done)
        self.assertEqual(position,(world.x,world.y,world.heading))
        self.assertGreater(world.stations[1]['timer'],0)
        self.assertEqual(world.deliveries,3);self.assertEqual(world.steps,4)

    def test_replenishment_never_automatically_picks_up_or_delivers(self):
        world=HaulWorld(1,continuous=True);world.stations[0]['stock']=0
        for _ in range(10):world.advance({'speed':0.,'turn':0.,'interact':False})
        self.assertEqual(world.stations[0]['stock'],3)
        self.assertEqual((world.pickups,world.transfers,world.deliveries,world.cargo),(0,0,0,0))

    def test_finite_mode_keeps_original_terminal_rules(self):
        world=HaulWorld(1,horizon=1)
        _,done=world.advance({'speed':0.,'turn':0.,'interact':False})
        self.assertTrue(done)

    def test_viewer_preserves_world_and_neural_state_after_third_delivery(self):
        viewer=ServiceViewer.__new__(ServiceViewer)
        viewer.world=SwarmWorld(1,flies=1);world=viewer.world
        agent=world.agents[0]
        agent.x=world.stations[3]['x'];agent.y=7.;agent.cargo=3;agent.deliveries=2
        neural_state=torch.ones((4,1))
        viewer.policy=SimpleNamespace(state=neural_state,reset=Mock(side_effect=AssertionError('No neural reset')),
            act=Mock(return_value=[{'speed':.1,'turn':.1,'interact':True}]))
        viewer.model=SimpleNamespace(**{'motor_'+name:torch.tensor([i]) for i,name in enumerate(('forward','left','right','interact'))})
        viewer.ticks=4800;viewer.pickups=viewer.transfers=0;viewer.products=2;viewer.reward=0.
        viewer.history=deque();viewer.motion_samples=deque(maxlen=32);viewer.capture=Mock()
        for _ in range(40):viewer.tick()
        self.assertIs(viewer.world,world);self.assertIs(viewer.policy.state,neural_state)
        viewer.policy.reset.assert_not_called()
        self.assertEqual(viewer.products,3);self.assertEqual(viewer.ticks,4840)
        self.assertEqual(len(viewer.motion_samples),32)
        self.assertGreater(agent.distance,0);self.assertEqual(world.stations[0]['stock'],3)


if __name__=='__main__':unittest.main()
