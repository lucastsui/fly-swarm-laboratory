import math
import unittest
import numpy as np
from .swarm_world import SwarmWorld,collide
from .haul_world import HaulWorld

STOP={'speed':0.,'turn':0.,'interact':False}

class SwarmTests(unittest.TestCase):
    def test_head_on_and_swept_tunnelling(self):
        p,pairs,_,held=collide([[9,7],[11,7]],[[20,0],[-20,0]],dt=.1)
        self.assertIn((0,1),pairs);self.assertFalse(held)
        self.assertGreaterEqual(p[1,0]-p[0,0],.44-1e-8)

    def test_dense_chain_at_wall(self):
        p=np.array([[.22+i*.44,7.] for i in range(4)])
        for _ in range(100):
            p,_,_,_=collide(p,[[-2,0]]*4)
            self.assertGreaterEqual(p[:,0].min(),.22-1e-7)
            self.assertGreaterEqual(np.diff(p[:,0]).min(),.44-1e-7)

    def test_stations_collisionless(self):
        w=SwarmWorld(2,flies=1);a=w.agents[0]
        a.x,a.y,a.heading=w.stations[0]['x']-.3,7.,0.
        for _ in range(6):w.advance([{'speed':2.,'turn':0.,'interact':False}])
        self.assertAlmostEqual(a.x,w.stations[0]['x']+.3)
        self.assertFalse(a.contact)

    def test_shared_clock_and_single_item(self):
        w=SwarmWorld(3,flies=4,continuous=False)
        for a in w.agents:a.x,a.y,a.heading=w.stations[0]['x']-.3,7.,0.
        # Transaction test with ghosts, independent of intentionally overlapping placement.
        w.collisions_enabled=False;w.stations[0]['stock']=1;w.stations[1]['timer']=1.
        w.advance([{**STOP,'interact':True}]*4)
        self.assertEqual(sum(a.cargo==1 for a in w.agents),1)
        self.assertEqual(w.stations[0]['stock'],0)
        self.assertAlmostEqual(w.stations[1]['timer'],.95)

    def test_spawn_and_peer_senses(self):
        for seed in range(20):
            w=SwarmWorld(seed)
            for i,a in enumerate(w.agents):
                for b in w.agents[:i]:self.assertGreater(math.hypot(a.x-b.x,a.y-b.y),.44)
            self.assertEqual(w.sensory().shape,(4,30))
        a=w.agents[0];with_peers=a.sensory();a.landmarks=a.stations
        self.assertTrue(np.any(with_peers[:24]!=a.sensory()[:24]))

    def test_continuing_shared_factory(self):
        w=SwarmWorld(1,horizon=1);w.agents[0].deliveries=10
        _,terminal=w.advance([STOP]*4)
        self.assertFalse(terminal);self.assertEqual(w.deliveries,10)
        self.assertEqual(w.stations[0]['stock'],3)

    def test_one_fly_matches_original_away_from_walls(self):
        one=HaulWorld(25,continuous=True);swarm=SwarmWorld(25,flies=1)
        for step in range(100):
            motor={'speed':.5,'turn':.4,'interact':step%20==0}
            reward,_=one.advance(motor);rewards,_=swarm.advance([motor])
            np.testing.assert_allclose(one.sensory(),swarm.sensory()[0],atol=1e-7)
            self.assertAlmostEqual(reward,rewards[0])
            self.assertEqual(one.stations,swarm.stations)

if __name__=='__main__':unittest.main()
