import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .fixed_layout_curriculum import load_layout, fixed_training_world, transition_training_world, SCHEMA
from .layout_recovery_world import PHYSICS, RecoveryWorld
from . import test_layout_recovery_protocol as protocol_tests
from .layout_recovery_teacher import LocalTeacher
from .layout_recovery_protocol import Exchange

POSITIONS = [[18.747888990904322,7.3994367959900025],[5.917935543911416,11.178116458349923],
             [14.240717861912344,9.973804726144254],[4.232500768799213,6.855373503855216]]


class FixedLayoutTests(unittest.TestCase):
    def spec(self, root):
        path = Path(root)/'layout.json'
        path.write_text(json.dumps({'schema':SCHEMA,'physics':PHYSICS,'flies':4,'collisions':True,
                                    'stationOrder':['raw','smelter','assembler','finished'],'positions':POSITIONS}))
        return path

    def test_layout_identity_and_invalid_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.spec(temp)
            positions, digest = load_layout(path)
            self.assertEqual(digest, load_layout(path)[1])
            np.testing.assert_array_equal(positions, POSITIONS)
            value=json.loads(path.read_text());value['positions'][1]=value['positions'][0]
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): load_layout(path)

    def test_training_episodes_preserve_exact_boxes_and_independent_bodies(self):
        rng=np.random.default_rng(42)
        for index in range(16):
            world,meta=fixed_training_world(rng,index,np.array(POSITIONS),'layout')
            np.testing.assert_array_equal([[s['x'],s['y']] for s in world.stations],POSITIONS)
            self.assertEqual(meta['layoutHash'],'layout')
            self.assertEqual(world.sensory().shape,(4,297))
            self.assertTrue(world.continuous)
            if index%4!=3:self.assertTrue(all(a.cargo==0 for a in world.agents))
            else:self.assertIn('training-only',meta['scenario'])
            for i,a in enumerate(world.agents):
                for b in world.agents[:i]:self.assertGreater(np.hypot(a.x-b.x,a.y-b.y),.44)

    def test_fixed_evaluation_worlds_have_empty_starts_and_same_layout(self):
        for seed in (13100000,13100001):
            world=RecoveryWorld(seed,positions=POSITIONS,kind='fixed-current',random_starts=True)
            self.assertEqual(world.deliveries,0)
            self.assertTrue(all(a.cargo==0 for a in world.agents))
            np.testing.assert_array_equal([[s['x'],s['y']] for s in world.stations],POSITIONS)

    def test_exchange_rejects_other_layout_experience(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'experience').mkdir();(root/'candidate.npz').write_bytes(b'test')
            exchange=Exchange(root,'token','fixed','run',layout_hash='correct')
            exchange.publish(0,'candidate.npz','parameter')
            packet=protocol_tests.ProtocolTests().packet
            for changes in ({},{'layoutHash':'wrong','worlds':[{'layoutHash':'wrong'}]},
                            {'layoutHash':'correct','worlds':[{'layoutHash':'wrong'}]}):
                with self.assertRaises(ValueError):exchange.accept(packet(**changes))
            self.assertEqual(exchange.accept(packet(layoutHash='correct',worlds=[{'layoutHash':'correct'}])),'accepted')

    def test_transition_practice_covers_stages_without_moving_boxes(self):
        rng = np.random.default_rng(921)
        cases = set(); positive_pickups = positive_deliveries = 0
        for index in range(48):
            world, meta = transition_training_world(rng, index, np.array(POSITIONS), 'fixed')
            np.testing.assert_array_equal([[s['x'], s['y']] for s in world.stations], POSITIONS)
            self.assertEqual(world.sensory().shape, (4, 297))
            self.assertFalse(meta['teacherActions'])
            phase, stage = index % 4, (index // 4) % 3
            cases.add((phase, stage))
            self.assertEqual(world.deliveries, 0)
            for i, agent in enumerate(world.agents):
                for other in world.agents[:i]:
                    self.assertGreater(np.hypot(agent.x-other.x, agent.y-other.y), .48)
                label = LocalTeacher().label(agent)
                if phase in (0, 1): self.assertEqual(agent.cargo, 0)
                else: self.assertEqual(agent.cargo, stage+1)
                if phase == 2: self.assertLess(label[2], 1., 'Departure must release grip')
                if phase == 1: positive_pickups += int(label[2] > 1.)
                if phase == 3: positive_deliveries += int(label[2] > 1.)
            if phase: self.assertTrue(meta['trainingOnlyFixture'])
        self.assertEqual(len(cases), 12)
        self.assertGreater(positive_pickups, 10)
        self.assertGreater(positive_deliveries, 10)

    def test_real_pickup_changes_teacher_grip_without_changing_senses(self):
        rng = np.random.default_rng(924)
        checked = 0
        for index in (1, 5, 9):
            world, _ = transition_training_world(rng, index, np.array(POSITIONS), 'fixed')
            for agent in world.agents:
                teacher = LocalTeacher()
                if teacher.label(agent)[2] <= 1.: continue
                old = world.sensory().copy()
                agent.interact({'interact': True})
                if agent.cargo == 0: continue
                self.assertLess(teacher.label(agent)[2], 1.)
                self.assertFalse(np.array_equal(old, world.sensory()))
                checked += 1
        self.assertGreater(checked, 0)

if __name__=='__main__':unittest.main()
