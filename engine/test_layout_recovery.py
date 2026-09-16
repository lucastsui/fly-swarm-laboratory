import unittest
import numpy as np
from .layout_world import LayoutSwarmWorld, color_sensory
from .layout_recovery_world import RecoveryWorld, recovery_sensory, CHANNELS
from .layout_recovery_brain import status_projection
from .layout_recovery_teacher import LocalTeacher, training_world


class RecoveryTests(unittest.TestCase):
    def place(self, world, index=0, station=1):
        agent = world.agents[index]
        agent.x, agent.y = world.stations[station]['x'], world.stations[station]['y']+.3
        agent.heading = -np.pi/2
        agent.cooldown = 0.
        return agent

    def test_return_requires_grip_and_preserves_material(self):
        world = RecoveryWorld(kind='original', continuous=False)
        agent = self.place(world)
        agent.cargo = 2
        world.stations[1]['stock'] = 2
        before = world.material_count()
        agent.interact({'interact': False})
        self.assertEqual(agent.cargo, 2)
        agent.interact({'interact': True})
        self.assertEqual((agent.cargo, world.stations[1]['stock'], world.stations[1]['returned']), (0, 2, 1))
        self.assertEqual(world.material_count(), before)
        agent.cooldown = 0
        agent.interact({'interact': True})
        self.assertEqual((agent.cargo, world.stations[1]['stock'], world.stations[1]['returned']), (2, 2, 0))
        self.assertEqual(world.material_count(), before)

    def test_all_loaded_full_assembler_deadlock_has_legal_escape(self):
        world = RecoveryWorld(kind='original', continuous=False)
        world.stations[2]['stock'] = 2
        for agent in world.agents:
            agent.cargo = 2
        before = world.material_count()
        agent = self.place(world, station=1)
        agent.interact({'interact': True})
        self.assertEqual(agent.cargo, 0)
        agent = self.place(world, station=2)
        agent.interact({'interact': True})
        self.assertEqual((agent.cargo, world.stations[2]['stock']), (3, 1))
        self.assertEqual(world.material_count(), before)
        # A second fly can now unload into the freed processing input.
        second = self.place(world, index=1, station=2)
        second.interact({'interact': True})
        self.assertEqual(second.cargo, 0)
        self.assertEqual(world.material_count(), before)

    def test_wrong_station_and_remote_interaction_cannot_destroy_material(self):
        world = RecoveryWorld(kind='original', continuous=False)
        agent = self.place(world, station=3)
        agent.cargo = 1
        before = world.material_count()
        agent.interact({'interact': True})
        self.assertEqual(agent.cargo, 1)
        agent.x, agent.y, agent.cooldown = 1., 1., 0.
        agent.interact({'interact': True})
        self.assertEqual(world.material_count(), before)
        self.assertEqual(agent.cargo, 1)

    def test_continuous_raw_supply_does_not_erase_returns(self):
        world = RecoveryWorld(kind='original')
        agent = self.place(world, station=0)
        agent.cargo = 1
        agent.interact({'interact': True})
        world.advance([{'speed': 0., 'turn': 0., 'interact': False}]*4)
        self.assertEqual(world.stations[0]['returned'], 1)

    def test_capacity_alias_is_removed_and_legacy_channels_are_unchanged(self):
        world = RecoveryWorld(kind='original')
        agent = self.place(world, station=2)
        agent.cargo = 2
        empty = recovery_sensory(agent)
        world.stations[2]['stock'] = 2
        full = recovery_sensory(agent)
        np.testing.assert_array_equal(empty[:129], full[:129])
        self.assertGreater(np.max(np.abs(empty[129:]-full[129:])), .1)
        self.assertEqual(world.sensory().shape, (4, CHANNELS))
        np.testing.assert_array_equal(full[:129], color_sensory(agent))

    def test_hidden_box_status_does_not_leak(self):
        world = RecoveryWorld(kind='original')
        agent = world.agents[0]
        agent.x, agent.y, agent.cargo = 1., 1., 2
        before = recovery_sensory(agent)
        world.stations[2]['stock'], world.stations[2]['timer'] = 2, 1.
        np.testing.assert_array_equal(before, recovery_sensory(agent))

    def test_projection_covers_all_status_bearings_without_new_cells(self):
        annotated = np.tile(np.arange(30, 126), 2)
        indices, mask = status_projection(annotated)
        self.assertEqual(len(indices), len(annotated))
        self.assertEqual(set(indices[mask > 0]), set(range(129, CHANNELS)))
        np.testing.assert_array_equal((indices-129) % 24, (annotated-30) % 24)

    def test_legacy_world_still_rejects_returns(self):
        world = LayoutSwarmWorld(kind='original')
        agent = self.place(world)
        agent.cargo = 2
        agent.interact({'interact': True})
        self.assertEqual(agent.cargo, 2)
        self.assertNotIn('returned', world.stations[1])

    def test_teacher_avoids_reflexive_return_after_pickup(self):
        world = RecoveryWorld(kind='original')
        agent = self.place(world, station=0)
        teacher = LocalTeacher()
        self.assertGreater(teacher.label(agent)[2], 1.)
        agent.cargo = 1
        self.assertLess(teacher.label(agent)[2], 1.)

    def test_training_teacher_remembers_a_visibly_full_destination(self):
        world = RecoveryWorld(kind='original')
        agent = self.place(world, station=2)
        agent.cargo = 2
        world.stations[2]['stock'] = 2
        teacher = LocalTeacher()
        teacher.label(agent)
        self.assertEqual(teacher.return_stage, 2)
        agent = self.place(world, station=1)
        self.assertGreater(teacher.label(agent)[2], 1.)
        # Labels do not alter the world or actuate the grip.
        self.assertEqual(agent.cargo, 2)
        self.assertEqual(world.returns, 0)

    def test_finite_conservation_under_random_actions(self):
        world = RecoveryWorld(seed=21, kind='compact', continuous=False, random_starts=True)
        before = world.material_count()
        rng = np.random.default_rng(21)
        for _ in range(200):
            world.advance([{'speed': float(rng.uniform(0, 2)), 'turn': float(rng.uniform(-2, 2)),
                            'interact': bool(rng.integers(2))} for _ in range(4)])
            self.assertEqual(world.material_count(), before)

    def test_curriculum_tagged_and_collision_free_starts(self):
        rng = np.random.default_rng(1)
        for i in range(20):
            world, meta = training_world(rng, i)
            positions = np.asarray([[a.x, a.y] for a in world.agents])
            self.assertGreater(min(np.linalg.norm(positions[a]-positions[b]) for a in range(4) for b in range(a)), .44)
            self.assertEqual(world.sensory().shape, (4, CHANNELS))
            if i % 4 in (2, 3):
                self.assertIn('training-only', meta['scenario'])


if __name__ == '__main__':
    unittest.main()
