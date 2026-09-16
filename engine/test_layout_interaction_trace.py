import unittest
from .layout_recovery_world import RecoveryWorld
from .layout_interaction_trace import before_interaction, interaction_events


class TraceTests(unittest.TestCase):
    def test_records_pickup_return_and_rejection_without_actuating(self):
        world = RecoveryWorld(kind='original')
        a = world.agents[0]
        a.x, a.y = world.stations[0]['x'], world.stations[0]['y']+.3
        before = before_interaction(world)
        self.assertEqual(interaction_events(world, before, 1), [])
        self.assertEqual(a.cargo, 0)
        a.interact({'interact': True})
        events = interaction_events(world, before, 1)
        self.assertEqual(events[0]['event'], 'pickup')
        self.assertEqual(events[0]['cargoAfter'], 1)
        self.assertEqual(events[0]['nearestBox'], 0)
        before = before_interaction(world)
        a.cooldown = 0
        a.interact({'interact': True})
        self.assertEqual(interaction_events(world, before, 21)[0]['event'], 'return')
        a.x, a.y, a.cooldown = 1., 1., 0.
        before = before_interaction(world)
        a.interact({'interact': True})
        event = interaction_events(world, before, 41)[0]
        self.assertEqual(event['event'], 'rejected')
        self.assertAlmostEqual(event['time'], 2.05)

    def test_no_observer_effect(self):
        a, b = RecoveryWorld(seed=42, kind='compact'), RecoveryWorld(seed=42, kind='compact')
        for tick in range(60):
            motors = [{'speed': .5, 'turn': .2, 'interact': True}]*4
            previous = before_interaction(a)
            a.advance(motors)
            interaction_events(a, previous, tick+1)
            b.advance(motors)
        self.assertEqual(a.snapshot(motors), b.snapshot(motors))


if __name__ == '__main__':
    unittest.main()
