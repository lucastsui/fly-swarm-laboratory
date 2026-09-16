import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .layout_recovery_world import RecoveryWorld
from .layout_recovery_demonstrations import (collect_episode, validate_episode, load_episode,
                                             motors_from_labels, body_state)


class DemonstrationTests(unittest.TestCase):
    def test_real_physics_replay_matches_each_observation_and_body(self):
        arrays, meta = collect_episode(9360000, 'compact', 80)
        world = RecoveryWorld(9360000, kind='compact')
        for frame in range(80):
            np.testing.assert_array_equal(arrays['observations'][frame], world.sensory())
            np.testing.assert_array_equal(arrays['bodies'][frame], body_state(world))
            world.advance(motors_from_labels(arrays['labels'][frame]))
        np.testing.assert_array_equal(arrays['bodies'][-1], body_state(world))
        self.assertEqual(meta['interactionCounts']['pickup'], world.pickups)
        self.assertEqual(meta['teacherProductsNotBrainEvidence'], world.deliveries)
        self.assertTrue(meta['teacherActions'])
        self.assertFalse(meta['isBrainEvidence'])
        self.assertFalse(meta['brainWasUsed'])
        self.assertEqual(arrays['episode_starts'].sum(), 1)
        # This is genuine evolving proprioception, not a constant scene bank.
        self.assertGreater(np.max(np.abs(arrays['observations'][0]-arrays['observations'][-1])), 0.)

    def test_numeric_roundtrip_and_determinism(self):
        first, meta = collect_episode(9361001, 'wide', 8, random_starts=True)
        second, _ = collect_episode(9361001, 'wide', 8, random_starts=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'training.npz'
            np.savez_compressed(path, **first, metadata=np.asarray(json.dumps(meta)))
            saved, saved_meta = load_episode(path)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
            np.testing.assert_array_equal(first[key], saved[key])
        self.assertEqual(meta, saved_meta)

    def test_rejects_validation_seeds_and_false_control_provenance(self):
        for seed in (8700000, 9400000, 9700000, 9900000):
            with self.assertRaises(ValueError):
                collect_episode(seed, 'compact', 2)
        arrays, meta = collect_episode(9360001, 'compact', 2)
        for change in ({'control': 'learner-only'}, {'isBrainEvidence': True}, {'teacherActions': False}):
            with self.assertRaises(ValueError):
                validate_episode(arrays, {**meta, **change})
        arrays['episode_starts'][1] = True
        with self.assertRaises(ValueError):
            validate_episode(arrays, meta)

    def test_raw_grip_threshold_matches_fixed_motor_scale(self):
        actions = motors_from_labels(np.array([[.2, -.3, 1.], [.8, .4, 2.2]], np.float32))
        self.assertFalse(actions[0]['interact'])
        self.assertTrue(actions[1]['interact'])
        self.assertAlmostEqual(actions[1]['speed'], .8)


if __name__ == '__main__':
    unittest.main()
