import unittest
from .layout_motor_replay_diagnostic import summarize


class MotorReplayTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(summarize([]), {'frames': 0})

    def test_actual_boolean_and_cooldown_context(self):
        rows = [{'candidate': [1., .2, 1], 'behavior': [1., .1, 0], 'cargo': 0, 'cooldown': 0.},
                {'candidate': [1.2, -.2, 0], 'behavior': [1., -.2, 1], 'cargo': 2, 'cooldown': .05},
                {'candidate': [1., 0., 1], 'behavior': [1., 0., 0], 'cargo': 3, 'cooldown': .2}]
        r = summarize(rows)
        self.assertEqual(r['frames'], 3)
        self.assertEqual(r['gripFlips'], 3)
        self.assertEqual(r['readyGripFlips'], 2)
        self.assertEqual(r['readyLoadedGripFlips'], 1)
        self.assertEqual(r['readyEmptyGripFlips'], 1)
        self.assertAlmostEqual(r['movementMaxAbs'][0], .2)
        self.assertAlmostEqual(r['movementMaxAbs'][1], .1)


if __name__ == '__main__':
    unittest.main()
