import contextlib
import io
import unittest
import numpy as np
from .layout_batched_prefix_cache import fill_states
from .test_layout_microfit_flow import TinyTrainBrain


class BatchedPrefixTests(unittest.TestCase):
    def test_world_batches_preserve_each_actor_full_history(self):
        rng = np.random.default_rng(91)
        episodes = [({'observations': rng.normal(size=(12, 4, 297)).astype(np.float32)}, {})
                    for _ in range(7)]
        specs = [{'episode': i, 'start': start} for i in range(7) for start in (0, 3, 11)]
        model = TinyTrainBrain().eval().requires_grad_(False)
        with contextlib.redirect_stdout(io.StringIO()):
            individual = fill_states(model, episodes, specs, 1)
            for count in (2, 4, 7, 32):
                np.testing.assert_array_equal(individual, fill_states(model, episodes, specs, count))
        self.assertTrue(np.all(individual[::3] == 0))
        self.assertGreater(np.abs(individual[2::3]).sum(), 0)
        self.assertFalse(np.array_equal(individual[2], individual[5]))

    def test_reject_invalid_batch_missing_window_and_different_horizons(self):
        model = TinyTrainBrain().eval().requires_grad_(False)
        episodes = [({'observations': np.zeros((12, 4, 297), np.float32)}, {})]
        specs = [{'episode': 0, 'start': 0}]
        for count in (0, 33, 1.5, True):
            with self.assertRaises(ValueError):
                fill_states(model, episodes, specs, count)
        with self.assertRaises(ValueError):
            fill_states(model, episodes, [{'episode': 1, 'start': 0}])
        episodes.append(({'observations': np.zeros((11, 4, 297), np.float32)}, {}))
        with self.assertRaises(ValueError):
            fill_states(model, episodes, specs)


if __name__ == '__main__':
    unittest.main()
