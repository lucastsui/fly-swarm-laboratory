"""CPU regression tests: fixed wiring gradients and independent factory state."""
import tempfile
import unittest
from pathlib import Path

import torch

from .brain import FixedWiringMultiply
from .parallel import ParallelFactories
from .service import Factory
from .training import TrainingRuntime


class ParallelTests(unittest.TestCase):
    def test_fixed_sparse_backward_matches_dense(self):
        torch.manual_seed(19)
        dense = torch.randn(23, 23, dtype=torch.double)
        dense[dense.abs() < .8] = 0
        wiring = dense.to_sparse_csr()
        transpose = dense.T.to_sparse_csr()
        state = torch.randn(23, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda s: FixedWiringMultiply.apply(wiring, transpose, s), (state,)))
        self.assertTrue(torch.allclose(FixedWiringMultiply.apply(wiring, transpose, state), dense @ state))

    def test_factories_share_no_items_rewards_or_fly_objects(self):
        batch = ParallelFactories(Factory, 8, 4)
        self.assertEqual(len(batch.flies), 32)
        self.assertEqual(len({id(f) for f in batch.flies}), 32)
        first, second = batch.worlds[:2]
        first.stock = 0
        first.flies[0]["cargo"] = 3
        first.raw_input.append([0])
        self.assertEqual(second.stock, 28)
        self.assertEqual(second.flies[0]["cargo"], 0)
        self.assertEqual(second.raw_input, [])
        for fly in batch.flies:
            batch.observation(fly)
        self.assertIs(batch.owners[id(first.flies[0])], first)

    def test_checkpoint_restores_worlds_and_randomness(self):
        batch = ParallelFactories(Factory, 2, 4)
        batch.worlds[0].part_output.append({"owners": [1, 3]})
        batch.worlds[1].flies[2]["x"] = 17.25
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pt"
            torch.save(batch.checkpoint(), path)
            restored = ParallelFactories(Factory, 2, 4)
            restored.restore(torch.load(path, weights_only=True))
        self.assertEqual(restored.worlds[1].flies[2]["x"], 17.25)
        self.assertEqual(restored.worlds[0].part_output[0]["owners"], [1, 3])
        self.assertEqual(batch.worlds[0].rng.random(), restored.worlds[0].rng.random())
        self.assertIs(restored.owners[id(restored.flies[4])], restored.worlds[1])

    def test_settings_limit_product_not_just_each_dimension(self):
        TrainingRuntime.validate_settings(4, 8, 50)
        TrainingRuntime.validate_settings(1, 32, 200)
        for setting in ((32, 32, 50), (0, 8, 50), (4, 8, 0)):
            with self.assertRaises(ValueError):
                TrainingRuntime.validate_settings(*setting)


if __name__ == "__main__":
    unittest.main()
