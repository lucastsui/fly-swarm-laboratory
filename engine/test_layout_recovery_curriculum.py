import unittest
import numpy as np
import torch
from .layout_recovery_train import motor_head_errors
from .layout_recovery_curriculum import balanced_world, episode_limit
from .layout_recovery_teacher import KINDS


class BalancedCurriculumTests(unittest.TestCase):
    def test_each_family_exercises_each_situation_and_both_buffers(self):
        rng = np.random.default_rng(9200001)
        groups = set()
        for i in range(256):
            world, meta = balanced_world(rng, i)
            groups.add((meta['kind'], meta['scenario'], meta['blockedStage']))
            if meta['scenario'] == 'normal':
                self.assertEqual([a.cargo for a in world.agents], [0]*4)
                self.assertEqual([s['stock'] for s in world.stations], [3, 0, 0, 0])
            else:
                self.assertTrue(meta['injectedCargo'])
            if meta['scenario'] == 'blocked':
                stage = meta['blockedStage']
                self.assertEqual(world.stations[stage]['stock'], 2)
                self.assertEqual([a.cargo for a in world.agents], [stage]*4)
            positions = np.asarray([[a.x, a.y] for a in world.agents])
            self.assertGreater(min(np.linalg.norm(positions[a]-positions[b]) for a in range(4) for b in range(a)), .44)
        for kind in KINDS:
            for scenario, stage in (('normal', None), ('loaded', None), ('blocked', 1), ('blocked', 2)):
                self.assertIn((kind, scenario, stage), groups)

    def test_normal_worlds_keep_long_trajectories(self):
        self.assertEqual(episode_limit({'scenario': 'normal'}, 300), 300)
        self.assertEqual(episode_limit({'scenario': 'loaded'}, 300), 90)
        self.assertEqual(episode_limit({'scenario': 'blocked'}, 300), 90)
        self.assertEqual(episode_limit({'scenario': 'normal'}, 60), 60)

    def test_reproducible_without_using_index_for_kind_and_cargo(self):
        a, ma = balanced_world(np.random.default_rng(11), 0)
        b, mb = balanced_world(np.random.default_rng(11), 99)
        self.assertEqual(ma, mb)
        np.testing.assert_array_equal(a.sensory(), b.sensory())

    def test_grip_loss_balances_class_counts(self):
        p = torch.tensor([[.5, .2, 1.3], [.8, -.2, 1.3]])
        y = torch.tensor([[.4, .3, 2.2], [.7, -.1, .2]])
        few = motor_head_errors(p, y, True)[2]
        many = motor_head_errors(torch.cat((p[:1], p[1:].repeat(100, 1))),
                                 torch.cat((y[:1], y[1:].repeat(100, 1))), True)[2]
        torch.testing.assert_close(few, many)

    def test_grip_on_and_off_contexts_get_opposite_gradients(self):
        pred = torch.ones((2, 3), requires_grad=True)
        labels = torch.tensor([[0., 0., 2.2], [0., 0., .2]])
        motor_head_errors(pred, labels, True)[2].backward()
        self.assertLess(float(pred.grad[0, 2]), 0)
        self.assertGreater(float(pred.grad[1, 2]), 0)

    def test_missing_grip_class_is_finite_and_legacy_loss_is_unchanged(self):
        p = torch.ones((4, 2, 3), requires_grad=True)
        y = torch.zeros_like(p)
        loss = motor_head_errors(p, y, True)
        self.assertTrue(torch.isfinite(loss).all())
        torch.testing.assert_close(motor_head_errors(p, y), (p-y).square().mean((0, 1)))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == '__main__':
    unittest.main()
