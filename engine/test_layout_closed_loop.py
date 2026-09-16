import unittest
import numpy as np
import torch
from .layout_closed_loop import (action_dicts, choose_training_actions,
                                  correction_loss, new_world, teacher_fraction)


class CorrectionTests(unittest.TestCase):
    def test_no_teacher_action_when_mask_off(self):
        pred = np.asarray([[.4, -.3, .03], [.8, .5, .01]])
        labels = np.asarray([[1., 1., .2], [0., -1., 2.2]])
        self.assertEqual(choose_training_actions(pred, labels, [False, False]), action_dicts(pred))
        mixed = choose_training_actions(pred, labels, [False, True])
        self.assertEqual(mixed[0], action_dicts(pred)[0])
        self.assertEqual(mixed[1], {'speed': 0., 'turn': -1., 'interact': True})

    def test_teacher_is_removed_before_final_quarter(self):
        self.assertEqual(teacher_fraction(1, 240, .6), .6)
        for update in (180, 181, 200, 240):
            self.assertEqual(teacher_fraction(update, 240, .6), 0.)
        self.assertEqual(teacher_fraction(1, 240, 0.), 0.)

    def test_all_motor_heads_receive_gradients(self):
        pred = torch.tensor([[3., -3., 2.]], requires_grad=True)
        loss, heads = correction_loss(pred, torch.zeros_like(pred))
        loss.backward()
        self.assertTrue(torch.all(pred.grad != 0))
        torch.testing.assert_close(heads, torch.tensor([9., 9., 4.]))

    def test_curriculum_is_explicit_and_normal_worlds_have_no_injected_cargo(self):
        rng = np.random.default_rng(8200000)
        world = new_world(rng, 0, ('wide',), False)
        self.assertEqual([a.cargo for a in world.agents], [0]*4)
        curriculum = new_world(rng, 0, ('wide',), True)
        self.assertEqual([a.cargo for a in curriculum.agents], list(range(4)))
        self.assertTrue(curriculum.collisions_enabled)
        self.assertEqual(curriculum.sensory().shape, (4, 129))


if __name__ == '__main__':
    unittest.main()
