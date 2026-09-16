import copy
import unittest
import torch
from .layout_motor_credit_probe import credit
from .test_layout_microfit_flow import TinyTrainBrain


class MotorCreditTests(unittest.TestCase):
    def test_exact_head_derivatives_do_not_mutate_brain_or_adam(self):
        b = TinyTrainBrain(); b.surrogate_training = False
        opt = torch.optim.Adam([{'params': [b.log_gains], 'lr': .0003},
                                {'params': [b.tonic], 'lr': .000001}])
        (b.log_gains.square().sum()+b.tonic.square().sum()).backward(); opt.step(); opt.zero_grad(set_to_none=True)
        adam = {'optimizer': copy.deepcopy(opt.state_dict()), 'updates': 1}
        original = copy.deepcopy(adam); before = b.checkpoint_hash()
        x, y = torch.ones(5, 4, 297), torch.zeros(5, 4, 3); y[:, :2, 2] = 2.2
        prediction, result = credit(b, x, y, torch.zeros(4, 4), adam, burn=2)
        self.assertEqual(prediction.shape, (3, 4, 3))
        self.assertEqual(len(result['headLosses']), 3)
        self.assertEqual(before, b.checkpoint_hash())
        self.assertTrue(all(p.grad is None for p in b.parameters()))
        for key, state in adam['optimizer']['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, original['optimizer']['state'][key][name], rtol=0, atol=0)
        self.assertTrue(all(v >= 0 for row in result['parameterBlocks'].values() for v in row['headNorms']))


if __name__ == '__main__': unittest.main()
