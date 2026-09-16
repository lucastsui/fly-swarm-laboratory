import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .layout_cooldown_continuation import restore_adam, refresh_selected_prefixes, train_segment
from .test_layout_cargo_response_probe import CargoBrain


def optimizer_for(model):
    return torch.optim.Adam([{'params': [model.log_gains], 'lr': .003, 'eps': 1e-14},
                              {'params': [model.tonic], 'lr': .00001, 'eps': 1e-10}])


class CooldownContinuationTests(unittest.TestCase):
    def test_saved_adam_continuation_equals_uninterrupted_next_update(self):
        x = torch.ones((96, 4, 297))
        x[:, ::2, 126] = 0.
        y = torch.full((96, 4, 3), .2)
        y[80:85, 1::2, 2] = 2.2
        state = torch.zeros((4, 4))
        first, continued = CargoBrain(), CargoBrain()
        opt = optimizer_for(first)
        with patch('builtins.print'):
            train_segment(first, opt, x, y, state, 0, 1, 1, lambda *a: None)
            continued.load_state_dict(first.state_dict())
            payload = {'updates': 1, 'optimizer': copy.deepcopy(opt.state_dict())}
            restored = optimizer_for(continued)
            restore_adam(restored, payload, 1)
            a = train_segment(first, opt, x, y, state, 1, 1, 1, lambda *a: None)
            b = train_segment(continued, restored, x, y, state, 1, 1, 1, lambda *a: None)
        self.assertEqual(first.checkpoint_hash(), continued.checkpoint_hash())
        self.assertEqual(a[-1]['update'], 2)
        self.assertEqual(a[-1]['lossBeforeUpdate'], b[-1]['lossBeforeUpdate'])
        for p, q in zip(first.parameters(), continued.parameters()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(opt.state[p][key], restored.state[q][key], rtol=0, atol=0)

    def test_corrupt_moments_and_changed_settings_fail_before_restore(self):
        model = CargoBrain()
        opt = optimizer_for(model)
        sum(p.sum() for p in model.parameters()).backward()
        opt.step()
        payload = {'updates': 1, 'optimizer': copy.deepcopy(opt.state_dict())}
        for mode in ('negative', 'counter', 'learning-rate', 'shape', 'nonfinite'):
            bad = copy.deepcopy(payload)
            state = next(iter(bad['optimizer']['state'].values()))
            if mode == 'negative': state['exp_avg_sq'].fill_(-1.)
            if mode == 'counter': state['step'].fill_(0.)
            if mode == 'learning-rate': bad['optimizer']['param_groups'][0]['lr'] *= 2
            if mode == 'shape': state['exp_avg'] = torch.zeros(999)
            if mode == 'nonfinite': state['exp_avg'].fill_(float('nan'))
            target = optimizer_for(CargoBrain())
            with self.assertRaises(ValueError):
                restore_adam(target, bad, 1)
            self.assertFalse(target.state)

    def test_selected_prefixes_encode_every_preceding_frame_without_reset(self):
        model = CargoBrain().eval().requires_grad_(False)
        rng = np.random.default_rng(123)
        episodes = [({'observations': rng.random((6, 4, 297), dtype=np.float32)},
                     {'seed': i, 'kind': str(i)}) for i in range(2)]
        selections = [{'episode': e, 'seed': e, 'kind': str(e), 'start': start, 'fly': fly}
                      for e, start, fly in ((0, 4, 2), (1, 3, 1), (0, 0, 3))]
        actual = refresh_selected_prefixes(model, episodes, selections)
        for col, s in enumerate(selections):
            expected = torch.zeros((4, 1))
            for row in episodes[s['episode']][0]['observations'][:s['start']]:
                _, expected = model(torch.as_tensor(row[s['fly']:s['fly']+1]), 4, expected, model.weights())
            torch.testing.assert_close(actual[:, col], expected[:, 0], rtol=0, atol=0)
        selections[0]['seed'] = -1
        with self.assertRaises(ValueError):
            refresh_selected_prefixes(model, episodes, selections)


if __name__ == '__main__':
    unittest.main()
