"""Exercise the actual trainer's failed-warmup exit without GPU or networking."""
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .layout_grip_calibration import CalibrationBank
from .layout_recovery_train import main
from .layout_microfit_resume import load_continuation


class TinyTrainBrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 4
        self.fixed_hash = 'unchanged-fixture'
        self.log_gains = torch.nn.Parameter(torch.ones(4)*.001)
        self.tonic = torch.nn.Parameter(torch.ones((4, 1))*.001)
        for name, index in [('forward', 0), ('left', 1), ('right', 2), ('interact', 3)]:
            setattr(self, 'motor_'+name, torch.tensor([index]))

    def checkpoint_hash(self):
        return hashlib.sha256(self.log_gains.detach().numpy().tobytes()+self.tonic.detach().numpy().tobytes()).hexdigest()

    def weights(self):
        return self.log_gains[:, None]

    def forward(self, x, steps, state=None, weights=None):
        if state is None:
            state = torch.zeros((4, len(x)))
        return None, .75*state+(weights+self.tonic)*x[:, :4].T

    def audit(self, initial):
        return {'finite': True, 'fixedGraphSensoryDecoderDynamicsUnchanged': True}


class MicrofitFlowTests(unittest.TestCase):
    def test_failed_gate_saves_boundary_checkpoint_and_never_enters_mixed_phase(self):
        model = TinyTrainBrain()
        bank = object.__new__(CalibrationBank)
        bank.burn, bank.parent_hash = 2, model.checkpoint_hash()
        bank.observations = np.ones((4, 16, 297), np.float32)
        bank.targets = np.zeros((2, 16, 3), np.float32)
        bank.targets[:, ::2, 2] = 2.2
        bank.metadata = [{'box': i, 'near': True} for i in range(4)]

        def no_physical_rollout():
            raise AssertionError('Failed warmup must not enter mixed training')

        world = SimpleNamespace(deliveries=0, returns=0, sensory=no_physical_rollout)
        fit = {'gripRecall': 0., 'gripFalsePositiveRate': 0., 'parentMotionMSE': [0., 0.]}
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            folder = Path(root)
            token = folder/'token'
            token.write_text('test-only-token')
            args = SimpleNamespace(root=folder, candidate=folder/'initial.npz', out=folder/'run',
                token_file=token, cert=folder/'cert', key=folder/'key', updates=3, worlds=1,
                burn=2, gradient_frames=2, save_every=2, lr=.001, tonic_lr=.00001,
                reset_seconds=300, seed=1, migrate=False, balanced_curriculum=True, balanced_grip=True,
                grip_calibration=True, calibration_scenes=4, grip_margin=.1, synapse_eps=1e-14,
                block_clip=True, exact_gradient=True, calibration_metrics=True,
                ranking_weight=1., microfit_updates=1, full_calibration_gradient=True)
            prefix = 'engine.layout_recovery_train.'
            stack.enter_context(patch(prefix+'load_model', return_value=model))
            stack.enter_context(patch(prefix+'CalibrationBank', return_value=bank))
            stack.enter_context(patch(prefix+'balanced_world', return_value=(world, {'scenario': 'normal'})))
            stack.enter_context(patch(prefix+'calibration_progress', return_value=fit))
            stack.enter_context(patch(prefix+'save_model', side_effect=lambda path, brain: path.write_bytes(b'fixture checkpoint')))
            stack.enter_context(patch(prefix+'start_server', return_value=SimpleNamespace(shutdown=lambda: None)))
            stack.enter_context(patch(prefix+'time.sleep'))
            stack.enter_context(patch(prefix+'torch.cuda.get_device_name', return_value='CPU unit test'))
            stack.enter_context(patch(prefix+'torch.cuda.max_memory_allocated', return_value=0))
            stack.enter_context(patch('builtins.print'))
            main(args)
            result = json.loads((args.out/'result.json').read_text())
            self.assertEqual(result['updates'], 1)
            self.assertEqual(result['plannedUpdates'], 3)
            self.assertFalse(result['microfitGatePassed'])
            self.assertIn('mixed training not started', result['earlyStop'])
            self.assertTrue((args.out/'candidate-1.npz').exists())
            self.assertTrue((args.out/'optimizer-1.pt').exists())
            self.assertFalse((args.out/'candidate-2.npz').exists())
            self.assertTrue(json.loads((args.out/'status.json').read_text())['finished'])
            manifest = json.loads((args.out/'manifest.json').read_text())
            self.assertEqual(manifest['calibrationDifferentiatedFrames'], 4)
            self.assertEqual(manifest['calibrationLossFrames'], 2)
            self.assertEqual(manifest['physicalReplayDifferentiatedFrames'], 2)
            # A resumed prerequisite keeps its ORIGINAL parent-motion labels
            # and original Adam moments; it must not silently relabel at C1.
            preserved = {name: (args.out/name).read_bytes() for name in
                         ('candidate-1.npz', 'optimizer-1.pt', 'calibration-bank.npz')}
            resumed = SimpleNamespace(**{**vars(args), 'out': folder/'continuation',
                'candidate': args.out/'candidate-1.npz', 'resume_microfit_run': args.out,
                'updates': 4, 'microfit_updates': 3})
            for key, wrong in (('lr', .01), ('full_calibration_gradient', False)):
                with self.assertRaisesRegex(ValueError, 'changes control'):
                    load_continuation(SimpleNamespace(**{**vars(resumed), key: wrong}), model)
            with patch.object(model, 'checkpoint_hash', return_value='wrong'):
                with self.assertRaisesRegex(ValueError, 'checkpoint/interface mismatch'):
                    load_continuation(resumed, model)
            for name, change in (
                    ('status.json', lambda value: {**value, 'finished': False}),
                    ('history.json', lambda value: [{**row, 'trainingPhase': 'mixed-service-training'} for row in value])):
                path = args.out/name
                original_contents = path.read_bytes()
                try:
                    path.write_text(json.dumps(change(json.loads(original_contents))))
                    with self.assertRaisesRegex(ValueError, 'exclusively synthetic prerequisite'):
                        load_continuation(resumed, model)
                finally:
                    path.write_bytes(original_contents)
            with patch(prefix+'CalibrationBank', side_effect=AssertionError('Do not recache parent labels')):
                main(resumed)
            continued_result = json.loads((resumed.out/'result.json').read_text())
            self.assertEqual(continued_result['startUpdate'], 1)
            self.assertEqual(continued_result['updates'], 3)
            self.assertEqual(continued_result['updatesThisRun'], 2)
            self.assertFalse(continued_result['microfitGatePassed'])
            self.assertFalse((resumed.out/'candidate-0.npz').exists())
            self.assertTrue((resumed.out/'candidate-1.npz').exists())
            continued_manifest = json.loads((resumed.out/'manifest.json').read_text())
            self.assertEqual(continued_manifest['calibrationParentParameterHash'], bank.parent_hash)
            self.assertTrue(continued_manifest['continuation']['optimizerMomentsPreserved'])
            old_bank = np.load(args.out/'calibration-bank.npz', allow_pickle=False)
            new_bank = np.load(resumed.out/'calibration-bank.npz', allow_pickle=False)
            np.testing.assert_array_equal(old_bank['targets'], new_bank['targets'])
            old_bank.close()
            new_bank.close()
            for name, contents in preserved.items():
                self.assertEqual((args.out/name).read_bytes(), contents)


if __name__ == '__main__':
    unittest.main()
