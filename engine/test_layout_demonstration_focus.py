import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .layout_demonstration_focus import event_choices, EventFocusedDemonstrations, resolve_focus_dataset
from .layout_demonstration_train import DemonstrationCache
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_demonstrations import collect_episode
from .layout_demonstration_cache import window_specs
from .layout_recovery_teacher import KINDS
from .test_layout_demonstration_curriculum import physical_curriculum_fixture
from .test_layout_microfit_flow import TinyTrainBrain


class FocusTests(unittest.TestCase):
    def test_collected_physical_event_windows_have_verified_actors(self):
        arrays, meta = collect_episode(9360123, 'compact', 400)
        specs = [s for s in window_specs(meta, 64, 32) if s['category'] != 'uniform-trajectory']
        self.assertTrue(specs)
        self.assertTrue(any(s['category'].startswith('after-pickup') for s in specs))
        for spec in specs:
            self.assertTrue(event_choices(spec, meta, arrays, 64, 32))

    def test_event_actor_and_post_pickup_window_match_real_transition(self):
        arrays = {'labels': np.full((200, 4, 3), .2), 'bodies': np.zeros((201, 4, 6))}
        arrays['labels'][99, 2, 2] = 2.2
        arrays['bodies'][100:, 2, 5] = 1
        meta = {'frames': 200, 'dt': .05, 'interactionEvents': [
            {'event': 'pickup', 'time': 5., 'fly': 2, 'cargoBefore': 0, 'cargoAfter': 1}]}
        for category, start in [('pickup-1', 19), ('after-pickup-1', 51)]:
            spec = {'category': category, 'start': start}
            self.assertEqual(event_choices(spec, meta, arrays, 64, 32), [{'fly': 2, 'eventTick': 99}])
        with self.assertRaises(ValueError):
            event_choices({'category': 'pickup-2', 'start': 19}, meta, arrays, 64, 32)
        arrays['bodies'][100, 2, 5] = 0
        with self.assertRaisesRegex(ValueError, 'transition'):
            event_choices({'category': 'pickup-1', 'start': 19}, meta, arrays, 64, 32)

    def test_real_data_selection_preserves_every_input_label_prefix_column(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            parent = folder/'candidate.npz'
            parent.write_bytes(b'test-parent')
            model = TinyTrainBrain()
            dataset, cache = physical_curriculum_fixture(folder, model, parent)
            bank = DemonstrationCache(cache, model.checkpoint_hash(), model.fixed_hash, 2, 2, 4)
            focused = EventFocusedDemonstrations(bank, dataset)
            x, y, state, detail = focused.sample(np.random.default_rng(1), 'cpu')
            self.assertEqual(tuple(x.shape), (4, 16, 297))
            self.assertEqual(tuple(y.shape), (4, 16, 3))
            self.assertEqual(tuple(state.shape), (4, 16))
            for kind in KINDS:
                self.assertEqual(sum(s['kind']==kind for s in detail['selections']), 4)
            for col, spec in enumerate(detail['selections']):
                i, fly = spec['windowIndex'], spec['fly']
                np.testing.assert_array_equal(x[:, col].numpy(), bank.x[i, :, fly])
                np.testing.assert_array_equal(y[:, col].numpy(), bank.y[i, :, fly])
                np.testing.assert_array_equal(state[:, col].numpy(), bank.states[i, :, fly])
            bank.x[0, 0, 0, 0] += .1
            with self.assertRaisesRegex(ValueError, 'physical inputs'):
                EventFocusedDemonstrations(bank, dataset)
            bank.manifest['datasetManifestHash'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'identity'):
                EventFocusedDemonstrations(bank, dataset)

    def test_sampler_policy_is_explicit_and_inherited_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder/'source'
            source.mkdir()
            path = source/'manifest.json'
            path.write_text('{}')
            record = {'sourceRun': str(source), 'sourceFileSHA256': {'manifest.json': file_hash(path)}}
            args = SimpleNamespace(event_focused_demonstrations=folder/'data')
            dataset, change = resolve_focus_dataset(args, {'record': record})
            self.assertTrue(change['samplingPolicyChanged'])
            self.assertTrue(change['rngStateRetainedButDrawInterpretationChanges'])
            path.write_text(json.dumps({'eventFocusedDataset': str(dataset)}))
            with self.assertRaises(ValueError):
                resolve_focus_dataset(args, {'record': record})
            record['sourceFileSHA256']['manifest.json'] = file_hash(path)
            inherited, change = resolve_focus_dataset(SimpleNamespace(), {'record': record})
            self.assertEqual(inherited, dataset)
            self.assertFalse(change['samplingPolicyChanged'])
            self.assertEqual(resolve_focus_dataset(SimpleNamespace(), None)[0], None)


if __name__ == '__main__':
    unittest.main()
