import asyncio
from collections import deque
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from .phase_viewer import EXPERIMENT, FIXED_HASH, PhaseViewer
from .service_viewer import ServiceViewer, app_for
from .layout_recovery_world import RecoveryWorld


class PhaseTests(unittest.TestCase):
    def test_playback_changes_scheduling_only(self):
        viewer = PhaseViewer.__new__(PhaseViewer)
        viewer.world, viewer.policy = object(), object()
        viewer.ticks, viewer.weight_hash = 123, 'fixed'
        identity = viewer.world, viewer.policy, viewer.ticks, viewer.weight_hash
        viewer.set_playback('max')
        self.assertEqual(viewer.playback_mode, 'max')
        viewer.set_playback('paced')
        self.assertEqual(viewer.playback_mode, 'paced')
        self.assertEqual((viewer.world, viewer.policy, viewer.ticks, viewer.weight_hash), identity)
        with self.assertRaises(ValueError):
            viewer.set_playback('skip-neural-steps')

    def test_fast_tick_preserves_every_physical_step(self):
        def create(cls):
            v = cls.__new__(cls)
            v.world = RecoveryWorld(123, flies=4, continuous=True, kind='wide')
            v.policy = SimpleNamespace(act=lambda *a, **k: [{'speed': .7, 'turn': .12, 'interact': True} for _ in range(4)])
            v.ticks = v.pickups = v.transfers = v.products = 0
            v.reward = 0.; v.history = deque(); v.motion_samples = deque(maxlen=256)
            v.session_id = 'same-factory'; v.weight_hash = 'same-brain'
            v.capture = Mock(); v.last_capture = v.rate_start = 0.; v.rate_ticks = 0
            return v
        original, fast = create(ServiceViewer), create(PhaseViewer)
        for _ in range(200): original.tick(); fast.tick()
        self.assertEqual(original.world.steps, fast.world.steps)
        self.assertEqual(original.ticks, 200)
        self.assertEqual(original.reward, fast.reward)
        self.assertEqual(original.world.snapshot(original.motors), fast.world.snapshot(fast.motors))
        self.assertEqual(list(original.history), list(fast.history))

    def test_metadata_refresh_preserves_factory_and_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'phase.json'
            path.write_text('{"new":true}')
            viewer = PhaseViewer.__new__(PhaseViewer)
            viewer.args = SimpleNamespace(phase=path)
            viewer.phase_bytes = b'old'
            viewer.weight_hash = 'same'
            viewer.lock = threading.RLock()
            viewer.world, viewer.policy = object(), object()
            viewer.publish_state = Mock()
            before = (viewer.world, viewer.policy)
            phase = {'parameterHash': 'same', 'validationSummary': 'new results'}
            with patch('engine.phase_viewer.read_phase', return_value=phase), patch('engine.phase_viewer.load_model') as load:
                self.assertFalse(viewer.reload_phase())
                load.assert_not_called()
            self.assertEqual((viewer.world, viewer.policy), before)
            self.assertEqual(viewer.phase, phase)

    def test_http_snapshots_do_not_wait_for_busy_simulation(self):
        viewer = PhaseViewer.__new__(PhaseViewer)
        class ForbiddenLock:
            def __enter__(self):
                raise AssertionError('HTTP response attempted to wait for inference')
        viewer.lock = ForbiddenLock()
        viewer.public_snapshot = {'checkpointHash': 'fixed', 'runId': 'factory', 'steps': 20}
        viewer.motion_snapshot = {'checkpointHash': 'fixed', 'sessionId': 'factory', 'samples': ({'step': 20},)}
        viewer.running = True; viewer.error = viewer.reload_error = None
        viewer.playback_mode = 'max'; viewer.measured_speed = 4.
        self.assertEqual(viewer.public()['steps'], 20)
        self.assertEqual(viewer.motion()['samples'][0]['step'], 20)
        viewer.running = False
        self.assertFalse(viewer.public()['running'])
        self.assertFalse(viewer.motion()['running'])

    def test_incremental_motion_keeps_endpoint_and_resets_for_new_factory(self):
        viewer = SimpleNamespace(motion=lambda: {'sessionId': 'factory', 'samples': [{'step': 1}, {'step': 2}, {'step': 3}]})
        routes = {route.path: route.endpoint for route in app_for(viewer).routes}
        self.assertEqual(routes['/api/plane/motion'](2, 'factory')['samples'], [{'step': 2}, {'step': 3}])
        self.assertEqual(len(routes['/api/plane/motion'](999, 'old-factory')['samples']), 3)
        self.assertEqual(routes['/api/plane/motion'](999, 'factory')['samples'], [{'step': 3}])

    def test_bad_phase_retains_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'phase.json'; path.write_text('bad')
            viewer = PhaseViewer.__new__(PhaseViewer)
            viewer.args = SimpleNamespace(phase=path)
            viewer.lock = threading.RLock()
            viewer.phase_bytes = b'old'; viewer.model = object()
            before = viewer.model
            with self.assertRaises(json.JSONDecodeError): viewer.reload_phase()
            self.assertIs(viewer.model, before)

    def test_new_graph_capture_excludes_live_inference(self):
        class CaptureProbe(Exception):
            pass
        class TrackingLock:
            depth = 0
            def __enter__(self): self.depth += 1
            def __exit__(self, *args): self.depth -= 1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'phase.json'; path.write_text('new')
            viewer = PhaseViewer.__new__(PhaseViewer)
            viewer.args = SimpleNamespace(phase=path, root=Path(temp), inference_backend='cuda-graph-per-fly')
            viewer.lock = TrackingLock()
            viewer.phase_bytes = b'old'; viewer.weight_hash = 'old'; viewer.flies = 4
            viewer.model = object(); previous = viewer.model
            phase = {'parameterHash': 'new', 'candidate': 'candidate.npz'}
            model = Mock(fixed_hash=FIXED_HASH)
            model.checkpoint_hash.return_value = 'new'
            def build(*args):
                self.assertGreater(viewer.lock.depth, 0, 'Graph capture raced with live tick')
                raise CaptureProbe()
            with patch('engine.phase_viewer.read_phase', return_value=phase), \
                 patch('engine.phase_viewer.load_model', return_value=model), \
                 patch('engine.phase_viewer.SplitFrozenPolicy', side_effect=build):
                with self.assertRaises(CaptureProbe): viewer.reload_phase()
            self.assertIs(viewer.model, previous)
            self.assertEqual(viewer.lock.depth, 0)

    def test_app_uses_combined_parameter_and_archive_checks(self):
        viewer = SimpleNamespace(lock=threading.RLock(), experiment=EXPERIMENT, snapshot={'ready': True},
            weight_hash='combined', error=None, running=True, simulation_speed=3., flies=4,
            checkpoint_unchanged=lambda: True, retained_checkpoint_unchanged=lambda: True,
            audit={'teacher': False, 'optimizerPresent': False})
        routes = {route.path: route.endpoint for route in app_for(viewer).routes}
        def control(value):
            async def json_body(): return value
            return asyncio.run(routes['/api/plane/control'](SimpleNamespace(json=json_body)))
        self.assertEqual(routes['/api/plane/health']()['experiment'], EXPERIMENT)
        self.assertFalse(routes['/api/plane/audit']()['weightsChangedDuringInference'])
        self.assertEqual(control({'action':'save'})['checkpointHash'], 'combined')
        viewer.retained_checkpoint_unchanged = lambda: False
        with self.assertRaises(HTTPException) as error: control({'action':'save'})
        self.assertEqual(error.exception.status_code, 409)
        viewer.checkpoint_unchanged = lambda: False
        self.assertTrue(routes['/api/plane/audit']()['weightsChangedDuringInference'])
        with self.assertRaises(HTTPException) as error: control({'action':'load', 'candidate':'arbitrary'})
        self.assertEqual(error.exception.status_code, 400)


if __name__ == '__main__': unittest.main()
