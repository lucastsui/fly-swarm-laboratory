"""Frozen experimental checkpoint viewer. Never trains or controls a learner.

The local publisher atomically selects immutable, hash-checked phase manifests.
A phase switch starts a fresh display factory; delivery never resets a factory.
The established service_viewer on port 8769 is deliberately left untouched.
"""
import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import torch
import uvicorn

from .evaluate_supervised import FrozenPolicy
from .fast_frozen_policy import FastFrozenPolicy, SplitFrozenPolicy
from .neural_telemetry import NeuralTelemetry
from .layout_recovery_brain import INTERFACE, load_model
from .layout_recovery_world import PHYSICS, RecoveryWorld
from .service_viewer import ServiceViewer, app_for, next_tick_deadline
from .plastic_brain import DT

EXPERIMENT = 'experimental-layout-phase-v1'
FIXED_HASH = '634df199d96325a7cd65aed1047a50b5079d22a2c8523f9439ced99bb3548d6f'


def sha_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inspect_candidate(path):
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['interface']) != INTERFACE or str(archive['fixed_hash']) != FIXED_HASH:
            raise ValueError('Wrong graph or sensory/motor interface')
        gains, tonic = archive['gains'], archive['tonic']
        if (gains.shape != (25582938,) or tonic.shape != (166700, 1)
                or gains.dtype != np.float32 or tonic.dtype != np.float32
                or not np.isfinite(gains).all() or not np.isfinite(tonic).all()
                or np.abs(gains).max() > 2.000001 or np.abs(tonic).max() > .100001):
            raise ValueError('Invalid brain parameters')
        h = hashlib.sha256(gains.tobytes() + tonic.tobytes()).hexdigest()
        return h


def read_phase(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (value.get('schema') != EXPERIMENT or value.get('physics') != PHYSICS
            or value.get('sensoryInterface') != INTERFACE or value.get('fixedHash') != FIXED_HASH
            or value.get('validationStatus') not in ('pending', 'incomplete', 'failed', 'passed')):
        raise ValueError('Unsupported phase manifest')
    candidate = Path(value['candidate']).resolve(strict=True)
    allowed = Path(__file__).resolve().parents[1] / '.runtime' / 'layout-training'
    if not candidate.is_relative_to(allowed.resolve()):
        raise ValueError('Checkpoint must be retained inside layout-training')
    if sha_file(candidate) != value['fileSha256'] or inspect_candidate(candidate) != value['parameterHash']:
        raise ValueError('Phase checkpoint hash mismatch')
    for key in ('phase', 'label', 'validationSummary', 'trainingMethod'):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError('Missing phase description: ' + key)
    return value


class PhaseViewer(ServiceViewer):
    experiment = EXPERIMENT

    def __init__(self, args, start_thread=True):
        torch.set_num_threads(4)
        self.args = args
        self.lock = threading.RLock()
        self.simulation_speed = float(args.speed)
        next_tick_deadline(0., 0., self.simulation_speed)
        self.playback_mode = getattr(args, 'playback', 'paced')
        self.measured_speed = 0.
        self.rate_start = time.perf_counter()
        self.rate_ticks = 0
        self.last_capture = 0.
        self.last_neural_capture = 0.
        self.running = True
        self.error = None
        self.reload_error = None
        self.phase_bytes = None
        self.flies = 4
        self.seed = args.seed
        self.layout_mode = True
        self.info = json.loads((args.root / 'brain-info.json').read_text())
        with np.load(args.root / 'brain-spec.npz', allow_pickle=False) as spec:
            self.view_indices = torch.as_tensor(spec['view_indices'].copy(), device='cuda')
        if len(self.view_indices) != len(self.info['nodes']):
            raise ValueError('Anatomy/activity mapping mismatch')
        self.reload_phase()
        if start_thread:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
            self.watcher = threading.Thread(target=self.watch, daemon=True)
            self.watcher.start()

    def reload_phase(self):
        # CUDA graph capture is process-wide sensitive to concurrent CUDA work.
        # Hold the SAME lock as tick/audit throughout model construction and
        # graph warmup/capture, not just while swapping Python references.
        # HTTP snapshot readers remain lock-free while this new phase loads.
        with self.lock:
            return self._reload_phase_locked()

    def _reload_phase_locked(self):
        phase_bytes = self.args.phase.read_bytes()
        if phase_bytes == self.phase_bytes:
            return False
        phase = read_phase(self.args.phase)
        with self.lock:
            if getattr(self, 'weight_hash', None) == phase['parameterHash']:
                # Later evaluation results do not reset the ongoing factory.
                self.phase = phase
                self.phase_bytes = phase_bytes
                self.reload_error = None
                self.publish_state()
                return False
        model = load_model(self.args.root, Path(phase['candidate']))
        model.eval().requires_grad_(False)
        if model.checkpoint_hash() != phase['parameterHash'] or model.fixed_hash != FIXED_HASH:
            raise ValueError('Loaded model identity mismatch')
        backend = getattr(self.args, 'inference_backend', 'original')
        policy = (SplitFrozenPolicy(model, self.flies) if backend == 'cuda-graph-per-fly' else
                  FrozenPolicy(model, self.flies) if backend == 'original' else
                  FastFrozenPolicy(model, self.flies, cuda_graph=backend.startswith('cuda-graph'),
                                   index32=backend.endswith('int32')))
        world = RecoveryWorld(self.seed, flies=self.flies, continuous=True,
                              kind=phase.get('initialLayout', 'wide'))
        # Perform real inference before swapping; failures retain the old view.
        policy.act(world.sensory())
        policy.reset()
        telemetry = (NeuralTelemetry(model, policy, self.view_indices, self.info['nodes'], phase['parameterHash'])
                     if backend != 'original' else None)
        with self.lock:
            self.phase, self.phase_bytes = phase, phase_bytes
            self.model, self.policy, self.world = model, policy, world
            self.telemetry = telemetry
            self.weight_hash = phase['parameterHash']
            self.args.candidate = Path(phase['candidate'])
            self.brain_stats = {k: self.info[k] for k in ('neurons', 'edges', 'sensory', 'initialWeightHash')}
            self.brain_stats.update(plasticSynapses=model.log_gains.numel(), changedSynapses=phase.get('changedSynapses', 0),
                                    weightChange=float(model.log_gains.abs().mean()))
            self.audit = {'checkpointHash': self.weight_hash, 'fileSha256': phase['fileSha256'],
                          'fixedGraphAndInterfaceHash': model.fixed_hash, 'learning': False,
                          'optimizerPresent': False, 'teacher': False, 'noise': False,
                          'decoderTrained': False, 'sensoryInterface': model.interface, 'physics': PHYSICS,
                          'trainableParametersDuringInference': sum(p.numel() for p in model.parameters() if p.requires_grad)}
            self.history = deque([{'seconds': 0., 'products': 0, 'reward': 0.}], maxlen=3600)
            self.motion_samples = deque(maxlen=256)
            self.ticks = self.episodes = self.pickups = self.transfers = self.products = self.layout_changes = 0
            self.reward = 0.
            self.last_delivery = None
            self.session_id = str(uuid.uuid4())
            if self.telemetry:
                self.telemetry.session_id = self.session_id
            self.motor = {'speed': 0., 'turn': 0., 'interact': False,
                          'rates': dict.fromkeys(('forward', 'left', 'right', 'interact'), 0.)}
            self.motors = [copy.deepcopy(self.motor) for _ in range(self.flies)]
            self.world.agents[0].events.append('New training phase: fresh display factory; deliveries never reset it')
            self.reload_error = None
            self.record_motion()
            if self.telemetry:
                self.telemetry.capture(0)
            self.capture(0.)
        print('PHASE_LOADED ' + json.dumps({'phase': phase['phase'], 'checkpointHash': self.weight_hash}), flush=True)
        return True

    @torch.inference_mode()
    def tick(self):
        began = time.perf_counter()
        before = self.world.pickups, self.world.transfers, self.world.deliveries
        self.motors = self.policy.act(self.world.sensory(), explore=False)
        self.motor = self.motors[0]
        rewards, _ = self.world.advance(self.motors)
        self.ticks += 1
        self.pickups += self.world.pickups - before[0]
        self.transfers += self.world.transfers - before[1]
        self.products += self.world.deliveries - before[2]
        self.reward += sum(rewards)
        delivered = self.world.deliveries > before[2]
        if delivered:
            self.last_delivery = {'product': self.world.deliveries, 'seconds': self.world.steps*DT}
        self.record_motion()
        if self.ticks % 20 == 0 or delivered:
            self.history.append({'seconds': self.ticks*DT, 'products': self.products, 'reward': self.reward})
        now = time.perf_counter()
        if getattr(self, 'telemetry', None) and now-self.last_neural_capture >= .05:
            self.telemetry.capture(self.ticks)
            self.last_neural_capture = time.perf_counter()
        self.rate_ticks += 1
        if now - self.rate_start >= 2.:
            self.measured_speed = self.rate_ticks*DT/(now-self.rate_start)
            self.rate_start, self.rate_ticks = now, 0
        # Presentation readback is capped in wall time, never physical/neural steps.
        if now - self.last_capture >= .1:
            self.capture((now-began)*1000)
            self.last_capture = time.perf_counter()

    @torch.inference_mode()
    def loop(self):
        try:
            deadline = time.perf_counter()
            while True:
                with self.lock:
                    if self.running:
                        self.tick()
                    else:
                        self.rate_start, self.rate_ticks, self.measured_speed = time.perf_counter(), 0, 0.
                if self.playback_mode == 'max' and self.running:
                    # Yield to controls and training/evaluation; no artificial 3x cap.
                    time.sleep(0)
                    deadline = time.perf_counter()
                else:
                    deadline = next_tick_deadline(deadline, time.perf_counter(), self.simulation_speed)
                    time.sleep(max(.0005, deadline-time.perf_counter()))
        except Exception as error:
            self.error = str(error)
            self.running = False

    def watch(self):
        while True:
            time.sleep(2.)
            try:
                self.reload_phase()
            except Exception as error:
                self.reload_error = str(error)

    def checkpoint_unchanged(self):
        return self.model.checkpoint_hash() == self.weight_hash

    def retained_checkpoint_unchanged(self):
        return sha_file(self.args.candidate) == self.phase['fileSha256']

    def record_motion(self):
        super().record_motion()
        # A new immutable packet is published atomically by the simulation
        # thread. HTTP readers must never contend with GPU inference for its
        # world lock: repeated lock reacquisition can starve them for seconds.
        self.motion_snapshot = {'sessionId': self.session_id, 'checkpointHash': self.weight_hash,
                                'samples': tuple(self.motion_samples)}

    def motion(self):
        return {**self.motion_snapshot, 'serverTimeMs': time.monotonic()*1000,
                'running': self.running, 'error': self.error}

    def neural_metadata(self):
        return self.telemetry.metadata_bytes if self.telemetry else None

    def neural_activity(self):
        # Publish one immutable packet identity to prevent cross-phase mixing.
        telemetry = self.telemetry
        if not telemetry:
            return None
        return telemetry.packet, {'X-Brain-Checkpoint': telemetry.metadata['checkpointHash'],
                                  'X-Brain-Session': telemetry.session_id,
                                  'X-Brain-Running': '1' if self.running and not self.error else '0'}

    def capture(self, milliseconds):
        super().capture(milliseconds)
        self.publish_state()

    def set_playback(self, mode):
        if mode not in ('max', 'paced'):
            raise ValueError('Playback must be max or paced')
        self.playback_mode = mode
        self.rate_start, self.rate_ticks, self.measured_speed = time.perf_counter(), 0, 0.

    def publish_state(self):
        # Called only under the simulation lock. Each HTTP response sees one
        # coherent checkpoint/factory snapshot without acquiring that lock.
        self.public_snapshot = {**self.snapshot, 'experiment': EXPERIMENT, 'mode': 'frozen', 'runId': self.session_id,
                    'checkpointHash': self.weight_hash, 'checkpointLabel': self.phase['label'],
                    'continuousSupply': True, 'automaticReset': False, 'simulationSpeed': self.simulation_speed,
                    'playbackMode': self.playback_mode, 'measuredSimulationSpeed': self.measured_speed,
                    'inferenceBackend': getattr(self.policy, 'backend', 'original'),
                    'sensoryInterface': self.model.interface, 'sensoryChannels': self.model.input_channels,
                    'layoutEvidence': None, 'swarmEvidence': None, 'layoutChanges': self.layout_changes,
                    'ready': True, 'running': self.running, 'learning': False, 'rewardEnabled': False,
                    'device': torch.cuda.get_device_name(), 'gpuMemoryGB': torch.cuda.memory_allocated()/1e9,
                    'trainingMethod': self.phase['trainingMethod'], 'evidence': None,
                    'phase': {k: self.phase[k] for k in ('phase', 'validationStatus', 'validationSummary', 'publishedAt')},
                    'phaseReloadError': self.reload_error, 'history': list(self.history), 'error': self.error,
                    'motorNeurons': self.info['motorNeurons']}

    def public(self):
        return {**self.public_snapshot, 'running': self.running, 'error': self.error,
                'phaseReloadError': self.reload_error, 'playbackMode': self.playback_mode,
                'measuredSimulationSpeed': self.measured_speed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--seed', type=int, default=5500000)
    parser.add_argument('--speed', type=float, choices=(1., 3.), default=3.)
    parser.add_argument('--playback', choices=('max', 'paced'), default='paced')
    parser.add_argument('--inference-backend', choices=('original', 'cached-csr', 'cuda-graph', 'cuda-graph-int32', 'cuda-graph-per-fly'),
                        default='original')
    viewer = PhaseViewer(parser.parse_args())
    uvicorn.run(app_for(viewer), host='127.0.0.1', port=viewer.args.port, access_log=False, log_level='warning')


if __name__ == '__main__':
    main()
