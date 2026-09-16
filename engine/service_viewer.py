"""Read-only live embodiment of the verified whole-line service checkpoint.

Uses exactly the frozen evaluator, original senses, four neural substeps and
world physics. No optimizer, teacher actions, worker aggregation or learning.
"""
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import uvicorn

from .evaluate_supervised import FrozenPolicy
from .supervised_steering import TrainableConnectome
from .swarm_world import SwarmWorld
from .layout_world import LayoutSwarmWorld,random_layout,validate_layout
from .layout_brain import LayoutConnectome
from .plastic_brain import digest, DT
from .haul_server import ORIGINS


EXPERIMENT = 'verified-whole-line-service-v1'


def viewer_interface(metadata):
    interface=metadata.get('sensoryInterface','original-30')
    if interface not in ('original-30','local-color-cargo-v1','local-color-cargo-v2','local-color-cargo-v3'):
        raise ValueError('Unsupported sensory interface; experimental checkpoints require explicit integration and verification')
    return interface


def next_tick_deadline(previous, now, speed):
    """Change wall-clock pacing only; never enlarge the physical/neural step."""
    if speed not in (1.,3.):raise ValueError('Supported simulation speeds: 1 or 3')
    return max(previous+DT/speed,now)


def verified_evidence(metadata, directory, weight_hash):
    """Use actual paired test reports, not an invented training curve."""
    if metadata['gainValuesSha256'] != weight_hash:
        raise ValueError('Selected checkpoint hash mismatch')
    finite = json.loads((directory/'sequence200-finite64.json').read_text())['finite']
    baseline = json.loads((directory/'baseline-finite64.json').read_text())['finite']
    continuous = json.loads((directory/'sequence200-continuous32.json').read_text())['continuous']
    control = json.loads((directory/'baseline-continuous32.json').read_text())['continuous']
    for learned, old in ((finite, baseline), (continuous, control)):
        if learned['gainsHash'] != weight_hash or old['gainsHash'] != metadata['baselineGainValuesSha256']:
            raise ValueError('Evidence/checkpoint mismatch')
        if [t['seed'] for t in learned['trials']] != [t['seed'] for t in old['trials']]:
            raise ValueError('Unpaired evidence')
        for report in (learned, old):
            if any(report[k] for k in ('teacher', 'learning', 'noise')):
                raise ValueError('Expected frozen unassisted verification')
    curve = [{'seconds': 0, 'baseline': 0., 'selected': 0.}]
    for seconds, counts in finite['checkpoints'].items():
        curve.append({'seconds': int(seconds),
                      'baseline': 100*baseline['checkpoints'][seconds]['episodesAllThree']/baseline['cases'],
                      'selected': 100*counts['episodesAllThree']/finite['cases']})
    final = finite['checkpoints'][str(int(finite['seconds']))]
    old = baseline['checkpoints'][str(int(baseline['seconds']))]
    return {'curve': curve, 'cases': finite['cases'], 'seconds': finite['seconds'],
            'cleared': final['episodesAllThree'], 'baselineCleared': old['episodesAllThree'],
            'products': final['products'], 'baselineProducts': old['products'],
            'continuousCases': continuous['cases'],
            'continuousProducts': sum(t['products'] for t in continuous['trials']),
            'baselineContinuousProducts': sum(t['products'] for t in control['trials']),
            'limitations': metadata['limitations']}


class ServiceViewer:
    def __init__(self, args, start_thread=True):
        torch.set_num_threads(4)
        self.args = args
        self.simulation_speed = float(getattr(args,'speed',3.))
        next_tick_deadline(0.,0.,self.simulation_speed)
        self.lock = threading.RLock()
        self.running = True
        self.error = None
        self.metadata = json.loads(args.metadata.read_text())
        interface=viewer_interface(self.metadata)
        with np.load(args.candidate, allow_pickle=False) as archive:
            if 'tonic' in archive.files:
                raise ValueError('This viewer does not yet support experimental excitability checkpoints')
            gains = archive['gains'].copy()
        self.weight_hash = digest(gains)
        self.layout_mode=interface!='original-30'
        if self.metadata['gainValuesSha256']!=self.weight_hash:raise ValueError('Checkpoint hash mismatch')
        self.evidence = None if self.layout_mode else verified_evidence(self.metadata, args.evidence, self.weight_hash)
        self.layout_evidence=None
        if self.layout_mode:
            report=json.loads(Path(self.metadata['layoutEvidence']).read_text())
            if (report['checkpointHash']!=self.weight_hash or not report['colored'] or report['teacher'] or report['learning']
                    or report.get('sensoryInterface')!=self.metadata['sensoryInterface']):
                raise ValueError('Expected hash-matched frozen layout evidence')
            self.layout_evidence={k:report[k] for k in ('summary','seconds','seed')}
        self.model = (LayoutConnectome(args.root,gains,overlay=self.metadata['sensoryInterface']=='local-color-cargo-v2',
                                      stock=self.metadata['sensoryInterface']=='local-color-cargo-v3')
                      if self.layout_mode else TrainableConnectome(args.root,gains))
        self.model.eval().requires_grad_(False)
        self.flies = int(getattr(args,'flies',4))
        self.policy = FrozenPolicy(self.model, self.flies)
        self.swarm_evidence = None
        self.load_swarm_evidence()
        self.info = json.loads((args.root/'brain-info.json').read_text())
        with np.load(args.root/'brain-spec.npz', allow_pickle=False) as spec:
            self.view_indices = torch.as_tensor(spec['view_indices'].copy(), device='cuda')
        if len(self.view_indices) != len(self.info['nodes']):
            raise ValueError('Anatomy/activity mapping mismatch')
        self.audit = {**self.model.audit(gains), 'backpropagation': False,
                      'trainingUsedBackpropagation': True, 'optimizerPresent': False,
                      'learning': False, 'teacher': False, 'noise': False,
                      'weightsChangedDuringInference': False, 'checkpointHash': self.weight_hash}
        self.brain_stats = {k: self.info[k] for k in ('neurons', 'edges', 'sensory', 'initialWeightHash')}
        self.brain_stats.update(plasticSynapses=len(gains),
                                changedSynapses=self.metadata['changedSynapticGains'],
                                weightChange=float(np.abs(gains).mean()))
        self.history = deque(maxlen=3600)
        self.ticks = self.episodes = self.pickups = self.transfers = self.products = 0
        self.reward = 0.
        self.seed = args.seed
        self.session_id = str(uuid.uuid4())
        self.last_delivery = None
        self.motion_samples = deque(maxlen=32)
        self.world = (LayoutSwarmWorld(self.seed,flies=self.flies,continuous=True,kind=self.metadata.get('initialLayout','wide'),
                                      stock=self.metadata['sensoryInterface']=='local-color-cargo-v3')
                      if self.layout_mode else SwarmWorld(self.seed, flies=self.flies, continuous=True))
        self.layout_changes=0
        self.motor = {'speed': 0., 'turn': 0., 'interact': False,
                      'rates': dict.fromkeys(('forward', 'left', 'right', 'interact'), 0.)}
        self.motors = [copy.deepcopy(self.motor) for _ in range(self.flies)]
        self.history.append({'seconds': 0., 'products': 0, 'reward': 0.})
        self.snapshot = {}
        self.record_motion()
        self.capture(0.)
        if start_thread:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def record_motion(self):
        poses=[{'id':i,'x':a.x,'y':a.y,'heading':a.heading,'cargoStage':a.cargo,
                'contact':a.contact,'deliveries':a.deliveries} for i,a in enumerate(self.world.agents)]
        self.motion_samples.append({'step':self.ticks,'timeMs':time.monotonic()*1000,
                                    **poses[0],'avatars':poses})

    def motion(self):
        with self.lock:
            return {'sessionId':self.session_id,'checkpointHash':self.weight_hash,
                    'serverTimeMs':time.monotonic()*1000,'running':self.running,
                    'samples':list(self.motion_samples),'error':self.error}

    @torch.inference_mode()
    def tick(self):
        began = time.perf_counter()
        before = (self.world.pickups, self.world.transfers, self.world.deliveries)
        self.motors = self.policy.act(self.world.sensory(), explore=False)
        self.motor = self.motors[0]
        rewards, _ = self.world.advance(self.motors)
        self.ticks += 1
        self.pickups += self.world.pickups-before[0]
        self.transfers += self.world.transfers-before[1]
        self.products += self.world.deliveries-before[2]
        self.reward += sum(rewards)
        delivered = self.world.deliveries > before[2]
        if delivered:
            self.last_delivery = {'product':self.world.deliveries,'seconds':self.world.steps*DT}
        self.record_motion()
        if self.ticks % 20 == 0 or delivered:
            self.history.append({'seconds': self.ticks*DT, 'products': self.products, 'reward': self.reward})
        # Five activity frames per simulated second; physics remains 20 Hz in simulation time.
        if self.ticks % 4 == 0 or delivered:
            self.capture((time.perf_counter()-began)*1000)

    def capture(self, milliseconds):
        state = self.policy.state
        if state is not None:
            names=('forward','left','right','interact')
            rates=torch.stack([state[getattr(self.model,'motor_'+name)].mean(0) for name in names]).cpu().numpy()
            for i,motor in enumerate(self.motors):motor['rates']={name:float(rates[j,i]) for j,name in enumerate(names)}
        activity = [0.]*len(self.view_indices) if state is None else state[self.view_indices, 0].cpu().tolist()
        self.snapshot = {**copy.deepcopy(self.world.snapshot(self.motors)),
                         'steps': self.ticks, 'simSeconds': self.world.steps*DT,
                         'elapsedSimSeconds': self.ticks*DT, 'seed': self.seed,
                         'episodes': self.episodes, 'lastDelivery': self.last_delivery,
                         'totalPickups': self.pickups, 'totalTransfers': self.transfers,
                         'totalProducts': self.products, 'tickMilliseconds': milliseconds,
                         'brain': {**self.brain_stats, 'meanActivity': 0. if state is None else float(state.mean())},
                         'brainView': {'nodes': self.info['nodes'], 'edges': [], 'bounds': [], 'activity': activity}}

    @torch.inference_mode()
    def loop(self):
        try:
            deadline = time.perf_counter()
            while True:
                with self.lock:
                    if self.running:
                        self.tick()
                deadline = next_tick_deadline(deadline,time.perf_counter(),self.simulation_speed)
                time.sleep(max(.0005,deadline-time.perf_counter()))
        except Exception as error:
            self.error = str(error)
            self.running = False

    def load_swarm_evidence(self):
        path=getattr(self.args,'swarm_evidence',None)
        # A bounded confirmation run can finish while the live view continues.
        # Only complete, hash-matched reports become dashboard evidence.
        if self.swarm_evidence is None and path and path.is_file():
            report=json.loads(path.read_text())
            if report['checkpointHash']!=self.weight_hash or report['learning'] or report['teacher'] or not report['weightsUnchanged']:
                raise ValueError('Swarm evidence/checkpoint mismatch')
            self.swarm_evidence={k:report[k] for k in ('summary','seconds','seed','minSeparation')}

    def public(self):
        with self.lock:
            self.load_swarm_evidence()
            return {**self.snapshot, 'experiment': EXPERIMENT, 'mode': 'frozen',
                    'runId': self.session_id, 'checkpointHash': self.weight_hash,
                    'continuousSupply':True,'automaticReset':False,
                    'simulationSpeed':self.simulation_speed,
                    'swarmEvidence':self.swarm_evidence,
                    'checkpointLabel': self.metadata.get('label','Whole-line service · sequence update 200'),
                    'sensoryInterface':self.metadata.get('sensoryInterface','original-30'),
                    'sensoryChannels':self.model.input_channels,
                    'layoutEvidence':self.layout_evidence,'layoutChanges':self.layout_changes,
                    'ready': bool(self.snapshot), 'running': self.running, 'learning': False,
                    'rewardEnabled': False, 'device': torch.cuda.get_device_name(),
                    'gpuMemoryGB': torch.cuda.memory_allocated()/1e9,
                    'trainingMethod': self.metadata['trainingMethod'],
                    'training': {'running': False, 'dopamineLearning': False, 'decoderTrained': False},
                    'evidence': self.evidence, 'history': list(self.history), 'error': self.error,
                    'motorNeurons': self.info['motorNeurons']}


def app_for(viewer):
    app = FastAPI(title='Verified whole-line service: frozen brain')

    @app.middleware('http')
    async def protect(request, call_next):
        origin = request.headers.get('origin', '')
        if request.url.hostname not in ('localhost', '127.0.0.1') or (origin and origin not in ORIGINS):
            return Response(status_code=403)
        response = Response(status_code=204) if request.method == 'OPTIONS' else await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        if origin:
            response.headers.update({'Access-Control-Allow-Origin': origin, 'Vary': 'Origin',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Expose-Headers': 'X-Brain-Checkpoint, X-Brain-Session, X-Brain-Running',
                'Access-Control-Allow-Private-Network': 'true'})
        return response

    @app.get('/api/plane/state')
    def state():
        return viewer.public()

    @app.get('/api/plane/motion')
    def motion(after: int | None = None, session: str | None = None):
        # Small, frequently polled stream; do not resend anatomy or chart history.
        packet = viewer.motion()
        if after is not None and session == packet['sessionId']:
            # Include the prior endpoint for interpolation. A new factory must
            # always send its full starting buffer, irrespective of old steps.
            samples = packet['samples']
            packet = {**packet, 'samples': [sample for sample in samples if sample['step'] >= after] or samples[-1:]}
        return packet

    @app.get('/api/plane/health')
    def health():
        return {'ready': bool(viewer.snapshot), 'experiment': getattr(viewer,'experiment',EXPERIMENT), 'learning': False,
                'checkpointHash': viewer.weight_hash, 'error': viewer.error,
                'continuousSupply':True,'automaticReset':False,'simulationSpeed':viewer.simulation_speed,
                'flies':viewer.flies,'flyCollisions':True}

    @app.get('/api/plane/neural/metadata')
    def neural_metadata():
        data = viewer.neural_metadata() if hasattr(viewer, 'neural_metadata') else None
        if data is None:
            raise HTTPException(404, 'Connection telemetry unavailable for this viewer')
        return Response(data, media_type='application/json')

    @app.get('/api/plane/neural/activity')
    def neural_activity():
        result = viewer.neural_activity() if hasattr(viewer, 'neural_activity') else None
        if result is None:
            raise HTTPException(404, 'Connection telemetry unavailable for this viewer')
        data, headers = result
        return Response(data, media_type='application/octet-stream', headers=headers)

    @app.get('/api/plane/audit')
    def audit():
        with viewer.lock:
            unchanged = (viewer.checkpoint_unchanged() if hasattr(viewer,'checkpoint_unchanged') else
                         digest(viewer.model.log_gains.detach().cpu().numpy()) == viewer.weight_hash)
            return {**viewer.audit, 'weightsChangedDuringInference': not unchanged}

    @app.post('/api/plane/control')
    async def control(request: Request):
        value = await request.json()
        with viewer.lock:
            if value == {'action': 'pause'}:
                viewer.running = False
            elif value == {'action': 'start'} and not viewer.error:
                viewer.running = True
            elif (isinstance(value, dict) and set(value) == {'action', 'mode'}
                  and value['action'] == 'playback' and hasattr(viewer, 'set_playback')):
                try:
                    viewer.set_playback(value['mode'])
                except ValueError as error:
                    raise HTTPException(400, str(error)) from error
            elif value == {'action': 'save'}:
                # Already durable and immutable: verify the retained file, never overwrite it.
                if hasattr(viewer,'retained_checkpoint_unchanged'):
                    if not viewer.retained_checkpoint_unchanged():
                        raise HTTPException(409, 'Retained checkpoint changed')
                else:
                    with np.load(viewer.args.candidate, allow_pickle=False) as archive:
                        if digest(archive['gains']) != viewer.weight_hash:
                            raise HTTPException(409, 'Retained checkpoint changed')
            elif isinstance(value,dict) and value.get('action') in ('layout','randomize'):
                if not viewer.layout_mode:raise HTTPException(409,'This brain is not the layout-trained model')
                try:
                    if value['action']=='randomize':
                        kind=value.get('kind','wide')
                        if kind not in ('original','rotated','permuted','compact','wide'):raise ValueError('Unknown layout kind')
                        positions=random_layout(viewer.seed+10000000+viewer.layout_changes,kind)
                    else:
                        kind='custom';positions=validate_layout(value.get('positions'))
                    viewer.world.set_layout(positions);viewer.world.layout_kind=kind;viewer.layout_changes+=1
                    viewer.world.agents[0].events.append('Box positions changed; bodies, cargo and brain states retained')
                    viewer.capture(0.)
                except (ValueError,TypeError) as error:raise HTTPException(400,str(error)) from error
            else:
                raise HTTPException(400, 'Frozen inference permits pause, start, save only; training is unavailable.')
        return {'running': viewer.running, 'learning': False, 'checkpointHash': viewer.weight_hash,
                'message': 'Verified checkpoint is already saved; weights unchanged.'}
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'metadata', 'evidence'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--seed', type=int, default=5500000)
    parser.add_argument('--speed', type=float, choices=(1.,3.), default=3.)
    parser.add_argument('--flies', type=int, default=4)
    parser.add_argument('--swarm-evidence', type=Path)
    args = parser.parse_args()
    viewer = ServiceViewer(args)
    print('FROZEN_SERVICE_VIEWER_READY '+viewer.weight_hash, flush=True)
    uvicorn.run(app_for(viewer), host='127.0.0.1', port=args.port, access_log=False, log_level='warning')
