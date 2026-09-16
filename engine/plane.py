"""Fresh laptop-only connectome embodiment. Observation mode cannot enable learning.

No checkpoint loading, optimizer, reward shaping, goals, pathfinder or random actions.
The 2D sensory projection and descending/motor readout are explicitly artificial.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import pyarrow.feather as feather
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import uvicorn

PROJECT = Path(__file__).resolve().parents[1]
GRAPH = PROJECT.parent / 'connectome-data'
STORAGE = PROJECT / '.runtime' / 'untrained-plane'
DT = .05
SEED = 260913
from .deployment import ORIGINS


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Plane:
    width, height = 20., 14.
    radius, sensing_range = .22, 6.

    def __init__(self):
        self.x, self.y, self.heading = 10., 7., -.6
        self.speed = self.turn = 0.
        self.cargo = False
        self.material = {'x': 6., 'y': 5., 'radius': .35, 'kind': 'material', 'color': '#efac69'}
        self.depot = {'x': 15., 'y': 9., 'radius': .6, 'kind': 'depot', 'color': '#88dec1'}
        self.landmarks = [self.material, self.depot,
                          {'x': 13., 'y': 3., 'radius': .5, 'kind': 'marker', 'color': '#a3a7ff'}]
        self.obstacles = [[8., 10., .7], [12., 6., .65]]
        self.trajectory = deque([[self.x, self.y]], maxlen=1500)
        self.distance = 0.
        self.collisions = self.deliveries = self.interactions = 0
        self.contact = False
        self.cooldown = 0.
        self.events = deque(maxlen=12)

    def sensory(self):
        # Two artificial 150-degree retinal fans, 12 bins per eye. Body-ID order
        # supplies a reproducible projection, NOT reconstructed ommatidial retinotopy.
        angles = np.r_[np.linspace(-2.6, -.05, 12), np.linspace(.05, 2.6, 12)]
        visual = np.full(24, .025, np.float32)
        for item in self.landmarks:
            if item is self.material and self.cargo:
                continue
            dx, dy = item['x'] - self.x, item['y'] - self.y
            distance = math.hypot(dx, dy)
            if distance > self.sensing_range:
                continue
            bearing = wrap(math.atan2(dy, dx) - self.heading)
            sigma = max(.12, math.atan2(item['radius'], max(distance, .1)))
            difference = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
            visual += np.exp(-.5 * (difference / sigma) ** 2) * (1 - distance / self.sensing_range)
        smell = []
        for lateral in (-.2, .2):
            ax = self.x + .25 * math.cos(self.heading) - lateral * math.sin(self.heading)
            ay = self.y + .25 * math.sin(self.heading) + lateral * math.cos(self.heading)
            distance = math.hypot(ax - self.material['x'], ay - self.material['y'])
            smell.append(max(0., 1 - distance / self.sensing_range) ** 2 if not self.cargo else 0.)
        wall = min(self.x, self.width - self.x, self.y, self.height - self.y)
        proximity = max(0., 1 - wall)
        for ox, oy, radius in self.obstacles:
            proximity = max(proximity, max(0., 1 - (math.hypot(ox - self.x, oy - self.y) - radius)))
        touch_material = float(not self.cargo and math.hypot(self.x - self.material['x'], self.y - self.material['y']) < .7)
        # Channels 24/25 smell, 26 contact/proximity, 27 cargo/contact taste proxy,
        # 28/29 self-motion. No target coordinates or demanded action are supplied.
        return np.r_[np.clip(visual, 0, 1), smell, max(float(self.contact), proximity),
                     max(float(self.cargo), touch_material), abs(self.speed) / 2, abs(self.turn) / 2].astype(np.float32)

    def advance(self, motor):
        self.speed, self.turn = float(motor['speed']), float(motor['turn'])
        self.heading = wrap(self.heading + self.turn * DT)
        x = self.x + self.speed * math.cos(self.heading) * DT
        y = self.y + self.speed * math.sin(self.heading) * DT
        blocked = not (self.radius <= x <= self.width-self.radius and self.radius <= y <= self.height-self.radius)
        blocked |= any(math.hypot(x-ox, y-oy) < r+self.radius for ox, oy, r in self.obstacles)
        if blocked and not self.contact:
            self.collisions += 1
        self.contact = bool(blocked)
        if not blocked:
            self.distance += math.hypot(x-self.x, y-self.y)
            self.x, self.y = x, y
        self.cooldown = max(0., self.cooldown-DT)
        outcome = 0.
        if motor['interact'] and self.cooldown <= 0:
            self.interactions += 1
            self.cooldown = 1.
            if self.cargo:
                self.cargo = False
                self.material.update(x=self.x, y=self.y)
                if math.hypot(self.x-self.depot['x'], self.y-self.depot['y']) < .85:
                    self.deliveries += 1
                    outcome = 1.
                    self.events.append('Material delivered. Reinforcement disabled; no dopamine pulse.')
                else:
                    self.events.append('Material dropped at the current position.')
            elif math.hypot(self.x-self.material['x'], self.y-self.material['y']) < .7:
                self.cargo = True
                self.events.append('Material picked up by the motor decoder.')
        self.trajectory.append([self.x, self.y])
        return outcome


class EmbodiedBrain:
    def __init__(self, graph_root=GRAPH, device='cuda'):
        self.device = torch.device(device)
        self.meta = json.loads((graph_root/'metadata.json').read_text())
        with np.load(graph_root/'graph.npz') as graph:
            self.ids = graph['body_ids'].copy()
            crow, col, values = graph['crow'].copy(), graph['col'].copy(), graph['values'].copy()
        self.n = len(self.ids)
        self.initial_hash = hashlib.sha256(values.tobytes()).hexdigest()
        self.source_values = values
        self.wiring = torch.sparse_csr_tensor(torch.as_tensor(crow, device=self.device),
            torch.as_tensor(col, device=self.device), torch.as_tensor(values.copy(), device=self.device),
            size=(self.n, self.n), check_invariants=True)
        annotations = feather.read_feather(graph_root/'annotations.feather').set_index('bodyId').reindex(self.ids)
        self.annotations = annotations
        self.types = annotations['type'].fillna('').to_numpy()
        side = annotations['rootSide'].fillna(annotations['somaSide']).fillna('').to_numpy()
        superclass, classes = annotations['superclass'].fillna(''), annotations['class'].fillna('')
        indices, channels = [], []
        def connect(mask, choices):
            selected = np.flatnonzero(np.asarray(mask))
            indices.extend(selected.tolist())
            channels.extend([choices[i % len(choices)] for i in range(len(selected))])
        for label, offset in [('L', 0), ('R', 12)]:
            connect((superclass == 'ol_sensory') & (side == label), list(range(offset, offset+12)))
            connect((classes == 'olfactory') & (side == label), [24 if label == 'L' else 25])
        connect(superclass.str.contains('sensory') & classes.str.contains('mechanosensory'), [26,28,29])
        connect(superclass.str.contains('sensory') & (classes == 'gustatory'), [27])
        assert len(set(indices)) == len(indices), 'A sensory neuron was mapped twice.'
        self.sensory_indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        self.sensory_channels = torch.tensor(channels, device=self.device, dtype=torch.long)
        self.motor_groups = {}
        for name, cell_type, hemisphere in [('forward','DNg100',None),('left','DNa02','L'),
                                          ('right','DNa02','R'),('interact','MN9',None)]:
            chosen = np.flatnonzero((self.types == cell_type) & ((side == hemisphere) if hemisphere else True))
            if not len(chosen):
                raise RuntimeError(f'Missing annotated motor output: {name}')
            self.motor_groups[name] = torch.tensor(chosen, device=self.device)
        self.dopamine_groups = {name: torch.tensor(np.flatnonzero(self.types == cell_type), device=self.device)
                                for name, cell_type in [('appetitive','PAM01'),('aversive','PPL101')]}
        assert all(len(group) for group in self.dopamine_groups.values())
        # Retain the original edge matrix. Dopamine is not treated as an ordinary
        # fast excitatory input: separate the contributions into modulatory channels.
        nt = feather.read_feather(graph_root/'neurotransmitters.feather').set_index('body').reindex(self.ids)
        dopamine = nt.consensus_nt.fillna(nt.predicted_nt).fillna('') == 'dopamine'
        self.fast_mask = torch.as_tensor((~dopamine).to_numpy(np.float32), device=self.device)[:, None]
        self.dopamine_count = int(dopamine.sum())
        self.reward_enabled = False
        self.plasticity_enabled = False
        self.updates = 0
        self.reward_pulses = 0
        self.state = torch.zeros((self.n,1), device=self.device)
        rng = np.random.default_rng(SEED)
        # Fixed heterogeneous tonic drive; not optimized, no online calibration.
        self.tonic = torch.tensor(rng.uniform(.003,.007,(self.n,1)).astype(np.float32), device=self.device)
        self.last_dopamine = {'appetitive':0., 'aversive':0.}
        self.initial_state_hash = hashlib.sha256(self.state.cpu().numpy().tobytes()).hexdigest()

    def reinforcement_drive(self, outcome):
        # Future reinforcement enters ONLY these annotated modulatory populations.
        # Both gates are hard disabled in this observation-only executable.
        drive = torch.zeros_like(self.state)
        self.last_dopamine = {'appetitive':0., 'aversive':0.}
        if self.reward_enabled:
            raise RuntimeError('Reinforcement is intentionally locked off in the untrained experiment.')
        return drive

    @torch.inference_mode()
    def forward(self, observation, outcome=0.):
        assert not self.plasticity_enabled and self.updates == 0
        obs = torch.as_tensor(observation, device=self.device)
        drive = torch.zeros_like(self.state)
        drive[self.sensory_indices,0] = .5 * obs[self.sensory_channels]
        drive += self.reinforcement_drive(outcome)
        for _ in range(4):
            recurrent = torch.sparse.mm(self.wiring, self.state * self.fast_mask)
            target = torch.tanh(torch.relu(1.5*recurrent + drive + self.tonic))
            self.state.mul_(.75).add_(target, alpha=.25)
        outputs = torch.stack([self.state[index,0].mean() for index in self.motor_groups.values()]).cpu().numpy()
        rates = dict(zip(self.motor_groups, map(float,outputs)))
        # Explicit fixed engineering gains, not a trained actor and not natural
        # motor semantics. Zero neural output yields zero movement and no interaction.
        motor = {'speed':float(np.clip(80*rates['forward'],0,2)),
                 'turn':float(np.clip(160*(rates['right']-rates['left']),-2,2)),
                 'interact':bool(rates['interact']>.025), 'rates':rates}
        return motor

    def audit(self):
        current_hash = hashlib.sha256(self.wiring.values().detach().cpu().numpy().tobytes()).hexdigest()
        return {'initialWeightHash':self.initial_hash, 'currentWeightHash':current_hash,
                'unchanged':current_hash == self.initial_hash, 'checkpointLoaded':False,
                'optimizerPresent':False, 'learningUpdates':self.updates,
                'reinforcementEnabled':self.reward_enabled, 'plasticityEnabled':self.plasticity_enabled}


class Experiment:
    def __init__(self):
        self.lock = threading.RLock()
        self.ready = False
        self.error = None
        self.running = False
        self.snapshot = {}
        self.history = deque(maxlen=360)
        self.brain = None

    def reset(self):
        self.brain.state.zero_()
        self.plane = Plane()
        self.steps = 0
        self.run_id = str(uuid.uuid4())
        self.history.clear()
        self.last_motor = {'speed':0., 'turn':0., 'interact':False, 'rates':dict.fromkeys(self.brain.motor_groups,0.)}
        self.last_observation = self.plane.sensory()
        self.outcome = 0.
        self.run_started = time.perf_counter()
        self.last_publish = 0.
        self.tick_seconds = 0.
        self.publish()

    def load(self):
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('Laptop CUDA GPU unavailable; no remote or fake fallback is used.')
            torch.set_num_threads(4)
            self.brain = EmbodiedBrain()
            manifest = json.loads((PROJECT/'public/anatomy/manifest.json').read_text())
            ranks = {int(value):i for i,value in enumerate(self.brain.ids)}
            self.view_nodes = [{'id':n['id'],'group':0 if 'sensory' in n['superclass'] else 2 if ('motor' in n['superclass'] or n['superclass']=='descending_neuron') else 1,
                                'position': n['soma'] or [0,0,0]} for n in manifest['neurons'] if n['id'] in ranks]
            self.view_indices = torch.tensor([ranks[n['id']] for n in self.view_nodes], device=self.brain.device)
            with self.lock:
                self.reset()
                self.ready = True
                self.running = True
                self.publish()
            STORAGE.mkdir(parents=True, exist_ok=True)
            (STORAGE/'initial-audit.json').write_text(json.dumps(self.brain.audit(),indent=2))
            print(f'UNTRAINED_PLANE_READY {self.brain.n} neurons on {torch.cuda.get_device_name()}; learning LOCKED OFF',flush=True)
        except Exception as exc:
            self.error = str(exc)
            self.running = False
            print(f'Plane failed: {exc}',flush=True)

    def publish(self):
        p, b = self.plane, self.brain
        activity = b.state[self.view_indices,0].cpu().numpy()
        neural_mean = float(b.state.mean().item())
        self.snapshot = {'experiment':'untrained-plane-v1','runId':self.run_id,'ready':self.ready,
            'running':self.running,'learning':False,'rewardEnabled':False,'updates':0,'reward':0.,'steps':self.steps,
            'simSeconds':self.steps*DT,'tickMilliseconds':self.tick_seconds*1000,'device':torch.cuda.get_device_name(),
            'gpuMemoryGB':torch.cuda.memory_allocated()/1e9,'error':self.error,
            'brain':{'neurons':b.n,'edges':b.meta['edges'],'sensory':len(b.sensory_indices),
                     'meanActivity':neural_mean,'weightChange':0,'checkpointLoaded':False,
                     'dopamineNeurons':b.dopamine_count,'initialWeightHash':b.initial_hash},
            'avatar':{'x':p.x,'y':p.y,'heading':p.heading,'cargo':p.cargo,'contact':p.contact,'distance':p.distance,
                      'collisions':p.collisions,'deliveries':p.deliveries,**self.last_motor},
            'plane':{'width':p.width,'height':p.height,'sensingRange':p.sensing_range,'landmarks':p.landmarks,
                     'obstacles':p.obstacles,'trajectory':list(p.trajectory)},
            'sensory':self.last_observation.tolist(), 'history':list(self.history),'events':list(p.events),
            'motorNeurons':{key:[int(b.ids[i]) for i in indices.cpu().tolist()] for key,indices in b.motor_groups.items()},
            'dopamine':{'externalPulses':0,'appetitiveCells':len(b.dopamine_groups['appetitive']),
                        'aversiveCells':len(b.dopamine_groups['aversive']),'plasticity':'disabled',
                        'note':'No reward stimulation or synaptic updates. Anatomical dopamine cells still have model activity.'},
            'brainView':{'nodes':self.view_nodes,'edges':[],'activity':activity.tolist(),'bounds':[]}}
        self.last_publish = time.perf_counter()

    def loop(self):
        self.load()
        while True:
            began = time.perf_counter()
            with self.lock:
                if self.ready and self.running:
                    try:
                        self.last_observation = self.plane.sensory()
                        self.last_motor = self.brain.forward(self.last_observation, self.outcome)
                        self.outcome = self.plane.advance(self.last_motor)
                        self.steps += 1
                        self.tick_seconds = time.perf_counter()-began
                        if self.steps % 20 == 0:
                            self.history.append({'seconds':self.steps*DT,'activity':float(self.brain.state.mean().item()),
                                'speed':self.last_motor['speed'],'turn':self.last_motor['turn'],'reward':0.})
                        if time.perf_counter()-self.last_publish > .2:
                            self.publish()
                    except Exception as exc:
                        self.error = str(exc)
                        self.running = False
                        self.publish()
            time.sleep(max(.001,DT-(time.perf_counter()-began)))


experiment = Experiment()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=experiment.loop,daemon=True,name='untrained-laptop-plane').start()
    yield


app = FastAPI(title='Untrained 2D connectome · laptop only',lifespan=lifespan)


@app.middleware('http')
async def protect(request, call_next):
    origin = request.headers.get('origin','')
    if request.url.hostname not in ('127.0.0.1','localhost') or (origin and origin not in ORIGINS):
        return JSONResponse({'detail':'Origin not allowed'},status_code=403)
    if request.method == 'OPTIONS':
        response = Response(status_code=204)
    else:
        response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    if origin:
        response.headers.update({'Access-Control-Allow-Origin':origin,'Vary':'Origin',
            'Access-Control-Allow-Methods':'GET, POST, OPTIONS','Access-Control-Allow-Headers':'Content-Type',
            'Access-Control-Allow-Private-Network':'true'})
    return response


@app.get('/api/plane/health')
def health():
    return {'ready':experiment.ready,'error':experiment.error,'experiment':'untrained-plane-v1','learning':False}


@app.get('/api/plane/state')
def state():
    if not experiment.ready:
        raise HTTPException(503,experiment.error or 'Loading original connectome on the laptop GPU…')
    return experiment.snapshot


@app.get('/api/plane/audit')
def audit():
    if not experiment.ready:
        raise HTTPException(503,'Brain not ready')
    with experiment.lock:
        return experiment.brain.audit()


@app.post('/api/plane/control')
async def control(request: Request):
    value = await request.json()
    if not isinstance(value,dict) or set(value) != {'action'} or value['action'] not in ('start','pause','reset'):
        raise HTTPException(400,'Only start, pause, or reset is allowed. Learning cannot be enabled.')
    with experiment.lock:
        if not experiment.ready:
            raise HTTPException(503,'Brain not ready')
        if value['action'] == 'reset':
            experiment.reset()
        else:
            experiment.running = value['action']=='start'
        experiment.publish()
    return {'running':experiment.running,'learning':False,'runId':experiment.run_id}


if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8768,access_log=False,log_level='warning')
