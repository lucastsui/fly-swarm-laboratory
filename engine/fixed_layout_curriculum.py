"""Train on an immutable, user-selected layout without changing the senses/body."""
import hashlib
import json
import math
import numpy as np
from .layout_world import validate_layout
from .layout_recovery_world import RecoveryWorld, PHYSICS

SCHEMA = 'fixed-dashboard-layout-v1'
TRANSITION_VERSION = 'fixed-layout-physical-transition-practice-v2'


def load_layout(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema') != SCHEMA or value.get('physics') != PHYSICS:
        raise ValueError('Unexpected layout schema or physics')
    if value.get('flies') != 4 or value.get('collisions') is not True:
        raise ValueError('Expected four colliding flies')
    if value.get('stationOrder') != ['raw', 'smelter', 'assembler', 'finished']:
        raise ValueError('Station identities must not be reordered')
    positions = validate_layout(value['positions'])
    fingerprint = hashlib.sha256(np.asarray(positions, dtype='<f8').tobytes()).hexdigest()
    return positions, fingerprint


def fixed_training_world(rng, index, positions, layout_hash):
    seed = int(rng.integers(12000000, 12900000))
    world = RecoveryWorld(seed=seed, positions=positions, kind='fixed-current',
                          flies=4, continuous=True, random_starts=index % 3 == 1)
    scenario = 'normal-empty-start'
    # A quarter of TRAINING episodes practice all three loaded legs. These
    # explicit fixtures never appear in independent service evaluations.
    if index % 4 == 3:
        scenario = 'injected-loaded-legs-training-only'
        for i, agent in enumerate(world.agents):
            agent.cargo = 1+i % 3
            station = world.stations[agent.cargo-1]
            for _ in range(10000):
                angle = rng.uniform(-math.pi, math.pi)
                x, y = station['x']+1.0*math.cos(angle), station['y']+1.0*math.sin(angle)
                if (.23 <= x <= 19.77 and .23 <= y <= 13.77
                        and all(math.hypot(x-b.x, y-b.y) > .48 for b in world.agents[:i])):
                    break
            else:
                raise ValueError('Cannot initialize separated training bodies')
            agent.x, agent.y = float(x), float(y)
            agent.heading = float(rng.uniform(-math.pi, math.pi))
            agent.trajectory.clear(); agent.trajectory.append([agent.x, agent.y])
    world.refresh_landmarks()
    return world, {'seed': seed, 'kind': 'fixed-current', 'scenario': scenario,
                   'layoutHash': layout_hash, 'positions': positions.tolist()}


def transition_training_world(rng, index, positions, layout_hash):
    """TRAINING-only fixtures; every subsequent action is the brain's.

    Cycle all three stages through pickup, departure and delivery, mixed with
    normal empty factories. No target signal is added to the fixed senses.
    """
    phase, stage = index % 4, (index // 4) % 3
    seed = int(rng.integers(12000000, 12900000))
    world = RecoveryWorld(seed=seed, positions=positions, kind='fixed-current',
                          flies=4, continuous=True, random_starts=phase == 0 and index % 8 == 4)
    scenario = 'normal-empty-start'
    if phase:
        scenario = ('pickup', 'loaded-departure', 'loaded-delivery')[phase-1] + '-training-only'
        for station in world.stations:
            station['stock'] = station['returned'] = 0
            station['timer'] = 0.
        world.stations[0]['stock'] = 3
        if phase == 1:
            world.stations[stage]['stock'] = 3 if stage == 0 else 1
        center = world.stations[stage + int(phase == 3)]
        for i, agent in enumerate(world.agents):
            agent.cargo = 0 if phase == 1 else stage+1
            for _ in range(10000):
                angle, distance = rng.uniform(-math.pi, math.pi), rng.uniform(.48, 1.15)
                x, y = center['x']+distance*math.cos(angle), center['y']+distance*math.sin(angle)
                if (.23 <= x <= 19.77 and .23 <= y <= 13.77
                        and all(math.hypot(x-b.x, y-b.y) > .5 for b in world.agents[:i])):
                    break
            else:
                raise ValueError('Cannot initialize separated practice bodies')
            agent.x, agent.y = float(x), float(y)
            agent.heading = float(math.atan2(center['y']-y, center['x']-x)+rng.uniform(-.7, .7))
            agent.trajectory.clear(); agent.trajectory.append([agent.x, agent.y])
    world.refresh_landmarks()
    return world, {'seed': seed, 'kind': 'fixed-current', 'scenario': scenario,
                   'layoutHash': layout_hash, 'positions': positions.tolist(),
                   'curriculum': TRANSITION_VERSION, 'practiceStage': stage if phase else None,
                   'trainingOnlyFixture': bool(phase), 'teacherActions': False}
