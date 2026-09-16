"""Revision2: layout family and training situation are independent.

Version1 accidentally coupled normal/loaded/blocked starts to layout families,
and only practised an assembler blocked by stage2 carriers. Keep its source
and results for provenance; these new fixtures deliberately cover both buffers.
There is no change to inference inputs, motor control or physics.
"""
import math
import numpy as np
from .layout_recovery_teacher import HARD_SEEDS, KINDS
from .layout_recovery_world import RecoveryWorld

VERSION = 'independent-layout-stage-curriculum-v2'
STRATIFIED_VERSION = 'stratified-layout-stage-curriculum-v3'
# These prior held-out failures are now TRAINING data, not fresh verification.
RECOVERY_FAILURES = HARD_SEEDS + ((8800005, 'compact'), (8800006, 'compact'),
                                 (8801001, 'wide'), (8801003, 'wide'),
                                 (8801004, 'wide'), (8801005, 'wide'), (8801007, 'wide'))


def balanced_world(rng, index, *, fixed_kind=None):
    del index  # Never use the same index to choose both family and cargo state.
    if fixed_kind is not None and fixed_kind not in KINDS:
        raise ValueError('Unknown fixed training family')
    kind = str(rng.choice(KINDS)) if fixed_kind is None else fixed_kind
    seed = int(rng.integers(9200000, 9290000))
    failures = (RECOVERY_FAILURES if fixed_kind is None else
                tuple(case for case in RECOVERY_FAILURES if case[1] == fixed_kind))
    replay_failure = bool(rng.random() < .25 and failures)
    if replay_failure:
        seed, kind = failures[int(rng.integers(len(failures)))]
    scenario = str(rng.choice(('normal', 'normal', 'loaded', 'blocked')))
    random_starts = bool(rng.integers(2))
    world = RecoveryWorld(seed=seed, kind=kind, random_starts=random_starts)
    blocked_stage = None
    if scenario == 'loaded':
        cargos = rng.permutation(4)
        for agent, cargo in zip(world.agents, cargos):
            agent.cargo = int(cargo)
        for station in world.stations[1:3]:
            station['stock'] = int(rng.integers(3))
    elif scenario == 'blocked':
        blocked_stage = int(rng.integers(1, 3))
        world.stations[blocked_stage]['stock'] = 2
        target = world.stations[blocked_stage]
        angle_offset = float(rng.uniform(-math.pi, math.pi))
        for i, agent in enumerate(world.agents):
            # Keep mutually distinct cardinal offsets even beside a wall.
            # Offset0 is safe; arbitrary rotation may bunch clipped bodies.
            angle = i*math.pi/2
            agent.x = float(np.clip(target['x']+1.2*math.cos(angle), .23, 19.77))
            agent.y = float(np.clip(target['y']+1.2*math.sin(angle), .23, 13.77))
            agent.heading = math.atan2(target['y']-agent.y, target['x']-agent.x) + angle_offset
            agent.cargo = blocked_stage
            agent.trajectory.clear()
            agent.trajectory.append([agent.x, agent.y])
    world.refresh_landmarks()
    meta = {'seed': seed, 'kind': kind, 'scenario': scenario,
            'injectedCargo': scenario != 'normal', 'blockedStage': blocked_stage,
            'randomStarts': random_starts, 'replayedFailure': replay_failure,
            'curriculum': VERSION if fixed_kind is None else STRATIFIED_VERSION}
    if fixed_kind is not None:
        meta['familyStratified'] = True
    return world, meta


def episode_limit(metadata, normal_seconds):
    # Let ordinary worlds finish long trips; recovery fixtures can cycle faster.
    return min(normal_seconds, 90.) if metadata['scenario'] != 'normal' else normal_seconds
