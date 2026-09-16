"""Offline teacher-controlled TRAINING trajectories; never brain evidence.

This is deliberately separate from the learner-only cluster packet protocol.
No brain or optimizer is loaded. The real collision/processing/return physics
produces every observation, cargo transition and subsequent training target.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .layout_interaction_trace import before_interaction, interaction_events
from .layout_recovery_protocol import atomic_json
from .layout_recovery_teacher import LocalTeacher, KINDS, LABEL_VERSION
from .layout_recovery_world import RecoveryWorld, CHANNELS, INTERFACE, PHYSICS
from .plastic_brain import DT

SCHEMA = 'physical-teacher-demonstrations-v1'
CONTROL = 'training-teacher-demonstration'
TRAIN_SEED_LOW, TRAIN_SEED_HIGH = 9360000, 9370000


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def motors_from_labels(labels):
    # Raw motor targets have MN9 mean multiplied by 40, so 1 == .025.
    return [{'speed': float(v[0]), 'turn': float(v[1]), 'interact': bool(v[2] > 1.)}
            for v in labels]


def body_state(world):
    return np.asarray([[a.x, a.y, a.heading, a.speed, a.turn, a.cargo]
                       for a in world.agents], np.float32)


def collect_episode(seed, kind, frames, random_starts=False, progress=False):
    if not TRAIN_SEED_LOW <= seed < TRAIN_SEED_HIGH:
        raise ValueError('Use the dedicated demonstration TRAINING seed range')
    if kind not in KINDS or frames < 1:
        raise ValueError('Invalid demonstration bounds')
    world = RecoveryWorld(seed, kind=kind, random_starts=random_starts)
    teachers = [LocalTeacher() for _ in world.agents]
    x = np.empty((frames, 4, CHANNELS), np.float32)
    y = np.empty((frames, 4, 3), np.float32)
    bodies = np.empty((frames+1, 4, 6), np.float32)
    starts = np.zeros(frames, np.bool_)
    starts[0] = True
    bodies[0] = body_state(world)
    events, times = [], []
    counts = dict.fromkeys(('pickup', 'return', 'transfer', 'delivery', 'rejected'), 0)
    positions = [[s['x'], s['y']] for s in world.stations]
    began = time.perf_counter()
    for tick in range(frames):
        x[tick] = world.sensory()
        y[tick] = [teacher.label(agent) for teacher, agent in zip(teachers, world.agents)]
        previous = before_interaction(world)
        old = world.deliveries
        world.advance(motors_from_labels(y[tick]))
        bodies[tick+1] = body_state(world)
        for event in interaction_events(world, previous, tick+1):
            events.append(event)
            counts[event['event']] += 1
        times.extend([(tick+1)*DT]*(world.deliveries-old))
        if progress and (tick+1) % 1200 == 0:
            print(json.dumps({'trainingDemonstration': True, 'seed': seed, 'kind': kind,
                              'simulationSeconds': (tick+1)*DT,
                              'teacherProductsNotBrainEvidence': world.deliveries,
                              'wallSeconds': time.perf_counter()-began}), flush=True)
    meta = {'schema': SCHEMA, 'control': CONTROL, 'interface': INTERFACE, 'physics': PHYSICS,
            'teacherActions': True, 'brainWasUsed': False, 'isBrainEvidence': False,
            'optimizerUsed': False, 'injectedCargo': False, 'deliveryResets': False,
            'seed': seed, 'kind': kind, 'randomStarts': random_starts,
            'frames': frames, 'dt': DT, 'labelVersion': LABEL_VERSION,
            'initialPositions': positions, 'teacherProductsNotBrainEvidence': world.deliveries,
            'deliveryTimes': times, 'interactionCounts': counts, 'interactionEvents': events,
            'interactionEventsTruncated': False, 'bumpEvents': world.bump_events,
            'finalStock': [s['stock'] for s in world.stations],
            'finalReturned': [s['returned'] for s in world.stations],
            'neuralStatePolicy': 'Re-encode from episode start, or use a checkpoint-versioned '
                                 'prefix state; never reset neural state at arbitrary chunk boundaries.'}
    arrays = {'observations': x, 'labels': y, 'bodies': bodies, 'episode_starts': starts}
    validate_episode(arrays, meta)
    return arrays, meta


def validate_episode(arrays, meta):
    expected = {'schema': SCHEMA, 'control': CONTROL, 'interface': INTERFACE, 'physics': PHYSICS,
                'teacherActions': True, 'brainWasUsed': False, 'isBrainEvidence': False,
                'optimizerUsed': False, 'injectedCargo': False, 'deliveryResets': False,
                'labelVersion': LABEL_VERSION, 'dt': DT}
    if any(meta.get(k) != value for k, value in expected.items()):
        raise ValueError('Wrong training demonstration provenance')
    if not TRAIN_SEED_LOW <= meta.get('seed', -1) < TRAIN_SEED_HIGH or meta.get('kind') not in KINDS:
        raise ValueError('Wrong training seed/family')
    frames = meta['frames']
    if not isinstance(frames, int) or frames < 1:
        raise ValueError('Invalid frame count')
    shapes = {'observations': (frames, 4, CHANNELS), 'labels': (frames, 4, 3),
              'bodies': (frames+1, 4, 6), 'episode_starts': (frames,)}
    if set(arrays) != set(shapes):
        raise ValueError('Wrong demonstration array keys')
    for key, shape in shapes.items():
        value = arrays[key]
        dtype = np.bool_ if key == 'episode_starts' else np.float32
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError('Invalid demonstration array '+key)
    if not arrays['episode_starts'][0] or arrays['episode_starts'][1:].any():
        raise ValueError('Unexpected mid-episode reset')
    if arrays['bodies'][0, :, 5].any():
        raise ValueError('Demonstrations must start with genuinely empty cargo')


def load_episode(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        arrays = {k: archive[k].copy() for k in archive.files if k != 'metadata'}
    validate_episode(arrays, meta)
    return arrays, meta


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve prior training data')
    if args.seconds < 1 or not 1 <= args.cases <= 1000:
        raise ValueError('Invalid dataset bounds')
    if not TRAIN_SEED_LOW <= args.seed or args.seed+3000+args.cases > TRAIN_SEED_HIGH:
        raise ValueError('Dataset would leave its dedicated training seed range')
    args.out.mkdir(parents=True)
    engine = Path(__file__).parent
    files = ('layout_recovery_demonstrations.py', 'layout_recovery_teacher.py',
             'layout_recovery_world.py', 'layout_world.py', 'swarm_world.py', 'haul_world.py',
             'layout_interaction_trace.py')
    manifest = {'schema': SCHEMA, 'control': CONTROL, 'isBrainEvidence': False,
                'teacherActions': True, 'brainWasUsed': False, 'optimizerUsed': False,
                'interface': INTERFACE, 'physics': PHYSICS, 'seconds': args.seconds,
                'casesPerFamily': args.cases, 'randomStartsPolicy': 'alternating within each family',
                'sourceHashes': {name: file_hash(engine/name) for name in files},
                'finished': False, 'episodes': []}
    atomic_json(args.out/'manifest.json', manifest)
    for j, kind in enumerate(KINDS):
        for i in range(args.cases):
            seed = args.seed+j*1000+i
            arrays, meta = collect_episode(seed, kind, round(args.seconds/DT),
                                           random_starts=bool(i % 2), progress=True)
            filename = f'{kind}-{seed}.npz'
            with (args.out/filename).open('xb') as stream:
                np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(meta)))
            manifest['episodes'].append({'file': filename, 'sha256': file_hash(args.out/filename),
                                         **{k: v for k, v in meta.items() if k != 'interactionEvents'}})
            atomic_json(args.out/'manifest.json', manifest)
    manifest['finished'] = True
    atomic_json(args.out/'manifest.json', manifest)
    print('TRAINING_DATA_COMPLETE '+json.dumps({'episodes': len(manifest['episodes']),
                                               'isBrainEvidence': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=TRAIN_SEED_LOW)
    parser.add_argument('--cases', type=int, default=2)
    parser.add_argument('--seconds', type=int, default=600)
    main(parser.parse_args())
