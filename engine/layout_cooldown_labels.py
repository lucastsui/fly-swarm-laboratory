"""TRAINING-label slack only where the unchanged body's cooldown ignores grip.

Extend each successful teacher grip pulse by200ms. Replaying the modified
commands must exactly reproduce the original physical observations, bodies
and interaction events. This does NOT add a controller to brain inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .layout_recovery_demonstrations import body_state, motors_from_labels, file_hash
from .layout_demonstration_cache import load_dataset
from .layout_interaction_trace import before_interaction, interaction_events
from .layout_recovery_world import RecoveryWorld
from .layout_recovery_protocol import atomic_json
from .plastic_brain import DT

VERSION = 'physically-equivalent-cooldown-grip-pulse-v1'


def extend_labels(original, meta, extra_frames=4):
    if type(extra_frames) is not int or not 1 <= extra_frames <= 6:
        raise ValueError('Bounded50..300ms pulse extension required')
    if original.shape != (meta['frames'], 4, 3) or meta['dt'] != DT:
        raise ValueError('Original episode label shape/timing required')
    revised = original.copy()
    for event in meta['interactionEvents']:
        if event['event'] not in ('pickup', 'transfer', 'delivery', 'return'):
            continue
        tick = round(event['time']/DT)-1
        fly = event['fly']
        if not 0 <= tick < len(original) or type(fly) is not int or fly not in range(4) or original[tick, fly, 2] <= 1.:
            raise ValueError('Successful event lacks original grip activation')
        revised[tick+1:min(len(original), tick+extra_frames+1), fly, 2] = 2.2
    if not np.array_equal(original[..., :2], revised[..., :2]):
        raise AssertionError('Motion labels may never change')
    return revised


def verify_equivalent_replay(arrays, meta, revised):
    original = arrays['labels']
    if (revised.shape != original.shape or not np.isfinite(revised).all()
            or not np.array_equal(revised[..., :2], original[..., :2])):
        raise ValueError('Only finite grip labels may change')
    changed = revised[..., 2] != original[..., 2]
    if np.any(changed & ((original[..., 2] > 1.) | (revised[..., 2] <= 1.))):
        raise ValueError('Only additional cooldown grip activations are permitted')
    world = RecoveryWorld(meta['seed'], kind=meta['kind'], random_starts=meta['randomStarts'])
    expected_events = {}
    for event in meta['interactionEvents']:
        expected_events.setdefault(round(event['time']/DT)-1, []).append(event)
    count = 0
    for tick in range(meta['frames']):
        if (not np.array_equal(world.sensory(), arrays['observations'][tick])
                or not np.array_equal(body_state(world), arrays['bodies'][tick])):
            raise ValueError('Modified labels changed recorded physical history')
        for fly in np.flatnonzero(changed[tick]):
            # interact() decrements the timer before testing it.
            if world.agents[fly].cooldown <= DT+1e-6:
                raise ValueError('Relabelled action is not protected by actual cooldown')
            count += 1
        previous = before_interaction(world)
        world.advance(motors_from_labels(revised[tick]))
        if (not np.array_equal(body_state(world), arrays['bodies'][tick+1])
                or interaction_events(world, previous, tick+1) != expected_events.get(tick, [])):
            raise ValueError('Modified labels changed bodies or actual interactions')
    if (world.deliveries != meta['teacherProductsNotBrainEvidence']
            or [s['stock'] for s in world.stations] != meta['finalStock']
            or [s['returned'] for s in world.stations] != meta['finalReturned']):
        raise ValueError('Modified labels changed final physical accounting')
    return {'seed': meta['seed'], 'kind': meta['kind'], 'frames': meta['frames'],
            'changedGripLabels': count, 'allChangesIgnoredByActualCooldown': True,
            'everyObservationBodyAndEventExactlyMatched': True, 'isServiceEvidence': False}


def build_selected_revision(dataset, proof, out, extra_frames=4):
    if out.exists():
        raise FileExistsError('Preserve label revisions')
    episodes, manifest = load_dataset(dataset)
    if any(Path(n).name != n or file_hash(Path(__file__).parent/n) != h for n, h in manifest['sourceHashes'].items()):
        raise ValueError('Original demonstration/physics source changed')
    if file_hash(dataset/'manifest.json') != proof['physicalTrainingData']['datasetManifestHash']:
        raise ValueError('Physical actor proof belongs to another dataset')
    if not proof['physicalTrainingData']['eventBodyTransitionsVerified']:
        raise ValueError('Original physical events must be attested')
    indexes = sorted({s['episode'] for s in proof['selectedWindows']})
    revised, audits = {}, []
    for index in indexes:
        arrays, meta = episodes[index]
        target = extend_labels(arrays['labels'], meta, extra_frames)
        audit = verify_equivalent_replay(arrays, meta, target)
        audits.append({'episode': index, **audit})
        revised[index] = target
        print('COOLDOWN_REPLAY '+json.dumps(audit), flush=True)
    original_parts, target_parts = [], []
    for s in proof['selectedWindows']:
        arrays, meta = episodes[s['episode']]
        if (meta['seed'] != s['seed'] or meta['kind'] != s['kind']
                or s['stop']-s['start'] != 96 or s['fly'] not in range(4)):
            raise ValueError('Selected physical history mismatch')
        original_parts.append(arrays['labels'][s['start']:s['stop'], s['fly']:s['fly']+1])
        target_parts.append(revised[s['episode']][s['start']:s['stop'], s['fly']:s['fly']+1])
    original, targets = [np.concatenate(v, axis=1) for v in (original_parts, target_parts)]
    out.mkdir(parents=True)
    with (out/'labels.npz').open('xb') as stream:
        np.savez_compressed(stream, original=original, revised=targets)
    result = {'version': VERSION, 'extraFrames': extra_frames, 'extraSeconds': extra_frames*DT,
              'proofSelections': proof['selectedWindows'], 'datasetManifestHash': file_hash(dataset/'manifest.json'),
              'originalDatasetFiles': manifest['episodes'], 'audits': audits,
              'labelsFileHash': file_hash(out/'labels.npz'), 'sourceHash': file_hash(Path(__file__)),
              'originalPhysicsSourceHashes': manifest['sourceHashes'],
              'originalPositiveLabels': int((original[64:, :, 2] > 1.).sum()),
              'revisedPositiveLabels': int((targets[64:, :, 2] > 1.).sum()),
              'physicalTrajectoriesUnchanged': True, 'teacherAtInference': False,
              'newRuntimeController': False, 'isServiceEvidence': False, 'finished': True}
    atomic_json(out/'manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('dataset', 'proof', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    build_selected_revision(args.dataset, json.loads(args.proof.read_text()), args.out)
