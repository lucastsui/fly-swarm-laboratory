"""Frozen time-course diagnostic on previously attested physical TRAINING data.

An updated candidate uses the original parent prefix, as in the microfit;
this is explicitly NOT a new physical rollout or exact updated full history.
No labels, inputs, parameters, physics or runtime controller are changed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_demonstration_train import DemonstrationCache
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json
from .layout_recovery_teacher import KINDS
from .supervised_joint import raw_readout


def attested_batch(bank, proof, device):
    expected = {'parametersUnchanged': True, 'inputsAndPrefixUnchanged': True,
                'optimizerUsed': False, 'candidateSaved': False, 'physicalWorldAdvanced': False,
                'isServiceEvidence': False, 'parameterHash': bank.manifest['parameterHash'],
                'fixedHash': bank.manifest['fixedHash']}
    if any(proof.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong frozen physical-actor attestation')
    if proof['physicalTrainingData']['datasetManifestHash'] != bank.manifest['datasetManifestHash']:
        raise ValueError('Physical data mismatch')
    parts = [[], [], []]
    seen = set()
    for selection in proof['selectedWindows']:
        index, fly = selection['windowIndex'], selection['fly']
        if type(index) is not int or not 0 <= index < len(bank.x) or type(fly) is not int or fly not in range(4):
            raise ValueError('Invalid physical actor index')
        spec = bank.manifest['windows'][index]
        if any(selection.get(k) != v for k, v in spec.items()):
            raise ValueError('Selected history differs from attested prefix')
        key = (spec['kind'], spec['category'])
        tick = selection['eventTick']-spec['start']
        if key in seen or not 0 <= tick < len(bank.x[index]) or bank.y[index, tick, fly, 2] <= 1.:
            raise ValueError('Invalid or duplicate pickup history')
        seen.add(key)
        for bucket, array in zip(parts, (bank.x, bank.y, bank.states)):
            bucket.append(array[index, :, fly:fly+1])
    if seen != {(kind, 'pickup-'+str(stage)) for kind in KINDS for stage in (1, 2, 3)}:
        raise ValueError('All twelve original family/stage histories required')
    return tuple(torch.as_tensor(np.concatenate(part, axis=1), device=device) for part in parts)


@torch.no_grad()
def predict_all(model, x, state):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Read-only frozen diagnostic required')
    weights, predictions = model.weights(), []
    for frame in range(len(x)):
        _, state = model(x[frame], 4, state, weights)
        predictions.append(raw_readout(model, state))
    result = torch.stack(predictions)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite frozen response')
    return result.cpu().numpy()


def analyze(x, labels, predictions, selections, dt=.05):
    if x.shape[:2] != labels.shape[:2] or labels.shape != predictions.shape:
        raise ValueError('Aligned input/label/prediction histories required')
    result = []
    for actor, selection in enumerate(selections):
        event = selection['eventTick']-selection['start']
        end = min(len(x), event+21)
        positive = labels[:, actor, 2] > 1.
        if not positive[event]:
            raise ValueError('Event must carry the actual positive teacher label')
        # Last onset of the legacy taste signal at/before this physical event.
        taste = x[:, actor, 27] > .5
        transitions = np.flatnonzero(taste & ~np.r_[False, taste[:-1]])
        transitions = transitions[transitions <= event]
        onset = int(transitions[-1]) if len(transitions) and taste[event] else None
        on = np.flatnonzero(predictions[event:end, actor, 2] > 1.)
        pulse_end = event+1
        while pulse_end < len(positive) and positive[pulse_end]:
            pulse_end += 1
        rows = []
        for frame in range(max(0, event-8), end):
            rows.append({'secondsFromPickupLabel': round((frame-event)*dt, 5),
                         'legacyTaste': float(x[frame, actor, 27]),
                         'cargo': x[frame, actor, 126:129].tolist(),
                         'teacherGrip': float(labels[frame, actor, 2]),
                         'rawGrip': float(predictions[frame, actor, 2]),
                         'speed': float(predictions[frame, actor, 0]),
                         'turn': float(predictions[frame, actor, 1])})
        result.append({'selection': selection, 'teacherPulseSecondsFromEvent': (pulse_end-event)*dt,
                       'tasteOnsetSecondsBeforeEvent': (event-onset)*dt if onset is not None else None,
                       'gripAtTeacherEvent': float(predictions[event, actor, 2]),
                       'firstGripThresholdSecondsAfterEvent': float(on[0]*dt) if len(on) else None,
                       'observedSecondsAfterEvent': (end-event-1)*dt, 'timeline': rows})
    return result


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostics')
    torch.set_num_threads(4)
    proof = json.loads(args.proof.read_text())
    cache_manifest = json.loads((args.cache/'manifest.json').read_text())
    if file_hash(args.cache/'manifest.json') != proof['cacheManifestHash']:
        raise ValueError('Cache/proof identity mismatch')
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    before, source_file = model.checkpoint_hash(), file_hash(args.candidate)
    bank = DemonstrationCache(args.cache, proof['parameterHash'], model.fixed_hash, 64, 32, model.n)
    x, y, state = attested_batch(bank, proof, model.tonic.device)
    prediction = predict_all(model, x, state)
    rows = analyze(x.cpu().numpy(), y.cpu().numpy(), prediction, proof['selectedWindows'])
    assert model.checkpoint_hash() == before and model.fingerprint() == model.fixed_hash
    assert file_hash(args.candidate) == source_file
    result = {'parameterHash': before, 'candidateFileHash': source_file, 'fixedHash': model.fixed_hash,
              'prefixParameterHash': cache_manifest['parameterHash'],
              'prefixMatchesEvaluatedCandidate': before == cache_manifest['parameterHash'],
              'prefixSemantics': 'original parent prefix; current frozen weights for the96 frames only',
              'physicalActorProofHash': file_hash(args.proof), 'cacheManifestHash': proof['cacheManifestHash'],
              'sourceHash': file_hash(Path(__file__)), 'parametersUnchanged': True,
              'optimizerUsed': False, 'physicalWorldAdvanced': False, 'isServiceEvidence': False,
              'trials': rows}
    atomic_json(args.out, result)
    print('TIMING_PROBE '+json.dumps([{k: v for k, v in r.items() if k not in ('timeline', 'selection')}
                                    for r in rows]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'proof', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
