"""Read-only matched frozen-service comparison, with complete event checks.

Never fit to these worlds or count a training diagnostic as held-out service.
Paired development comparisons guide selection, not universal-layout claims.
"""
import argparse
import json
from pathlib import Path
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_world import INTERFACE, PHYSICS
from .layout_service_diagnostics import diagnose_report
from .layout_recovery_eval import service_metrics, wilson
from .layout_recovery_protocol import atomic_json


def attest(report, manifest, expected_hash):
    for key in ('teacher', 'learning', 'noise', 'injectedCargo', 'deliveryResets'):
        if report.get(key) is not False:
            raise ValueError('Assisted or unfrozen evaluation')
    if (report['checkpointHash'] != expected_hash or manifest['checkpointHash'] != expected_hash
            or report['fixedHash'] != manifest['fixedHash'] or report['interface'] != INTERFACE
            or report['physics'] != PHYSICS or not report['interactionEventRecording']):
        raise ValueError('Wrong candidate, fixed interface, physics or missing events')
    if (report['seconds'] != manifest['seconds'] or report['randomStarts'] != manifest['random_starts']
            or report['moveAtSeconds'] != manifest['move_at'] or manifest['original_senses']
            or manifest['migrate'] or not manifest['interaction_events']):
        raise ValueError('Physical controls mismatch')
    kinds = manifest['kinds'].split(',')
    expected = {(kind, manifest['seed']+1000*j+i) for j, kind in enumerate(kinds)
                for i in range(manifest['cases'])}
    observed = [(t['kind'], t['seed']) for t in report['trials']]
    if len(observed) != len(set(observed)) or set(observed) != expected or report['batchActors'] != len(expected)*4:
        raise ValueError('Wrong or duplicate evaluation worlds')
    if report['moveAtSeconds']:
        raise ValueError('Moved-layout validation needs a separately segmented comparison')
    diagnostics = diagnose_report(report)
    for trial, diagnostic in zip(report['trials'], diagnostics['trials']):
        times = trial['deliveryTimes']
        if (any(not 0 < t <= report['seconds'] for t in times) or sorted(times) != times
                or len(times) != trial['products']
                or service_metrics(times, report['seconds'])['sustainedSuccess'] != trial['sustainedSuccess']
                or len(trial['initialAvatars']) != 4 or len(trial['initialPositions']) != 4
                or trial['initialPositions'] != trial['finalPositions']
                or not trial['trace'] or trial['trace'][-1]['time'] != report['seconds']
                or trial['trace'][-1]['products'] != trial['products']
                or trial['pickups'] != diagnostic['pickups'] or trial['returns'] != diagnostic['returns']
                or trial['transfers'] != diagnostic['forwardTransfersIncludingDelivery']):
            raise ValueError('Incorrect physical horizon, material account or sustained service')
    for kind in kinds:
        subset = [t for t in report['trials'] if t['kind'] == kind]
        successes = sum(t['sustainedSuccess'] for t in subset)
        actual = {'cases': len(subset), 'products': sum(t['products'] for t in subset),
                  'worldsWithProduct': sum(t['products'] > 0 for t in subset),
                  'worldsWithThree': sum(t['products'] >= 3 for t in subset),
                  'sustainedSuccesses': successes, 'wilson95': wilson(successes, len(subset))}
        if report['summary'][kind] != actual:
            raise ValueError('Reported summary disagrees with complete worlds')
    return diagnostics


def compare(candidate, candidate_manifest, baseline, baseline_manifest, expected_hash):
    diagnostic = attest(candidate, candidate_manifest, expected_hash)
    attest(baseline, baseline_manifest, baseline['checkpointHash'])
    controls = ('fixedHash', 'seconds', 'randomStarts', 'moveAtSeconds', 'device', 'physics', 'interface', 'batchActors')
    if any(candidate[k] != baseline[k] for k in controls):
        raise ValueError('Unmatched device, body or experiment controls')
    if candidate_manifest['sourceHash'] != baseline_manifest['sourceHash']:
        raise ValueError('Different evaluator source')
    old = {(t['kind'], t['seed']): t for t in baseline['trials']}
    rows = []
    for trial in candidate['trials']:
        key = trial['kind'], trial['seed']
        if key not in old or any(trial[k] != old[key][k] for k in ('initialPositions', 'initialAvatars')):
            raise ValueError('Mismatched initial world/avatars')
        rows.append({'kind': key[0], 'seed': key[1], 'baselineProducts': old[key]['products'],
                     'candidateProducts': trial['products'], 'baselineSustained': old[key]['sustainedSuccess'],
                     'candidateSustained': trial['sustainedSuccess']})
    if len(old) != len(rows):
        raise ValueError('Unmatched number of worlds')
    return {'candidateHash': expected_hash, 'baselineHash': baseline['checkpointHash'],
            'matchedControlsAndFullEventAccountingPassed': True, 'sourceHash': candidate_manifest['sourceHash'],
            'candidateSummary': candidate['summary'], 'baselineSummary': baseline['summary'],
            'gainedWorlds': sum(r['candidateSustained'] and not r['baselineSustained'] for r in rows),
            'lostWorlds': sum(r['baselineSustained'] and not r['candidateSustained'] for r in rows),
            'pairedTrials': rows, 'candidateDiagnostics': diagnostic,
            'developmentSelectionNotFreshConfirmation': True, 'universalLayoutClaim': False}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve previous comparisons')
    folders = (args.candidate_eval, args.baseline_eval)
    c, b = [json.loads((p/'evaluation.json').read_text()) for p in folders]
    cm, bm = [json.loads((p/'manifest.json').read_text()) for p in folders]
    result = compare(c, cm, b, bm, args.expected_parameter_hash)
    if cm['sourceHash'] != file_hash(Path(__file__).parent/'layout_recovery_eval.py'):
        raise ValueError('Published evaluator source differs from local original')
    result['inputHashes'] = {str(p/n): file_hash(p/n) for p in folders for n in ('manifest.json', 'evaluation.json')}
    result['comparisonSourceHash'] = file_hash(Path(__file__))
    atomic_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('pairedTrials', 'candidateDiagnostics', 'inputHashes')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('candidate-eval', 'baseline-eval', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-parameter-hash', required=True)
    main(p.parse_args())
