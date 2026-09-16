"""Read-only analysis of unassisted physical evaluation event histories.

Short pickup/return cycles are observations, not proof that a return was wrong.
The analyser never supplies a teacher label or changes a simulation/candidate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def diagnose_trial(trial):
    if trial.get('interactionEventsTruncated') is not False:
        raise ValueError('Complete recorded event history required')
    events = trial['interactionEvents']
    counts = Counter(e['event'] for e in events)
    if any(counts[k] != v for k, v in trial['interactionCounts'].items()):
        raise ValueError('Event counts mismatch')
    if any(a['time'] > b['time'] for a, b in zip(events, events[1:])):
        raise ValueError('Events are not chronological')
    pending, returned, progressed = {}, [], []
    by_stage = {str(i): Counter() for i in (1, 2, 3)}
    rejected = Counter()
    for event in events:
        fly = event['fly']
        kind = event['event']
        if kind == 'pickup':
            if fly in pending or event['cargoBefore'] != 0 or event['cargoAfter'] not in (1, 2, 3):
                raise ValueError('Invalid pickup/cargo history')
            pending[fly] = (event['time'], event['cargoAfter'])
        elif kind in ('return', 'transfer', 'delivery'):
            if fly not in pending:
                raise ValueError('Cargo operation without a physical pickup')
            start, cargo = pending.pop(fly)
            if event['cargoBefore'] != cargo or event['cargoAfter'] != 0:
                raise ValueError('Cargo transition mismatch')
            elapsed = event['time']-start
            if elapsed < 0:
                raise ValueError('Negative carrying duration')
            if kind == 'return':
                returned.append(elapsed)
                by_stage[str(cargo)]['returned'] += 1
                by_stage[str(cargo)]['returnedWithin2Seconds'] += elapsed <= 2.00001
            else:
                progressed.append(elapsed)
                by_stage[str(cargo)]['forwardTransfers'] += 1
        elif kind == 'rejected':
            if event['boxDistance'] >= .8:
                rejected['outsideGripRange'] += 1
            elif event['cargoBefore'] and event['nearestBox'] not in (
                    event['cargoBefore'], event['cargoBefore']-1):
                rejected['nearNeitherDestinationNorReturnSource'] += 1
            else:
                # Do not infer exact stock/cooldown state from a 5-second trace.
                rejected['otherNearBoxReasonNotResolved'] += 1
        else:
            raise ValueError('Unknown event')
    return {'seed': trial['seed'], 'kind': trial['kind'], 'products': trial['products'],
            'sustainedSuccess': trial['sustainedSuccess'], 'pickups': counts['pickup'],
            'returns': len(returned), 'forwardTransfersIncludingDelivery': len(progressed),
            'stillCarryingAtEnd': len(pending),
            'returnsWithin2Seconds': sum(t <= 2.00001 for t in returned),
            'returnsWithin5Seconds': sum(t <= 5.00001 for t in returned),
            'medianReturnCarrySeconds': statistics.median(returned) if returned else None,
            'medianForwardCarrySeconds': statistics.median(progressed) if progressed else None,
            'stageCounts': {k: dict(v) for k, v in by_stage.items()},
            'rejected': dict(rejected)}


def diagnose_report(report):
    flags = ('teacher', 'learning', 'noise', 'injectedCargo', 'deliveryResets')
    if any(report.get(k) is not False for k in flags):
        raise ValueError('Only unassisted frozen evaluation may be analysed')
    if report.get('moveAtSeconds', 0):
        raise ValueError('Moved layouts need a separately segmented analysis')
    trials = [diagnose_trial(t) for t in report['trials']]
    fields = ('products', 'sustainedSuccess', 'pickups', 'returns',
              'forwardTransfersIncludingDelivery', 'returnsWithin2Seconds', 'returnsWithin5Seconds')
    families = {}
    for kind in sorted({t['kind'] for t in trials}):
        part = [t for t in trials if t['kind'] == kind]
        families[kind] = {'worlds': len(part), **{k: sum(t[k] for t in part) for k in fields}}
    return {'checkpointHash': report['checkpointHash'], 'fixedHash': report['fixedHash'],
            'analysisOnly': True, 'shortReturnsAreNotNecessarilyWrong': True,
            'families': families, 'trials': trials}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve previous diagnostics')
    results = []
    for path in args.reports:
        blob = path.read_bytes()
        results.append({'source': str(path), 'sourceSHA256': hashlib.sha256(blob).hexdigest(),
                        **diagnose_report(json.loads(blob))})
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps([{'source': r['source'], 'families': r['families']} for r in results]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
