"""Read-only summaries of independent service tests; never controls a fly."""
import argparse
import hashlib
import json
from pathlib import Path


def trial_metrics(trial, seconds):
    deliveries = trial['deliveryTimes']
    if sorted(deliveries) != deliveries or any(not 0 <= t <= seconds for t in deliveries):
        raise ValueError('Invalid delivery times')
    if len(deliveries) != trial['products']:
        raise ValueError('Delivery count mismatch')
    events = trial.get('interactionEvents', [])
    last_pickup = {}
    returns = immediate = 0
    stage_transfers = {str(i): 0 for i in (1, 2, 3)}
    for event in events:
        fly = event['fly']
        if event['event'] == 'pickup':
            last_pickup[fly] = event
        elif event['event'] == 'return':
            returns += 1
            previous = last_pickup.get(fly)
            if previous and (previous['nearestBox'] == event['nearestBox']
                             and 0 <= event['time']-previous['time'] <= 1.2+1e-8):
                immediate += 1
        elif event['event'] in ('transfer', 'delivery'):
            stage_transfers[str(event['cargoBefore'])] += 1
    bounds = [0., *deliveries, seconds]
    contributions = trial['agentContributions']
    return {'seed': trial['seed'], 'products': trial['products'],
            'sustained': len(deliveries) >= 3 and any(t > seconds/2 for t in deliveries),
            'secondHalfProducts': sum(t > seconds/2 for t in deliveries),
            'productiveFlies': sum(a['transfers'] > 0 for a in contributions),
            'allFourMadeTransfers': len(contributions) == 4 and all(a['transfers'] > 0 for a in contributions),
            'longestNoDeliverySeconds': max(b-a for a, b in zip(bounds, bounds[1:])),
            'terminalNoDeliverySeconds': seconds-(deliveries[-1] if deliveries else 0.),
            'returnsTotal': trial['returns'], 'recordedReturns': returns,
            'recordedImmediatePickupReturns': immediate,
            'eventRecordingComplete': bool('interactionEvents' in trial and
                                           not trial.get('interactionEventsTruncated', False)),
            'recordedTransfersByStage': stage_transfers}


def review(report):
    for flag in ('teacher', 'learning', 'noise', 'injectedCargo', 'deliveryResets'):
        if report.get(flag) is not False:
            raise ValueError('Independent frozen evidence required: '+flag)
    if report.get('fixedLayoutOnly') is not True or report.get('moveAtSeconds') != 0:
        raise ValueError('This reviewer is for fixed-layout tests only')
    trials = [trial_metrics(t, report['seconds']) for t in report['trials']]
    if not trials:
        raise ValueError('Empty evaluation')
    total_returns = sum(t['recordedReturns'] for t in trials)
    immediate = sum(t['recordedImmediatePickupReturns'] for t in trials)
    return {'checkpointHash': report['checkpointHash'], 'layoutHash': report['layoutHash'],
            'fixedHash': report['fixedHash'], 'seconds': report['seconds'],
            'randomStarts': report['randomStarts'], 'cases': len(trials),
            'products': sum(t['products'] for t in trials),
            'sustained': sum(t['sustained'] for t in trials),
            'worldsWithProduct': sum(t['products'] > 0 for t in trials),
            'worldsAllFourContributed': sum(t['allFourMadeTransfers'] for t in trials),
            'recordedReturns': total_returns, 'recordedImmediateReturns': immediate,
            'recordedImmediateReturnFraction': immediate/total_returns if total_returns else None,
            'eventRecordingComplete': all(t['eventRecordingComplete'] for t in trials),
            'trials': trials}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve prior reviews')
    results = []
    for path in args.report:
        raw = path.read_bytes()
        results.append({'source': str(path.resolve()), 'sourceSha256': hashlib.sha256(raw).hexdigest(),
                        **review(json.loads(raw))})
    if len({(r['layoutHash'], r['fixedHash'], r['seconds']) for r in results}) != 1:
        raise ValueError('Do not mix task identities or test horizons')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'schema': 'fixed-layout-frozen-review-v1', 'results': results}, indent=2))
    print(json.dumps([{k: r[k] for k in ('checkpointHash', 'randomStarts', 'cases', 'products', 'sustained',
                      'worldsAllFourContributed', 'recordedImmediateReturnFraction')} for r in results]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
