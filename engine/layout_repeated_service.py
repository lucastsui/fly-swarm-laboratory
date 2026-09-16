"""Account for repeated frozen runs without counting layouts more than once."""
import argparse
import json
from pathlib import Path
import numpy as np
from .layout_frozen_comparison import compare
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def aggregate(reports):
    if not reports:
        raise ValueError('At least one complete report required')
    lookup = [{(t['kind'],t['seed']): t for t in r['trials']} for r in reports]
    if any(set(x) != set(lookup[0]) for x in lookup):
        raise ValueError('Different physical layouts cannot be treated as repeats')
    result = {}
    for kind in sorted({k for k,_ in lookup[0]}):
        keys = sorted(k for k in lookup[0] if k[0] == kind)
        rows = [{'seed': seed, 'products': [r[kind,seed]['products'] for r in lookup],
                 'sustained': [r[kind,seed]['sustainedSuccess'] for r in lookup]} for _,seed in keys]
        result[kind] = {'uniqueLayouts': len(keys), 'repeats': len(reports),
                        'sustainedWorldsPerRepeat': [sum(r[k]['sustainedSuccess'] for k in keys) for r in lookup],
                        'productsPerRepeat': [sum(r[k]['products'] for k in keys) for r in lookup],
                        'layoutsSustainedInEveryRepeat': sum(all(r['sustained']) for r in rows),
                        'layoutsWithInconsistentSustainedOutcome': sum(any(r['sustained']) and not all(r['sustained']) for r in rows),
                        'trials': rows}
    return result


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve repeated-service evidence')
    groups = []
    for paths in (args.baseline, args.candidate):
        reports = [json.loads((p/'evaluation.json').read_text()) for p in paths]
        manifests = [json.loads((p/'manifest.json').read_text()) for p in paths]
        if len({r['checkpointHash'] for r in reports}) != 1:
            raise ValueError('A repeat group must use one exact brain')
        if len({str(p.resolve()) for p in paths}) != len(paths):
            raise ValueError('Do not duplicate an evaluation file to manufacture repeats')
        for report, manifest in zip(reports, manifests):
            # This module only accepts the original/default numerical protocol.
            # The strict-mode wrapper has a separate required matched protocol.
            if any((p/'numerical-protocol.json').exists() or (p/'deterministic-failure.json').exists() for p in paths):
                raise ValueError('Do not mix numerical evaluation protocols')
            compare(report, manifest, reports[0], manifests[0], reports[0]['checkpointHash'])
        groups.append((reports, manifests))
    baseline, candidate = groups
    compare(candidate[0][0], candidate[1][0], baseline[0][0], baseline[1][0], candidate[0][0]['checkpointHash'])
    result = {'schema': 'repeated-frozen-development-service-v1',
              'baselineHash': baseline[0][0]['checkpointHash'], 'candidateHash': candidate[0][0]['checkpointHash'],
              'baseline': aggregate(baseline[0]), 'candidate': aggregate(candidate[0]),
              'fullOriginalControlsAndEventsChecked': True,
              'balancedRepeats': len(baseline[0])==len(candidate[0]),
              'independentLayoutCount': len(baseline[0][0]['trials']),
              'repeatsAreNotAdditionalIndependentLayouts': True,
              'developmentSelectionNotFreshConfirmation': True, 'universalLayoutClaim': False,
              'inputs': {str(p.resolve()): {n: file_hash(p/n) for n in ('manifest.json','evaluation.json')}
                         for p in args.baseline+args.candidate}}
    atomic_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','baseline','candidate')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',nargs='+',type=Path,required=True)
    p.add_argument('--candidate',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
