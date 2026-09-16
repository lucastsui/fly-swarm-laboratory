"""Read-only serialized physical replay and honest operation-coverage audit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .layout_correction_prefix import load_histories, DATA_SOURCES
from .layout_operation_prefix import operation_specs, CONTEXTS, KINDS
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def eligible_counts(a,m):
    n=m['frames'];ticks=np.arange(n);eligible=(ticks>=80)&(ticks<n-16)
    counts={c:0 for c in CONTEXTS}
    missed=((a['bodies'][:-1,:,5]==0)&(a['labels'][...,2]>1.)
            &(a['appliedActions'][...,2]<.5)&(a['cooldowns'][:-1]<=.05))
    counts[CONTEXTS[0]]=int((missed&eligible[:,None]).sum())
    for event in m['interactionEvents']:
        tick,fly=round(event['time']/.05)-1,event['fly']
        if not 0<=tick<n or fly not in range(4):raise ValueError('Invalid original event')
        if not eligible[tick]:continue
        on=a['labels'][tick,fly,2]>1.
        if event['event']=='return':counts[CONTEXTS[3 if on else 2]]+=1
        elif event['event'] in ('transfer','delivery') and on:counts[CONTEXTS[1]]+=1
    return counts


def main(args):
    if args.out.exists():raise FileExistsError('Preserve existing audit')
    began=time.perf_counter()
    episodes,manifest=load_histories(args.dataset,replay=True)
    rows=[{'kind':m['kind'],'seed':m['seed'],'eligibleEventCounts':eligible_counts(a,m)} for a,m in episodes]
    counts={k:{c:sum(r['eligibleEventCounts'][c] for r in rows if r['kind']==k) for c in CONTEXTS} for k in KINDS}
    coverage_error=None;windows=[]
    try:windows=operation_specs(episodes)
    except ValueError as error:
        if str(error)!='Missing actual operation family/context':raise
        coverage_error=str(error)
    selected={k:{c:sum(s['kind']==k and s['category']==c for s in windows) for c in CONTEXTS} for k in KINDS} if windows else None
    names=set(DATA_SOURCES)|{'layout_correction_prefix.py','layout_operation_prefix.py',Path(__file__).name}
    report={'canonicalRun':manifest['canonicalRun'],'canonicalVersion':manifest['canonicalVersion'],
            'parameterHash':manifest['parameterHash'],'datasetManifestHash':file_hash(args.dataset/'manifest.json'),
            'episodes':len(episodes),'allSerializedPhysicalReplaysPassed':True,'isServiceEvidence':False,
            'optimizerUsed':False,'teacherControlsBodies':False,'operationCurriculumComplete':coverage_error is None,
            'coverageError':coverage_error,'eligibleEventCounts':counts,'perEpisode':rows,
            'selectedWindowCounts':selected,'selectedWindows':len(windows),
            'sourceHashes':{n:file_hash(Path(__file__).parent/n) for n in sorted(names)},
            'seconds':time.perf_counter()-began}
    atomic_json(args.out,report)
    print('SERIALIZED_OPERATION_AUDIT_COMPLETE '+json.dumps({k:v for k,v in report.items() if k not in ('perEpisode','sourceHashes')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
