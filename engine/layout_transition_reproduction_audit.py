"""Observe the ORIGINAL commit checks without changing their return values.

Run two bounded, fully restored attempts to distinguish a numerical identity
mismatch from a training-guard failure. No candidate, optimizer or promotion.
The temporary observer only reads already computed values at the original gate.
"""
import argparse
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from . import layout_loaded_transition_train as training
from .layout_operation_sampling import OperationCache
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def observe_attempt(model,bank,published,original):
    before,fixed,mode=model.checkpoint_hash(),model.fingerprint(),model.training
    params=(model.log_gains,model.tonic);base=[p.detach().clone() for p in params]
    original_gate=training.preservation_gate;checks=[];snapshot=None;result=None;error=None
    def observer(*args,**kwargs):
        nonlocal snapshot
        passed=original_gate(*args,**kwargs)
        if snapshot is None:
            snapshot={'parameterHash':model.checkpoint_hash(),
                      'arrays':[p.detach().cpu().numpy().copy() for p in params]}
        checks.append({'passed':passed,'before':copy.deepcopy(args[0]),'after':copy.deepcopy(args[1]),
                       'minimumImprovement':args[2] if len(args)>2 else kwargs.get('minimum_improvement',.01)})
        return passed
    try:
        with patch.object(training,'preservation_gate',observer):
            try:result=training.apply_verified_step(model,bank,published,original)
            except Exception as exc:error={'type':type(exc).__name__,'error':str(exc)}
    finally:
        with torch.no_grad():
            for p,b in zip(params,base):p.copy_(b)
        model.train(mode)
        if (training.preservation_gate is not original_gate or model.checkpoint_hash()!=before
                or model.fingerprint()!=fixed or any(p.grad is not None for p in params)):
            raise AssertionError('Original function, brain or gradient restoration failed')
    expected=next(t['temporaryParameterHash'] for t in published['trials'] if t['allTrainingGuards'])
    report={'parentParameterHash':before,'fixedHash':fixed,'originalCommitReturned':result is not None,
            'originalError':error,'originalGuardChecks':checks,'allOriginalGuardsPassed':len(checks)==2 and all(c['passed'] for c in checks),
            'expectedTemporaryParameterHash':expected,'observedTemporaryParameterHash':None if snapshot is None else snapshot['parameterHash'],
            'exactPublishedHashMatched':snapshot is not None and snapshot['parameterHash']==expected,
            'parametersRestored':True,'originalFunctionsRestored':True,'candidateSaved':False,'isServiceEvidence':False}
    return report,None if snapshot is None else snapshot['arrays']


def main(args):
    if args.out.exists():raise FileExistsError('Preserve reproduction evidence')
    old_manifest=json.loads((args.failed/'manifest.json').read_text())
    failure=json.loads((args.failed/'failure.json').read_text())
    if failure!={'type':'ValueError','error':'Original candidate or preservation guards did not reproduce'}:
        raise ValueError('Requires the original observed V31 failure')
    if (args.failed/'candidate-1.npz').exists():raise ValueError('Original failure must have saved no candidate')
    hashes={**old_manifest['inputFileHashes'],**{str((args.failed/n).resolve()):file_hash(args.failed/n)
            for n in ('manifest.json','status.json','failure.json')}}
    for p,h in hashes.items():
        if file_hash(Path(p))!=h:raise ValueError('Original input bytes changed')
    sources={**old_manifest['sourceHashes'],Path(__file__).name:file_hash(Path(__file__))}
    for n,h in sources.items():
        if Path(n).name!=n or file_hash(Path(__file__).parent/n)!=h:raise ValueError('Original source changed')
    for p in (args.candidate,args.cache/'cache.npz',args.cache/'manifest.json',args.dataset/'manifest.json',
              args.probe/'manifest.json',args.probe/'result.json',args.rejected_source/'result.json'):
        if hashes.get(str(p.resolve()))!=file_hash(p):raise ValueError('Wrong original input path')
    published=json.loads((args.probe/'result.json').read_text());original=json.loads((args.rejected_source/'result.json').read_text())
    torch.set_num_threads(4);model=load_model(args.root,args.candidate).train()
    bank=OperationCache(args.cache,args.dataset,published['parentParameterHash'],model.fixed_hash,model.n)
    before=model.checkpoint_hash();args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json',{'purpose':'Read-only exact original-check instrumentation','attempts':2,
        'parentParameterHash':before,'fixedHash':model.fixed_hash,'inputFileHashes':hashes,'sourceHashes':sources,
        'originalGuardsAndReturnValuesUnchanged':True,'candidateSaved':False,'optimizerUsed':False,'isServiceEvidence':False})
    began=time.perf_counter();first=None;rows=[]
    try:
        for attempt in range(2):
            row,arrays=observe_attempt(model,bank,published,original)
            row['attempt']=attempt+1
            if arrays is not None and first is not None:
                row['comparedWithFirstAttempt']=[{'differentEntries':int(np.count_nonzero(a!=b)),
                    'maximumAbsoluteDifference':float(np.max(np.abs(a-b))),
                    'rmsDifference':float(np.sqrt(np.mean((a.astype(np.float64)-b)**2)))} for a,b in zip(arrays,first)]
            if arrays is not None and first is None:first=arrays
            rows.append(row);atomic_json(args.out/f'attempt-{attempt+1}.json',row)
            print('ORIGINAL_COMMIT_AUDIT '+json.dumps({k:v for k,v in row.items() if k!='originalGuardChecks'}),flush=True)
            if not row['allOriginalGuardsPassed']:break
        if any(file_hash(Path(p))!=h for p,h in hashes.items()) or any(file_hash(Path(__file__).parent/n)!=h for n,h in sources.items()):
            raise AssertionError('Original inputs or sources changed')
        atomic_json(args.out/'result.json',{'finished':True,'attempts':len(rows),'parametersRestored':model.checkpoint_hash()==before,
            'sourcesAndInputsUnchanged':True,'candidateSaved':False,'optimizerUsed':False,'isServiceEvidence':False,
            'seconds':time.perf_counter()-began})
    except BaseException as exc:
        atomic_json(args.out/'failure.json',{'type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','candidate','cache','dataset','probe','rejected-source','failed','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
