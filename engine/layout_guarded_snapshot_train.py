"""Save the exact tested proposal instead of requiring cross-run byte identity.

Explicit ONE-step algorithm. The original numerical, motion, handling and
parameter bounds are unchanged. A completed identity-mismatch audit is required.
No old Adam/RNG state, teacher at inference, new decoder or deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_transition_reproduction_audit import observe_attempt
from .layout_loaded_transition_train import same_measure
from .layout_pickup_preservation_probe import measure_bank,preservation_gate,CAPS,MOTION_BUDGET,LOADED_GRIP_BUDGET
from .layout_operation_sampling import OperationCache
from .layout_recovery_brain import load_model,save_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

ALGORITHM='single-pass-guarded-loaded-transition-snapshot-v1'
IDENTITY_ERROR={'type':'ValueError','error':'Original candidate or preservation guards did not reproduce'}


def validate_audit(manifest,result,attempts,published):
    if (not result.get('finished') or result.get('attempts')!=2 or not result.get('parametersRestored')
            or not result.get('sourcesAndInputsUnchanged') or result.get('candidateSaved') is not False
            or result.get('optimizerUsed') is not False or result.get('isServiceEvidence') is not False or len(attempts)!=2
            or manifest['parentParameterHash']!=published['parentParameterHash']
            or manifest['fixedHash']!=published['fixedHash'] or not manifest['originalGuardsAndReturnValuesUnchanged']):
        raise ValueError('Completed original two-attempt audit required')
    chosen=next(t for t in published['trials'] if t['allTrainingGuards'])
    for i,row in enumerate(attempts):
        if (row['parentParameterHash']!=published['parentParameterHash'] or row['fixedHash']!=published['fixedHash']
                or row['attempt']!=i+1 or row['originalError']!=IDENTITY_ERROR or row['originalCommitReturned']
                or not row['allOriginalGuardsPassed'] or row['exactPublishedHashMatched']
                or not row['parametersRestored'] or not row['originalFunctionsRestored'] or row['candidateSaved']
                or row['expectedTemporaryParameterHash']!=chosen['temporaryParameterHash']
                or row['observedTemporaryParameterHash'] in (row['expectedTemporaryParameterHash'],published['parentParameterHash'])):
            raise ValueError('The original audit must isolate an identity-only mismatch')
        if len(row['originalGuardChecks'])!=2:raise ValueError('Both original gates required')
        for j,c in enumerate(row['originalGuardChecks']):
            if c['minimumImprovement']!=(.01 if j==0 else 0.) or not c['passed'] or not preservation_gate(c['before'],c['after'],c['minimumImprovement']):
                raise ValueError('Original training guard failed')
            same_measure(c['before'],published['initial' if j==0 else 'initialDiagnostic'])
            same_measure(c['after'],chosen['measured' if j==0 else 'diagnostic'])
    differences=attempts[1]['comparedWithFirstAttempt']
    if len(differences)!=2 or any(d['differentEntries']<0 or not np.isfinite([d['maximumAbsoluteDifference'],d['rmsDifference']]).all() for d in differences):
        raise ValueError('Actual repeated-array comparison required')


def apply_snapshot_step(model,bank,published,original):
    before,fixed,mode=model.checkpoint_hash(),model.fingerprint(),model.training
    params=(model.log_gains,model.tonic);base=[p.detach().clone() for p in params];committed=False
    try:
        observed,arrays=observe_attempt(model,bank,published,original)
        if (arrays is None or not observed['allOriginalGuardsPassed'] or not observed['parametersRestored']
                or not observed['originalFunctionsRestored'] or observed['parentParameterHash']!=before
                or observed['fixedHash']!=fixed or model.checkpoint_hash()!=before
                or (not observed['originalCommitReturned'] and observed['originalError']!=IDENTITY_ERROR)):
            raise ValueError('No fully guarded original proposal snapshot')
        if len(arrays)!=2:raise ValueError('Both brain arrays required')
        for a,b,cap in zip(arrays,base,CAPS):
            if (a.shape!=tuple(b.shape) or a.dtype!=np.float32 or not np.isfinite(a).all()
                    or np.max(np.abs(a-b.cpu().numpy()))>cap+1e-7):
                raise ValueError('Original parameter trust bound or snapshot shape failed')
        if np.max(np.abs(arrays[0]))>2. or np.max(np.abs(arrays[1]))>.1:
            raise ValueError('Original absolute parameter bounds failed')
        exact=hashlib.sha256(arrays[0].tobytes()+arrays[1].tobytes()).hexdigest()
        if exact!=observed['observedTemporaryParameterHash'] or exact==before:
            raise ValueError('Snapshot bytes differ from tested proposal')
        with torch.no_grad():
            for p,a in zip(params,arrays):p.copy_(torch.as_tensor(a,device=p.device))
        model.eval()
        ids=published['windowIndexes'];diagnostic_ids=published['diagnosticWindowIndexes']
        measured=[measure_bank(model,bank,ids),measure_bank(model,bank,diagnostic_ids)]
        for j,(check,current) in enumerate(zip(observed['originalGuardChecks'],measured)):
            same_measure(current,check['after'])
            if not preservation_gate(check['before'],current,.01 if j==0 else 0.):
                raise ValueError('Exact restored snapshot failed an original training guard')
        if model.checkpoint_hash()!=exact or model.fingerprint()!=fixed:
            raise ValueError('Measured snapshot identity changed')
        committed=True
        return {'beforeHash':before,'afterHash':exact,'snapshotBytesReverified':True,
            'crossRunByteIdentityRequired':False,'previousProbeTemporaryHash':observed['expectedTemporaryParameterHash'],
            'numericalAgreementWithOriginalProbe':True,'allOriginalTrainingGuardsPassed':True,
            'windowIndexes':ids,'diagnosticWindowIndexes':diagnostic_ids,'activeLoadedFrames':published['activeLoadedFrames'],
            'initial':observed['originalGuardChecks'][0]['before'],'after':measured[0],
            'initialDiagnostic':observed['originalGuardChecks'][1]['before'],'afterDiagnostic':measured[1],
            'maximumParameterChanges':[float((p.detach()-b).abs().max()) for p,b in zip(params,base)],
            'originalAttemptOutcome':{'returned':observed['originalCommitReturned'],'error':observed['originalError']},
            'isServiceEvidence':False}
    finally:
        if not committed:
            with torch.no_grad():
                for p,b in zip(params,base):p.copy_(b)
        model.train(mode)
        if (model.fingerprint()!=fixed or any(p.grad is not None for p in params)
                or (not committed and model.checkpoint_hash()!=before)):
            raise AssertionError('Fixed state, gradient or rollback invariant failed')


def verify_saved_snapshot(path,model):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'gains','tonic','interface','status_scale','fixed_hash'}:raise ValueError('Wrong saved arrays')
        for name,p in [('gains',model.log_gains),('tonic',model.tonic)]:
            if not np.array_equal(z[name],p.detach().cpu().numpy()):raise ValueError('Saved array differs from tested weights')
        if (str(z['fixed_hash'])!=model.fixed_hash or str(z['interface'])!=model.interface
                or not np.array_equal(z['status_scale'],model.status_scale.cpu().numpy())
                or hashlib.sha256(z['gains'].tobytes()+z['tonic'].tobytes()).hexdigest()!=model.checkpoint_hash()):
            raise ValueError('Saved snapshot identity differs')


def main(args):
    if args.out.exists():raise FileExistsError('Preserve every prior run')
    am=json.loads((args.audit/'manifest.json').read_text());ar=json.loads((args.audit/'result.json').read_text())
    rows=[json.loads((args.audit/f'attempt-{i}.json').read_text()) for i in (1,2)]
    published=json.loads((args.probe/'result.json').read_text());original=json.loads((args.rejected_source/'result.json').read_text())
    if (args.audit/'failure.json').exists():raise ValueError('Failed audit cannot authorize snapshot method')
    validate_audit(am,ar,rows,published)
    hashes={**am['inputFileHashes'],**{str((args.audit/n).resolve()):file_hash(args.audit/n)
            for n in ('manifest.json','result.json','attempt-1.json','attempt-2.json')}}
    sources={**am['sourceHashes'],Path(__file__).name:file_hash(Path(__file__))}
    def unchanged():
        if any(file_hash(Path(p))!=h for p,h in hashes.items()) or any(Path(n).name!=n or file_hash(Path(__file__).parent/n)!=h for n,h in sources.items()):
            raise ValueError('Original source/input bytes changed')
    unchanged()
    for p in (args.candidate,args.cache/'cache.npz',args.cache/'manifest.json',args.dataset/'manifest.json',
              args.probe/'manifest.json',args.probe/'result.json',args.rejected_source/'manifest.json',args.rejected_source/'result.json'):
        if hashes.get(str(p.resolve()))!=file_hash(p):raise ValueError('Wrong original input path')
    torch.set_num_threads(4);model=load_model(args.root,args.candidate).train()
    bank=OperationCache(args.cache,args.dataset,published['parentParameterHash'],model.fixed_hash,model.n)
    gains=model.log_gains.detach().cpu().numpy().copy();tonic=model.tonic.detach().cpu().numpy().copy()
    args.out.mkdir(parents=True)
    manifest={'algorithm':ALGORITHM,'maximumUpdates':1,'canonicalLearner':'Spark1 only','parentAdamConsumed':False,
        'optimizerState':'One stateless original20-output solve; snapshot tested and saved in the same run; no RNG',
        'parentParameterHash':model.checkpoint_hash(),'fixedHash':model.fixed_hash,'sourceHashes':sources,'inputFileHashes':hashes,
        'exactReLUDerivative':True,'gradientFrames':96,'prefixAgeUpdates':0,'jacobianOutputs':20,
        'activeLoadedFrames':published['activeLoadedFrames'],'motionBudget':list(MOTION_BUDGET),'loadedGripBudget':LOADED_GRIP_BUDGET,
        'coordinateCaps':list(CAPS),'teacherAtInference':False,'decoderTrained':False,'externalDecisionNetwork':False,
        'originalLabelsUnchanged':True,'dopamineLearning':False,'isServiceEvidence':False,'crossRunByteIdentityRequired':False,
        'exactSavedSnapshotIdentityRequired':True,'physicalAndTrainingGuardsUnchanged':True}
    atomic_json(args.out/'manifest.json',manifest);began=time.perf_counter()
    atomic_json(args.out/'status.json',{'finished':False,'acceptedUpdates':0,'parameterHash':model.checkpoint_hash()})
    try:
        row=apply_snapshot_step(model,bank,published,original);unchanged()
        audit=model.audit(gains);path=args.out/'candidate-1.npz';save_model(path,model);verify_saved_snapshot(path,model)
        unchanged();atomic_json(args.out/'step-1.json',row)
        result={'algorithm':ALGORITHM,'acceptedUpdates':1,'finalParameterHash':model.checkpoint_hash(),
            'candidateFileHash':file_hash(path),'audit':audit,'changedNeurons':int(np.count_nonzero(tonic!=model.tonic.detach().cpu().numpy())),
            'seconds':time.perf_counter()-began,'sourceFilesUnchanged':True,'savedArraysExactlyMatchTestedSnapshot':True,
            'requiresFrozenPhysicalVerification':True,'isServiceEvidence':False,'automaticContinuation':False}
        atomic_json(args.out/'result.json',result)
        atomic_json(args.out/'status.json',{'finished':True,'acceptedUpdates':1,'parameterHash':model.checkpoint_hash()})
        print('GUARDED_SNAPSHOT_TRAINING_COMPLETE '+json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(args.out/'failure.json',{'type':type(exc).__name__,'error':str(exc)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','candidate','cache','dataset','probe','rejected-source','audit','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
