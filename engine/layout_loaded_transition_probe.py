"""Add observed loaded-transition constraints to the rejected V29 proposal.

Same parent, physical histories, labels, budgets and minimum improvement.
Only the Jacobian solve gains actual scored-frame preservation rows.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_pickup_preservation_probe import measure,measure_bank,preservation_gate,CAPS,BACKTRACK,LOADED_GRIP_BUDGET
from .layout_context_step_probe import direction,trust_scale
from .layout_operation_focus import focus_values
from .layout_operation_prefix import CONTEXTS
from .layout_operation_sampling import OperationCache
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def active_constraints(original):
    if any(t['allTrainingGuards'] for t in original['trials']):raise ValueError('Requires rejected original proposal')
    trial=original['trials'][0]
    if trial['backtrack']!=1.:raise ValueError('Original full-scale trial required')
    before=original['initial'];after=trial['measured']
    mask=np.asarray(before['loadedMask'],dtype=bool)
    if mask.shape!=(32,16) or before['loadedMask']!=after['loadedMask']:raise ValueError('Original loaded frames required')
    indexes=np.argwhere(mask);old=np.asarray(before['protectedLoadedGrip']);new=np.asarray(after['protectedLoadedGrip'])
    if old.shape!=(len(indexes),) or new.shape!=old.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Aligned finite original loaded outputs required')
    delta=np.abs(new-old);rows=[]
    for actor in range(16):
        choices=np.flatnonzero(indexes[:,1]==actor)
        if not len(choices):continue
        worst=int(choices[np.argmax(delta[choices])])
        if delta[worst]>LOADED_GRIP_BUDGET:
            rows.append({'actorInBatch':actor,'scoredFrame':int(indexes[worst,0]),
                         'windowIndex':original['windowIndexes'][actor],'originalMaxChange':float(delta[worst])})
    if not 1<=len(rows)<=16:raise ValueError('At least one actual loaded-frame violation required')
    return rows


def same_measure(a,b):
    if a['loadedMask']!=b['loadedMask']:raise ValueError('Changed original cargo mask')
    for key in ('rawGrip','pickupRawGrip','protectedLoadedGrip','pickupHinge','parentMotionMSE'):
        np.testing.assert_allclose(a[key],b[key],rtol=1e-5,atol=1e-6)


def probe(model,bank,original):
    active=active_constraints(original)
    before,fixed,mode=model.checkpoint_hash(),model.fingerprint(),model.training
    params=(model.log_gains,model.tonic);base=[p.detach().clone() for p in params]
    if (before!=original['parentParameterHash'] or before!=bank.manifest['parameterHash']
            or fixed!=original['fixedHash'] or model.surrogate_training or any(p.grad is not None for p in params)):
        raise ValueError('Original exact parent/derivative/prefix required')
    ids=original['windowIndexes'];selections=[bank.manifest['windows'][i] for i in ids]
    diagnostic_ids=original['diagnosticWindowIndexes']
    if selections!=original['selections'] or diagnostic_ids!=bank.diagnostic_indexes():raise ValueError('Original selections changed')
    try:
        model.eval();diagnostic=measure_bank(model,bank,diagnostic_ids);same_measure(diagnostic,original['initialDiagnostic'])
        x,y,state,motion=bank.batch(ids,model.tonic.device);model.train()
        p=recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        initial=measure(p,y[64:],x[64:],motion,selections);same_measure(initial,original['initial'])
        grip,_=focus_values(p,y[64:],selections)
        outputs=list(grip.unbind())+[p[a['scoredFrame'],a['actorInBatch'],2] for a in active]+[p[...,h].mean() for h in range(2)]
        residual=[max(0.,1.1-float(grip[i].detach())) if s['category']==CONTEXTS[0] else 0.
                  for i,s in enumerate(selections)]+[0.]*(len(active)+2)
        gradients=[]
        for i,o in enumerate(outputs):
            gradients.append([g.detach() for g in torch.autograd.grad(o,params,retain_graph=i<len(outputs)-1)])
            print(json.dumps({'transitionJacobianRows':i+1,'total':len(outputs),'permanentUpdates':False}),flush=True)
        delta,linear=direction(gradients,residual);del p,grip,outputs,gradients
        bounded=trust_scale(delta,CAPS);trials=[];model.eval()
        with torch.no_grad():
            for backtrack in BACKTRACK:
                for param,base_value,d in zip(params,base,delta):param.copy_(base_value+bounded*backtrack*d)
                params[0].clamp_(-2.,2.);params[1].clamp_(-.1,.1)
                measured=measure(frozen_predictions(model,x,state,64),y[64:],x[64:],motion,selections)
                passed=preservation_gate(initial,measured)
                fit=measure_bank(model,bank,diagnostic_ids) if passed else None
                all_pass=passed and preservation_gate(diagnostic,fit,0.)
                row={'backtrack':backtrack,'scale':bounded*backtrack,'batchGuardPassed':passed,'allTrainingGuards':all_pass,
                     'measured':measured,'diagnostic':fit,'temporaryParameterHash':model.checkpoint_hash(),
                     'maximumParameterChanges':[float((p-b).abs().max()) for p,b in zip(params,base)]}
                trials.append(row)
                print('LOADED_TRANSITION_TRIAL '+json.dumps({k:row[k] for k in ('backtrack','batchGuardPassed','allTrainingGuards')}),flush=True)
        return {'parentParameterHash':before,'fixedHash':fixed,'activeLoadedFrames':active,
                'windowIndexes':ids,'selections':selections,'diagnosticWindowIndexes':diagnostic_ids,
                'initial':initial,'initialDiagnostic':diagnostic,'linearProposal':linear,'trials':trials,
                'parametersRestored':True,'candidateSaved':False,'optimizerUsed':False,'isServiceEvidence':False,
                'exactReLUDerivative':True,'prefixExactAtInitialization':True,'gradientFrames':96,
                'scoredFrames':32,'prefixGradientDetached':True,'oldTrainingAndServiceGuardsUnchanged':True}
    finally:
        with torch.no_grad():
            for p,b in zip(params,base):p.copy_(b)
        model.train(mode)
        if model.checkpoint_hash()!=before or model.fingerprint()!=fixed or any(p.grad is not None for p in params):
            raise AssertionError('Original brain was not restored')


def main(args):
    if args.out.exists():raise FileExistsError('Preserve prior probe')
    source=json.loads((args.source/'manifest.json').read_text());original=json.loads((args.source/'result.json').read_text())
    if ((args.source/'failure.json').exists() or not original['parametersRestored'] or original['candidateSaved']
            or not original['sourceFilesUnchanged'] or original['inputFileHashes']!=source['inputFileHashes']
            or original['sourceHashes']!=source['sourceHashes']):raise ValueError('Complete restored original probe required')
    active=active_constraints(original)
    hashes={**original['inputFileHashes'],**{str((args.source/n).resolve()):file_hash(args.source/n) for n in ('manifest.json','result.json')}}
    for p,h in hashes.items():
        if file_hash(Path(p))!=h:raise ValueError('Original input changed')
    for n,h in original['sourceHashes'].items():
        if Path(n).name!=n or file_hash(Path(__file__).parent/n)!=h:raise ValueError('Original source changed')
    for p in (args.candidate,args.cache/'cache.npz',args.cache/'manifest.json',args.dataset/'manifest.json'):
        if hashes.get(str(p.resolve()))!=file_hash(p):raise ValueError('Wrong original cache/candidate/data')
    torch.set_num_threads(4);model=load_model(args.root,args.candidate).train()
    bank=OperationCache(args.cache,args.dataset,original['parentParameterHash'],model.fixed_hash,model.n)
    sources={**original['sourceHashes'],Path(__file__).name:file_hash(Path(__file__))}
    args.out.mkdir(parents=True)
    manifest={'parentParameterHash':original['parentParameterHash'],'fixedHash':original['fixedHash'],
              'inputFileHashes':hashes,'sourceHashes':sources,'activeLoadedFrames':active,
              'purpose':'Additional observed loaded-frame Jacobian constraints; no changed acceptance or service rules',
              'optimizerUsed':False,'isServiceEvidence':False}
    atomic_json(args.out/'manifest.json',manifest)
    try:
        r=probe(model,bank,original)
        if any(file_hash(Path(p))!=h for p,h in hashes.items()) or any(file_hash(Path(__file__).parent/n)!=h for n,h in sources.items()):
            raise AssertionError('Original files changed')
        r.update(sourceFilesUnchanged=True,inputFileHashes=hashes,sourceHashes=sources)
        atomic_json(args.out/'result.json',r)
        print('LOADED_TRANSITION_PROBE_COMPLETE '+json.dumps({'passingBacktracks':[t['backtrack'] for t in r['trials'] if t['allTrainingGuards']],
              'parametersRestored':True,'candidateSaved':False}),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','candidate','cache','dataset','source','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
