"""Reversible individual-output pickup correction on original brain histories.

Only missed empty-pickup outputs receive a desired change. Every sampled
loaded grip decision over the32 scored frames is protected, including returns
opposed by the teacher. This is a TRAINING probe, not service evidence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_context_step_probe import direction,trust_scale
from .layout_operation_sampling import OperationCache
from .layout_operation_prefix import KINDS,CONTEXTS
from .layout_operation_focus import focus_values,focus_fit
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

CAPS=(.03,1e-4)
MOTION_BUDGET=(2.5e-5,2.5e-4)
LOADED_GRIP_BUDGET=.002
BACKTRACK=(1.,.5,.25,.125,.0625,.03125)


def measure(predictions,labels,observations,motion,selections):
    grip,_=focus_values(predictions,labels,selections)
    pickup=torch.as_tensor([s['category']==CONTEXTS[0] for s in selections],device=grip.device)
    if observations.shape!=(32,len(selections),297):raise ValueError('Original scored sensory frames required')
    cargo=observations[...,126:129]
    if not torch.all((cargo==0)|(cargo==1)) or not torch.all(cargo.sum(-1)<=1):
        raise ValueError('Original one-hot physical cargo required')
    loaded=cargo.sum(-1)>0
    if not pickup.any() or not loaded.any() or loaded[16,pickup].any():
        raise ValueError('Real empty-pickup and loaded histories required')
    raw=grip[pickup]
    return {'pickupRawGrip':raw.detach().cpu().tolist(),
            'pickupHinge':float(torch.relu(1.1-raw).square().mean().detach()),
            'protectedLoadedGrip':predictions[...,2][loaded].detach().cpu().tolist(),
            'parentMotionMSE':(predictions[...,:2]-motion).square().mean((0,1)).detach().cpu().tolist(),
            'rawGrip':grip.detach().cpu().tolist(),'loadedMask':loaded.cpu().tolist()}


def preservation_gate(before,after,minimum_improvement=.01):
    if not 0<=minimum_improvement<1:raise ValueError('Bounded improvement fraction required')
    for key in ('pickupRawGrip','protectedLoadedGrip','parentMotionMSE','rawGrip'):
        old,new=np.asarray(before[key]),np.asarray(after[key])
        if old.shape!=new.shape or not old.size or not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError('Finite aligned measured histories required')
    if before['loadedMask']!=after['loadedMask']:raise ValueError('Physical cargo mask changed')
    old,new=np.asarray(before['pickupRawGrip']),np.asarray(after['pickupRawGrip'])
    loaded,changed=np.asarray(before['protectedLoadedGrip']),np.asarray(after['protectedLoadedGrip'])
    if not np.isfinite([before['pickupHinge'],after['pickupHinge']]).all():raise ValueError('Finite loss required')
    return bool(before['pickupHinge']>0
        and after['pickupHinge']<=(1-minimum_improvement)*before['pickupHinge']+1e-9
        and np.all(new>=np.minimum(old,1.1)-1e-7)
        and not np.any((old>1.)&(new<=1.))
        and np.array_equal(loaded>1.,changed>1.)
        and np.max(np.abs(loaded-changed))<=LOADED_GRIP_BUDGET
        and np.all(np.asarray(after['parentMotionMSE'])<=np.asarray(MOTION_BUDGET)))


@torch.no_grad()
def measure_bank(model,bank,ids):
    pieces=[]
    for first in range(0,len(ids),16):
        selected=ids[first:first+16];x,y,state,motion=bank.batch(selected,model.tonic.device)
        pieces.append((frozen_predictions(model,x,state,64),y[64:],x[64:],motion))
    p,y,x,m=(torch.cat([part[i] for part in pieces],dim=1) for i in range(4))
    return measure(p,y,x,m,[bank.manifest['windows'][i] for i in ids])


def probe(model,bank):
    before,fixed,mode=model.checkpoint_hash(),model.fingerprint(),model.training
    params=(model.log_gains,model.tonic);base=[p.detach().clone() for p in params]
    if before!=bank.manifest['parameterHash'] or model.surrogate_training or any(p.grad is not None for p in params):
        raise ValueError('Exact derivative/current full-history parent required')
    ids=[bank.groups[k,c][0] for k in KINDS for c in CONTEXTS]
    selections=[bank.manifest['windows'][i] for i in ids]
    diagnostic_ids=bank.diagnostic_indexes()
    x,y,state,motion=bank.batch(ids,model.tonic.device)
    try:
        model.eval();diagnostic=measure_bank(model,bank,diagnostic_ids)
        model.train()
        p=recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        initial=measure(p,y[64:],x[64:],motion,selections)
        if initial['pickupHinge']<=0:raise ValueError('No missed-pickup margin deficit in selected batch')
        grip,_=focus_values(p,y[64:],selections)
        outputs=list(grip.unbind())+[p[...,h].mean() for h in range(2)]
        residual=[max(0.,1.1-float(grip[i].detach())) if s['category']==CONTEXTS[0] else 0.
                  for i,s in enumerate(selections)]+[0.,0.]
        gradients=[]
        for i,o in enumerate(outputs):
            gradients.append([g.detach() for g in torch.autograd.grad(o,params,retain_graph=i<len(outputs)-1)])
            print(json.dumps({'individualJacobianRows':i+1,'total':len(outputs),'permanentUpdates':False}),flush=True)
        delta,linear=direction(gradients,residual)
        del p,grip,outputs,gradients
        bounded=trust_scale(delta,CAPS);trials=[];model.eval()
        with torch.no_grad():
            for scale in BACKTRACK:
                for param,original,d in zip(params,base,delta):param.copy_(original+bounded*scale*d)
                params[0].clamp_(-2.,2.);params[1].clamp_(-.1,.1)
                measured=measure(frozen_predictions(model,x,state,64),y[64:],x[64:],motion,selections)
                batch_pass=preservation_gate(initial,measured)
                fit=measure_bank(model,bank,diagnostic_ids) if batch_pass else None
                passed=batch_pass and preservation_gate(diagnostic,fit,minimum_improvement=0.)
                row={'backtrack':scale,'scale':bounded*scale,'batchGuardPassed':batch_pass,
                     'allTrainingGuards':passed,'measured':measured,'diagnostic':fit,
                     'temporaryParameterHash':model.checkpoint_hash(),
                     'maximumParameterChanges':[float((p-b).abs().max()) for p,b in zip(params,base)]}
                trials.append(row)
                print('PICKUP_PRESERVATION_TRIAL '+json.dumps({k:row[k] for k in
                    ('backtrack','batchGuardPassed','allTrainingGuards','maximumParameterChanges')}),flush=True)
        return {'parentParameterHash':before,'fixedHash':fixed,'windowIndexes':ids,'selections':selections,
                'diagnosticWindowIndexes':diagnostic_ids,'initial':initial,'initialDiagnostic':diagnostic,
                'linearProposal':linear,'trials':trials,'parametersRestored':True,'candidateSaved':False,
                'optimizerUsed':False,'isServiceEvidence':False,'prefixExactAtInitialization':True,
                'gradientFrames':96,'scoredFrames':32,'prefixGradientDetached':True,
                'originalLabelsUnchanged':True,'teacherAtInference':False,'loadedReturnsPreservedRegardlessOfTeacher':True,
                'motionBudget':list(MOTION_BUDGET),'loadedGripBudget':LOADED_GRIP_BUDGET}
    finally:
        with torch.no_grad():
            for p,b in zip(params,base):p.copy_(b)
        model.train(mode)
        if model.checkpoint_hash()!=before or model.fingerprint()!=fixed or any(p.grad is not None for p in params):
            raise AssertionError('Original model was not restored')


def main(args):
    if args.out.exists():raise FileExistsError('Preserve all prior experiments')
    torch.set_num_threads(4)
    model=load_model(args.root,args.candidate).train()
    if model.checkpoint_hash()!=args.expected_parameter_hash:raise ValueError('Wrong original parent')
    bank=OperationCache(args.cache,args.dataset,args.expected_parameter_hash,model.fixed_hash,model.n)
    m=bank.manifest
    if m['candidateFileHash']!=file_hash(args.candidate):raise ValueError('Original candidate bytes required')
    paths=[args.candidate,args.cache/'manifest.json',args.cache/'cache.npz',args.dataset/'manifest.json']
    paths += [args.dataset/e['file'] for e in m['datasetFiles']]
    hashes={str(p.resolve()):file_hash(p) for p in paths}
    names={'layout_pickup_preservation_probe.py','layout_operation_sampling.py','layout_operation_focus.py',
           'layout_context_step_probe.py','layout_context_credit_probe.py','layout_recovery_train.py',
           'layout_demonstration_step_probe.py','layout_grip_objective.py','layout_recovery_protocol.py'}
    sources={**m['sourceHashes'],**{n:file_hash(Path(__file__).parent/n) for n in names}}
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json',{'parentParameterHash':model.checkpoint_hash(),'fixedHash':model.fixed_hash,
        'inputFileHashes':hashes,'sourceHashes':sources,'canonicalPrefixRun':m['prefixCanonicalRun'],
        'canonicalPrefixVersion':m['prefixCanonicalVersion'],'behaviorParameterHash':m['behaviorParameterHash'],
        'purpose':'Read-only individual-output pickup preservation test, NOT service','optimizerUsed':False})
    try:
        full=focus_fit(model,bank,all_windows=True)
        atomic_json(args.out/'initial-all-window-fit.json',full)
        print('INITIAL_FULL_CACHE_FIT '+json.dumps({k:full[k] for k in
            ('focusRecall','focusFalsePositiveRate','objectiveHeads','prefixIsExactForCandidate')}),flush=True)
        result=probe(model,bank)
        if any(file_hash(Path(p))!=h for p,h in hashes.items()) or any(file_hash(Path(__file__).parent/n)!=h for n,h in sources.items()):
            raise AssertionError('Original inputs or source changed')
        result.update(inputFileHashes=hashes,sourceHashes=sources,sourceFilesUnchanged=True)
        atomic_json(args.out/'result.json',result)
        print('PICKUP_PRESERVATION_COMPLETE '+json.dumps({'passingBacktracks':[r['backtrack'] for r in result['trials'] if r['allTrainingGuards']],
              'parametersRestored':True,'candidateSaved':False}),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','candidate','cache','dataset','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-parameter-hash',required=True)
    main(p.parse_args())
