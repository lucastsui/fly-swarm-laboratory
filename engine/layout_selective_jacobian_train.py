"""At most eight canonical, guarded damped-Jacobian brain-training updates.

This is an explicit NEW optimizer algorithm, not an Adam continuation. Exact
recurrent backprop generates output Jacobians. Only existing signed gains and
excitabilities change; fixed decoder/senses/physics and original labels stay.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_context_step_probe import direction, trust_scale, scores
from .layout_decision_resume import source_run, AnchoredOperationCache
from .layout_correction_prefix import load_histories
from .layout_operation_sampling import OperationCache
from .layout_operation_prefix import KINDS, CONTEXTS
from .layout_operation_focus import focus_values, focus_fit
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

VERSION='guarded-context-jacobian-training-v1'
CAPS=(.03,1e-4)
MOTION_BUDGET=(1e-4,.001)
BACKTRACK=(1.,.5,.25,.125,.0625,.03125)


def context_gate(before, after, reference_motion):
    arrays=[before['heads'],after['heads'],reference_motion,before['meanRawGripByContext'],after['meanRawGripByContext']]
    if (any(not np.isfinite(a).all() for a in arrays)
            or len(before['meanRawGripByContext'])!=4 or len(after['meanRawGripByContext'])!=4):
        raise ValueError('Finite original four-context scores required')
    directional=[]
    for i,(old,new) in enumerate(zip(before['meanRawGripByContext'],after['meanRawGripByContext'])):
        directional.append(new<=max(.9,old)+1e-7 if i==2 else new>=min(1.1,old)-1e-7)
    return bool(all(directional) and after['heads'][2]<=.99*before['heads'][2]
        and after['recall']>=before['recall'] and after['falsePositiveRate']<=before['falsePositiveRate']
        and all(after['heads'][h]<=reference_motion[h]+MOTION_BUDGET[h] for h in range(2)))


def diagnostic_gate(initial, current, proposed):
    return bool(proposed['objectiveHeads'][2]<=current['objectiveHeads'][2]+1e-8
        and proposed['focusRecall']>=initial['focusRecall']
        and proposed['focusFalsePositiveRate']<=initial['focusFalsePositiveRate']
        and all(proposed['parentMotionMSE'][h]<=initial['parentMotionMSE'][h]+MOTION_BUDGET[h] for h in range(2)))


def guarded_update(model,bank,fresh,ids,initial_fit,current_fit):
    params=(model.log_gains,model.tonic)
    originals=[p.detach().clone() for p in params]
    before,fixed=model.checkpoint_hash(),model.fingerprint()
    if model.surrogate_training or any(p.grad is not None for p in params):
        raise ValueError('Exact derivative and no accumulated gradients required')
    selections=[bank.manifest['windows'][i] for i in ids]
    if len(ids)!=16 or {(s['kind'],s['category']) for s in selections}!={(k,c) for k in KINDS for c in CONTEXTS}:
        raise ValueError('One original physical decision per family/context required')
    x,y,state,motion=bank.batch(ids,model.tonic.device)
    reference=fresh.batch(ids,model.tonic.device)[3]
    reference_motion=(reference-motion).square().mean((0,1)).cpu().tolist()
    accepted=False
    try:
        model.train()
        p=recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        initial=scores(p,y[64:],motion,selections)
        grip,_=focus_values(p,y[64:],selections)
        output=[grip[[j for j,s in enumerate(selections) if s['category']==c]].mean() for c in CONTEXTS]
        output += [p[...,h].mean() for h in range(2)]
        values=[float(o.detach()) for o in output]
        residual=[max(0.,1.1-v) if i!=2 else min(0.,.9-v) for i,v in enumerate(values[:4])]+[0.,0.]
        gradients=[]
        for i,o in enumerate(output):
            gradients.append([g.detach() for g in torch.autograd.grad(o,params,retain_graph=i<5)])
            print(json.dumps({'jacobianRow':i+1,'acceptedUpdatePending':True}),flush=True)
        norms={name:[float(g[b].norm()) for g in gradients] for b,name in enumerate(('log_gains','tonic'))}
        delta,linear=direction(gradients,residual)
        del p,grip,output,gradients
        bounded=trust_scale(delta,CAPS);trials=[];chosen_fit=None
        model.eval()
        with torch.no_grad():
            for backtrack in BACKTRACK:
                scale=bounded*backtrack
                for param,base,d in zip(params,originals,delta):param.copy_(base+scale*d)
                params[0].clamp_(-2.,2.);params[1].clamp_(-.1,.1)
                measured=scores(frozen_predictions(model,x,state,64),y[64:],motion,selections)
                local_pass=context_gate(initial,measured,reference_motion)
                diagnostic=focus_fit(model,bank) if local_pass else None
                passed=local_pass and diagnostic_gate(initial_fit,current_fit,diagnostic)
                trial={'backtrack':backtrack,'scale':scale,'measured':measured,'contextGatePassed':local_pass,
                       'fixedDiagnostic':diagnostic,'accepted':passed,
                       'maximumParameterChanges':[float((param-base).abs().max()) for param,base in zip(params,originals)]}
                trials.append(trial)
                print('GUARDED_CONTEXT_TRIAL '+json.dumps(trial),flush=True)
                if passed:
                    if model.checkpoint_hash()==before:raise AssertionError('Accepted update changed no weights')
                    chosen_fit=diagnostic;accepted=True;break
        return {'accepted':accepted,'beforeHash':before,'afterHash':model.checkpoint_hash() if accepted else before,
                'windowIndexes':ids,'selections':selections,'originalLabelsUnchanged':True,'teacherActions':False,
                'initialScores':initial,'referenceParentMotionMSE':reference_motion,'linearProposal':linear,
                'jacobianNorms':norms,'trials':trials,'trainingFit':chosen_fit,'isServiceEvidence':False}
    finally:
        if not accepted:
            with torch.no_grad():
                for p,base in zip(params,originals):p.copy_(base)
            if model.checkpoint_hash()!=before:raise AssertionError('Rejected update not restored')
        if model.fingerprint()!=fixed or any(p.grad is not None for p in params):
            raise AssertionError('Fixed graph/decoder/senses or gradient state changed')


def train(model,bank,fresh,rng,updates,record,on_initial=None):
    if type(updates)is not int or not 1<=updates<=8:raise ValueError('Hard eight-update/prefix-age bound required')
    if model.checkpoint_hash()!=fresh.manifest['parameterHash']:raise ValueError('Initial full-history prefix mismatch')
    initial=focus_fit(model,bank);current=initial;history=[]
    if on_initial is not None:on_initial(initial)
    for update in range(1,updates+1):
        # First step reproduces the previously tested physical context selection.
        ids=[bank.groups[k,c][0] if update==1 else int(rng.choice(bank.groups[k,c])) for k in KINDS for c in CONTEXTS]
        row=guarded_update(model,bank,fresh,ids,initial,current)
        row['update']=update;row['prefixAgeUpdates']=update-1
        if row['accepted']:current=row['trainingFit']
        history.append(row);record(update,row)
        if not row['accepted']:break
    return initial,history


def main(args):
    if args.out.exists():raise FileExistsError('Never overwrite canonical experiment')
    if type(args.updates)is not int or not 1<=args.updates<=8:raise ValueError('At most eight updates')
    source,parent,original_result,_=source_run(args.source)
    prior=json.loads((args.probe/'result.json').read_text())
    if ((args.probe/'failure.json').exists() or not prior['parametersUnchanged'] or prior['candidateSaved']
            or not prior['temporaryChangesAlwaysRestored'] or prior['selectedTemporaryTrial'] is None
            or not prior['trials'][prior['selectedTemporaryTrial']]['trainingProbePass']):
        raise ValueError('Completed bounded successful/restored original proposal required')
    hashes={**prior['inputFileHashes'],str((args.probe/'result.json').resolve()):file_hash(args.probe/'result.json')}
    for path,h in hashes.items():
        if file_hash(Path(path))!=h:raise ValueError('Original probe/brain/data changed')
    for n,h in prior['sourceHashes'].items():
        if file_hash(Path(__file__).parent/n)!=h:raise ValueError('Original probe source changed')
    for p in (args.cache/'cache.npz',args.cache/'manifest.json',args.original_cache/'cache.npz',
              args.original_cache/'manifest.json',args.dataset/'manifest.json'):
        if hashes.get(str(p.resolve()))!=file_hash(p):raise ValueError('Not the original selected probe inputs')
    torch.set_num_threads(4)
    model=load_model(args.root,source/'candidate-100.npz').train()
    before=model.checkpoint_hash();fixed=model.fixed_hash
    if before!=prior['parameterHash'] or before!=original_result['finalParameterHash']:
        raise ValueError('Original C100 candidate mismatch')
    fresh=OperationCache(args.cache,args.dataset,before,fixed,model.n)
    original=OperationCache(args.original_cache,args.dataset,parent['parentParameterHash'],fixed,model.n)
    if (fresh.manifest['candidateFileHash']!=file_hash(source/'candidate-100.npz')
            or fresh.manifest['prefixCanonicalRun']!=source.name or fresh.manifest['prefixCanonicalVersion']!=100):
        raise ValueError('Exact original full C100 prefix required')
    bank=AnchoredOperationCache(fresh,original)
    print('SERIALIZED_ORIGINAL_PHYSICAL_REPLAY',flush=True)
    episodes,_=load_histories(args.dataset,replay=True)
    if len(episodes)!=16:raise ValueError('Original sixteen-world dataset required')
    del episodes
    sources={**prior['sourceHashes'],Path(__file__).name:file_hash(Path(__file__))}
    args.out.mkdir(parents=True)
    original_gains=model.log_gains.detach().cpu().numpy().copy()
    original_tonic=model.tonic.detach().cpu().numpy().copy()
    rng=np.random.default_rng(9270001);began=time.perf_counter()
    save_model(args.out/'candidate-0.npz',model)
    manifest={'experiment':VERSION,'parentParameterHash':before,'parentCanonicalRun':source.name,'parentCanonicalVersion':100,
        'fixedHash':fixed,'inputFileHashes':hashes,'sourceHashes':sources,'maximumUpdates':args.updates,
        'maximumPrefixAgeUpdates':8,'gradientFrames':96,'lossFrames':32,'gripFocusOffset':16,
        'optimizer':'New damped six-output Jacobian solver; NOT saved Adam continuation',
        'parentAdamStateConsumed':False,'damping':.001,'coordinateScales':[.003,1e-5],'coordinateCaps':list(CAPS),
        'backtracking':list(BACKTRACK),'seed':9270001,'initialSamplerRNG':rng.bit_generator.state,
        'fixedDiagnosticMotionBudget':list(MOTION_BUDGET),'firstBatchReproducesOriginalProbe':True,
        'originalMotionAnchorHash':original.manifest['parameterHash'],'behaviorParameterHash':fresh.manifest['behaviorParameterHash'],
        'serializedPhysicalReplayOnTrainingHostPassed':True,'datasetEpisodes':16,'labelsUnchanged':True,
        'canonicalLearner':'Spark1 only','teacherAtInference':False,'externalDecisionNetwork':False,
        'decoderTrained':False,'dopamineLearning':False,'exactReLUDerivative':True,'learning':'recurrent backpropagation of existing gains/excitabilities',
        'isServiceEvidence':False,'interface':model.interface,'device':torch.cuda.get_device_name()}
    atomic_json(args.out/'manifest.json',manifest)
    atomic_json(args.out/'status.json',{'finished':False,'acceptedUpdates':0,'parameterHash':before})
    accepted=[]
    def record(update,row):
        row['seconds']=time.perf_counter()-began
        atomic_json(args.out/f'attempt-{update}.json',row)
        if row['accepted']:
            save_model(args.out/f'candidate-{update}.npz',model)
            atomic_json(args.out/f'solver-{update}.json',{'updates':update,'parameterHash':model.checkpoint_hash(),
                'rng':rng.bit_generator.state,'optimizer':manifest['optimizer'],'lastScale':row['trials'][-1]['scale']})
            accepted.append(update)
        atomic_json(args.out/'status.json',{'finished':False,'acceptedUpdates':len(accepted),'lastAttempt':update,
            'parameterHash':model.checkpoint_hash(),'lastAcceptedVersion':accepted[-1] if accepted else 0})
        print('SELECTIVE_TRAINING_UPDATE '+json.dumps({'update':update,'accepted':row['accepted'],
            'parameterHash':model.checkpoint_hash(),'seconds':row['seconds']}),flush=True)
    try:
        initial,history=train(model,bank,fresh,rng,args.updates,record,lambda fit:atomic_json(args.out/'fit-0.json',fit))
        if any(file_hash(Path(p))!=h for p,h in hashes.items()) or any(file_hash(Path(__file__).parent/n)!=h for n,h in sources.items()):
            raise AssertionError('Original source/data/checkpoint changed')
        result={'experiment':VERSION,'parentParameterHash':before,'finalParameterHash':model.checkpoint_hash(),
            'acceptedVersions':accepted,'initialTrainingFit':initial,'history':history,'sourceFilesUnchanged':True,
            'seconds':time.perf_counter()-began,'audit':model.audit(original_gains),
            'changedNeurons':int(np.count_nonzero(model.tonic.detach().cpu().numpy()!=original_tonic)),
            'earlyStop':None if all(r['accepted'] for r in history) else 'No guarded step passed; rejected weights restored',
            'requiresIndependentVerification':True}
        atomic_json(args.out/'result.json',result)
        atomic_json(args.out/'status.json',{'finished':True,'acceptedUpdates':len(accepted),'parameterHash':model.checkpoint_hash(),
            'lastAcceptedVersion':accepted[-1] if accepted else 0})
        print('SELECTIVE_TRAINING_FINISHED '+json.dumps({k:result[k] for k in ('acceptedVersions','earlyStop','finalParameterHash')}),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error),'lastAcceptedVersion':accepted[-1] if accepted else 0})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','source','probe','cache','original-cache','dataset','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,default=8)
    main(p.parse_args())
