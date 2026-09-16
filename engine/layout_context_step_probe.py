"""Bounded, reversible damped-Jacobian TRAINING proposal; no deployed weights.

Test whether context-selective changes beat a shared grip-activity shift.
No optimizer state is consumed or reset. All temporary changes are restored.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_context_credit_probe import geometry
from .layout_operation_sampling import OperationCache
from .layout_operation_prefix import CONTEXTS, KINDS
from .layout_decision_resume import AnchoredOperationCache
from .layout_operation_focus import focus_values, focused_heads, focus_fit
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def direction(gradients, residual, scales=(.003,1e-5), damping=.001):
    g = geometry(gradients,scales)
    norms = np.asarray(g['scaledRowNorms'])
    residual = np.asarray(residual,dtype=np.float64)
    if residual.shape != norms.shape or not np.isfinite(residual).all() or not np.isfinite(damping) or damping <= 0:
        raise ValueError('Finite aligned residual and positive damping required')
    if np.any(norms == 0):
        raise ValueError('A zero-gradient output needs a separate experiment')
    matrix = np.asarray(g['scaledCosines'])
    coefficients = np.linalg.solve(matrix+damping*np.eye(len(norms)),residual/norms)/norms
    delta = [sum(gradients[i][b]*float(coefficients[i]*scales[b]**2) for i in range(len(norms))) for b in range(2)]
    if any(not torch.isfinite(d).all() for d in delta):
        raise FloatingPointError('Nonfinite proposal')
    predicted = np.asarray(g['scaledOutputGram'])@coefficients
    return delta, {'damping':damping,'coordinateScales':list(scales),'requestedOutputChange':residual.tolist(),
                   'predictedUnboundedOutputChange':predicted.tolist()}


def trust_scale(delta,caps):
    if len(delta) != 2 or len(caps) != 2 or any(not np.isfinite(c) or c <= 0 for c in caps):
        raise ValueError('Two finite positive per-coordinate trust bounds required')
    if any(not torch.isfinite(d).all() for d in delta):
        raise FloatingPointError('Nonfinite direction')
    return min([1.]+[cap/max(float(d.abs().max()),1e-30) for d,cap in zip(delta,caps)])


def scores(prediction,labels,motion,selections):
    grip,target = focus_values(prediction,labels,selections)
    positive = target > 1.; active = grip > 1.
    return {'heads':focused_heads(prediction,labels,motion,selections).detach().cpu().tolist(),
            'recall':float(active[positive].float().mean()),
            'falsePositiveRate':float(active[~positive].float().mean()),
            'rawGrip':grip.detach().cpu().tolist(),
            'meanRawGripByContext':[float(grip[[i for i,s in enumerate(selections) if s['category']==c]].mean().detach()) for c in CONTEXTS]}


def rollback_probe(model,bank):
    if model.surrogate_training or model.checkpoint_hash() != bank.manifest['parameterHash']:
        raise ValueError('Exact forward/derivative/current full-history prefix required')
    params = (model.log_gains,model.tonic)
    if any(p.grad is not None for p in params): raise ValueError('Accumulated gradients not allowed')
    before, fixed, mode = model.checkpoint_hash(), model.fingerprint(), model.training
    original = [p.detach().clone() for p in params]
    ids = [bank.groups[k,c][0] for k in KINDS for c in CONTEXTS]
    selections = [bank.manifest['windows'][i] for i in ids]
    x,y,state,motion = bank.batch(ids,model.tonic.device)
    try:
        model.train()
        p = recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        initial = scores(p,y[64:],motion,selections)
        grip,_ = focus_values(p,y[64:],selections)
        outputs = [grip[[i for i,s in enumerate(selections) if s['category']==c]].mean() for c in CONTEXTS]
        outputs += [p[...,h].mean() for h in range(2)]
        target = [1.1,1.1,.9,1.1]
        values = [float(o.detach()) for o in outputs]
        residual = [max(0.,t-v) if i!=2 else min(0.,t-v) for i,(t,v) in enumerate(zip(target,values))]+[0.,0.]
        gradients = []
        for i,o in enumerate(outputs):
            gradients.append([g.detach() for g in torch.autograd.grad(o,params,retain_graph=i<len(outputs)-1)])
            print(json.dumps({'proposalJacobianRows':i+1,'parametersPermanentlyUpdated':False}),flush=True)
        delta, linear = direction(gradients,residual)
        del p,grip,outputs,gradients
        trials = []; best = None
        model.eval()
        with torch.no_grad():
            for caps in ((.003,1e-5),(.01,3e-5),(.03,1e-4)):
                bounded = trust_scale(delta,caps)
                for backtrack in (1.,.5,.25):
                    scale = bounded*backtrack
                    for param,base,d in zip(params,original,delta): param.copy_(base+scale*d)
                    params[0].clamp_(-2.,2.); params[1].clamp_(-.1,.1)
                    prediction = frozen_predictions(model,x,state,64)
                    measured = scores(prediction,y[64:],motion,selections)
                    acceptable = (measured['heads'][2] <= .99*initial['heads'][2]
                        and measured['recall'] >= initial['recall']
                        and measured['falsePositiveRate'] <= initial['falsePositiveRate']
                        and measured['heads'][0] <= initial['heads'][0]+1e-4
                        and measured['heads'][1] <= initial['heads'][1]+.001)
                    trial = {'coordinateCaps':list(caps),'backtrack':backtrack,'actualScale':scale,
                             'maximumParameterChanges':[float((param-base).abs().max()) for param,base in zip(params,original)],
                             'measured':measured,'trainingProbePass':acceptable,'isServiceEvidence':False}
                    trials.append(trial)
                    if acceptable and (best is None or measured['heads'][2]<trials[best]['measured']['heads'][2]): best=len(trials)-1
                    print('TEMPORARY_CONTEXT_STEP '+json.dumps(trial),flush=True)
            diagnostic = None
            if best is not None:
                for param,base,d in zip(params,original,delta): param.copy_(base+trials[best]['actualScale']*d)
                params[0].clamp_(-2.,2.); params[1].clamp_(-.1,.1)
                diagnostic = focus_fit(model,bank)
        return {'parameterHash':before,'fixedHash':fixed,'linearProposal':linear,'initial':initial,'trials':trials,
                'selectedTemporaryTrial':best,'selectedTrainingDiagnostic':diagnostic,'windowIndexes':ids,
                'parametersUnchanged':True,'optimizerUsed':False,'candidateSaved':False,'isServiceEvidence':False,
                'physicalActionsExecuted':False,'temporaryChangesAlwaysRestored':True,
                'qualification':'One six-mean-output linearization; candidate proposal only, not evidence of learned service'}
    finally:
        with torch.no_grad():
            for param,base in zip(params,original): param.copy_(base)
        model.train(mode)
        if model.checkpoint_hash()!=before or model.fingerprint()!=fixed or any(p.grad is not None for p in params):
            raise AssertionError('Failed to restore original brain after proposal')


def main(args):
    if args.out.exists(): raise FileExistsError('Preserve previous probe')
    prior=json.loads(args.credit.read_text())
    if (not prior['parametersUnchanged'] or not prior['prefixIsExactForCandidate'] or prior['optimizerUsed']
            or prior['isServiceEvidence'] or prior['candidateSaved']):
        raise ValueError('Original exact read-only credit report required')
    for path,h in prior['inputFileHashes'].items():
        if file_hash(Path(path))!=h: raise ValueError('Credit inputs changed')
    for n,h in prior['sourceHashes'].items():
        if file_hash(Path(__file__).parent/n)!=h: raise ValueError('Credit source changed')
    torch.set_num_threads(4)
    model=load_model(args.root,args.candidate).train()
    if model.checkpoint_hash()!=prior['parameterHash'] or model.fixed_hash!=prior['fixedHash']:
        raise ValueError('Wrong original credit candidate')
    fresh=OperationCache(args.cache,args.dataset,prior['parameterHash'],model.fixed_hash,model.n)
    original=OperationCache(args.original_cache,args.dataset,prior['motionAnchorParameterHash'],model.fixed_hash,model.n)
    if fresh.manifest['candidateFileHash']!=file_hash(args.candidate): raise ValueError('Wrong current prefix')
    bank=AnchoredOperationCache(fresh,original)
    sources={**prior['sourceHashes'],Path(__file__).name:file_hash(Path(__file__)),
             'layout_demonstration_step_probe.py':file_hash(Path(__file__).parent/'layout_demonstration_step_probe.py')}
    hashes={**prior['inputFileHashes'],str(args.credit.resolve()):file_hash(args.credit)}
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json',{'inputFileHashes':hashes,'sourceHashes':sources,'parametersPermanentlyUpdated':False})
    try:
        report=rollback_probe(model,bank)
        if any(file_hash(Path(p))!=h for p,h in hashes.items()) or any(file_hash(Path(__file__).parent/n)!=h for n,h in sources.items()):
            raise AssertionError('Original source/data/brain files changed')
        report.update(inputFileHashes=hashes,sourceHashes=sources)
        atomic_json(args.out/'result.json',report)
        print('CONTEXT_STEP_PROBE_COMPLETE '+json.dumps({'selectedTemporaryTrial':report['selectedTemporaryTrial'],'parametersRestored':True}),flush=True)
    except BaseException as e:
        atomic_json(args.out/'failure.json',{'type':type(e).__name__,'error':str(e)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','candidate','credit','cache','original-cache','dataset','out'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
