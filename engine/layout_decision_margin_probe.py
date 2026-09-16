"""Reversible test of decision-preserving training guards on rejected V27 step5.

Keep every formerly correct decision and its small confidence cushion, instead
of requiring every positive-context MEAN to grow. No inference rule changes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_context_step_probe import direction,trust_scale,scores
from .layout_selective_jacobian_train import context_gate,diagnostic_gate,CAPS,BACKTRACK,MOTION_BUDGET
from .layout_decision_resume import AnchoredOperationCache
from .layout_operation_sampling import OperationCache
from .layout_operation_prefix import CONTEXTS
from .layout_operation_focus import focus_values,focus_fit
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def decision_gate(before,after,reference_motion,selections,cushion=.01):
    old,new=np.asarray(before['rawGrip']),np.asarray(after['rawGrip'])
    if (old.shape!=new.shape or old.shape!=(len(selections),) or not np.isfinite(old).all()
            or not np.isfinite(new).all() or not 0<cushion<.1):raise ValueError('Original finite aligned decisions required')
    want=np.asarray([s['category']!=CONTEXTS[2] for s in selections])
    if any(s['category'] not in CONTEXTS for s in selections):raise ValueError('Unknown operation context')
    correct=(old>1.)==want
    if np.any(correct&((new>1.)!=want)):return False
    floor=np.where(old>1.,np.minimum(old,1.+cushion),old)
    ceiling=np.where(old<=1.,np.maximum(old,1.-cushion),old)
    preserve=np.where(want,new>=floor-1e-7,new<=ceiling+1e-7)
    return bool(preserve.all() and after['heads'][2]<=.99*before['heads'][2]
        and after['recall']>=before['recall'] and after['falsePositiveRate']<=before['falsePositiveRate']
        and all(after['heads'][h]<=reference_motion[h]+MOTION_BUDGET[h] for h in range(2)))


def compare_rejected(model,bank,fresh,rejected,initial_fit,current_fit):
    before,fixed,mode=model.checkpoint_hash(),model.fingerprint(),model.training
    if model.surrogate_training or before!=rejected['beforeHash'] or rejected['accepted']:
        raise ValueError('Exact original rejected-step candidate required')
    params=(model.log_gains,model.tonic);base=[p.detach().clone() for p in params]
    ids=rejected['windowIndexes'];selections=[bank.manifest['windows'][i] for i in ids]
    if selections!=rejected['selections']:raise ValueError('Original physical selections changed')
    x,y,state,motion=bank.batch(ids,model.tonic.device)
    reference=(fresh.batch(ids,model.tonic.device)[3]-motion).square().mean((0,1)).cpu().tolist()
    try:
        model.train()
        p=recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        initial=scores(p,y[64:],motion,selections)
        for key in ('rawGrip','heads','meanRawGripByContext'):
            np.testing.assert_allclose(initial[key],rejected['initialScores'][key],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(reference,rejected['referenceParentMotionMSE'],rtol=1e-5,atol=1e-7)
        grip,_=focus_values(p,y[64:],selections)
        outputs=[grip[[i for i,s in enumerate(selections) if s['category']==c]].mean() for c in CONTEXTS]
        outputs += [p[...,h].mean() for h in range(2)]
        values=[float(o.detach()) for o in outputs]
        residual=[max(0.,1.1-v) if i!=2 else min(0.,.9-v) for i,v in enumerate(values[:4])]+[0.,0.]
        gradients=[]
        for i,o in enumerate(outputs):
            gradients.append([g.detach() for g in torch.autograd.grad(o,params,retain_graph=i<5)])
            print(json.dumps({'rejectedStepJacobianRows':i+1,'permanentUpdates':False}),flush=True)
        delta,linear=direction(gradients,residual)
        np.testing.assert_allclose(linear['predictedUnboundedOutputChange'],rejected['linearProposal']['predictedUnboundedOutputChange'],rtol=1e-4,atol=1e-5)
        del p,grip,outputs,gradients
        bounded=trust_scale(delta,CAPS);rows=[]
        model.eval()
        with torch.no_grad():
            for scale in BACKTRACK:
                for param,original,d in zip(params,base,delta):param.copy_(original+bounded*scale*d)
                params[0].clamp_(-2.,2.);params[1].clamp_(-.1,.1)
                measured=scores(frozen_predictions(model,x,state,64),y[64:],motion,selections)
                old_pass=context_gate(initial,measured,reference)
                decision_pass=decision_gate(initial,measured,reference,selections)
                fit=focus_fit(model,bank) if decision_pass else None
                passed=decision_pass and diagnostic_gate(initial_fit,current_fit,fit)
                row={'backtrack':scale,'oldMeanGuard':old_pass,'decisionGuard':decision_pass,'allTrainingGuards':passed,
                     'measured':measured,'fixedDiagnostic':fit,'temporaryParameterHash':model.checkpoint_hash()}
                rows.append(row);print('DECISION_MARGIN_TRIAL '+json.dumps(row),flush=True)
        return {'parentParameterHash':before,'fixedHash':fixed,'originalRejectedStepReproduced':True,
            'storedPrefixAgeUpdates':4,'prefixIsExactForCandidate':False,'initial':initial,'trials':rows,
            'candidateSaved':False,'parametersRestored':True,'optimizerUsed':False,'isServiceEvidence':False,
            'cushion':.01,'changedRule':'Training acceptance guard only; original model, labels, histories and Jacobian proposal unchanged'}
    finally:
        with torch.no_grad():
            for param,original in zip(params,base):param.copy_(original)
        model.train(mode)
        if model.checkpoint_hash()!=before or model.fingerprint()!=fixed or any(p.grad is not None for p in params):
            raise AssertionError('Original brain was not restored')


def main(args):
    if args.out.exists():raise FileExistsError('Preserve prior probe')
    m,r,s=[json.loads((args.source/n).read_text()) for n in ('manifest.json','result.json','status.json')]
    rejected=json.loads((args.source/'attempt-5.json').read_text())
    if (not s['finished'] or r['acceptedVersions']!=[1,2,3,4] or r['history'][-1]!=rejected
            or not r['sourceFilesUnchanged'] or rejected['beforeHash']!=r['finalParameterHash']
            or (args.source/'failure.json').exists() or (args.source/'candidate-5.npz').exists()):
        raise ValueError('Original completed/rejected V27 publication required')
    paths=[args.source/n for n in ('manifest.json','result.json','status.json','attempt-5.json','fit-0.json','candidate-4.npz')]
    hashes={**m['inputFileHashes'],**{str(p.resolve()):file_hash(p) for p in paths}}
    for p,h in hashes.items():
        if file_hash(Path(p))!=h:raise ValueError('Original input changed')
    for n,h in m['sourceHashes'].items():
        if file_hash(Path(__file__).parent/n)!=h:raise ValueError('Original trainer source changed')
    for p in (args.cache/'cache.npz',args.cache/'manifest.json',args.original_cache/'cache.npz',args.original_cache/'manifest.json',args.dataset/'manifest.json'):
        if hashes.get(str(p.resolve()))!=file_hash(p):raise ValueError('Not original cache/data')
    torch.set_num_threads(4)
    model=load_model(args.root,args.source/'candidate-4.npz').train()
    if model.checkpoint_hash()!=r['finalParameterHash']:raise ValueError('Wrong original final brain')
    fresh=OperationCache(args.cache,args.dataset,m['parentParameterHash'],model.fixed_hash,model.n)
    original=OperationCache(args.original_cache,args.dataset,m['originalMotionAnchorHash'],model.fixed_hash,model.n)
    bank=AnchoredOperationCache(fresh,original)
    initial=json.loads((args.source/'fit-0.json').read_text());current=r['history'][-2]['trainingFit']
    sources={**m['sourceHashes'],Path(__file__).name:file_hash(Path(__file__))}
    report=compare_rejected(model,bank,fresh,rejected,initial,current)
    if any(file_hash(Path(p))!=h for p,h in hashes.items()) or any(file_hash(Path(__file__).parent/n)!=h for n,h in sources.items()):
        raise AssertionError('Read-only probe changed original source/inputs')
    report.update(inputFileHashes=hashes,sourceHashes=sources)
    atomic_json(args.out,report)
    print('DECISION_MARGIN_PROBE_COMPLETE '+json.dumps({'passingBacktracks':[r['backtrack'] for r in report['trials'] if r['allTrainingGuards']],
          'parametersRestored':True,'isServiceEvidence':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','source','cache','original-cache','dataset','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
