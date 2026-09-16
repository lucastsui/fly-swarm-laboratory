"""Commit ONE independently reproducible, previously guarded brain update.

Explicit new stateless18-output damped-Jacobian algorithm, not Adam resume.
No continuation loop: one candidate must receive frozen physical validation.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_pickup_preservation_probe import measure,measure_bank,preservation_gate,CAPS,MOTION_BUDGET,LOADED_GRIP_BUDGET
from .layout_context_step_probe import direction,trust_scale
from .layout_operation_prefix import KINDS,CONTEXTS
from .layout_operation_sampling import OperationCache
from .layout_operation_focus import focus_values
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model,save_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

ALGORITHM='one-verified-individual-output-pickup-step-v1'


def same_measure(a,b):
    if a['loadedMask']!=b['loadedMask']:raise ValueError('Original loaded-history mask changed')
    for key in ('rawGrip','pickupRawGrip','protectedLoadedGrip','pickupHinge','parentMotionMSE'):
        np.testing.assert_allclose(a[key],b[key],rtol=1e-5,atol=1e-6)


def apply_verified_step(model,bank,published):
    before,fixed,mode=model.checkpoint_hash(),model.fingerprint(),model.training
    params=(model.log_gains,model.tonic);base=[p.detach().clone() for p in params]
    if (before!=published['parentParameterHash'] or before!=bank.manifest['parameterHash']
            or fixed!=published['fixedHash'] or model.surrogate_training or any(p.grad is not None for p in params)):
        raise ValueError('Original exact parent, prefix and derivative required')
    trials=[t for t in published['trials'] if t['allTrainingGuards']]
    if not trials:raise ValueError('No verified passing proposal; do not train')
    chosen=trials[0];ids=published['windowIndexes'];diagnostic_ids=published['diagnosticWindowIndexes']
    selections=[bank.manifest['windows'][i] for i in ids]
    if (ids!=[bank.groups[k,c][0] for k in KINDS for c in CONTEXTS]
            or selections!=published['selections'] or diagnostic_ids!=bank.diagnostic_indexes()):
        raise ValueError('Original deterministic selections required')
    committed=False
    try:
        model.eval();diagnostic=measure_bank(model,bank,diagnostic_ids)
        same_measure(diagnostic,published['initialDiagnostic'])
        x,y,state,motion=bank.batch(ids,model.tonic.device)
        model.train();p=recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        initial=measure(p,y[64:],x[64:],motion,selections);same_measure(initial,published['initial'])
        grip,_=focus_values(p,y[64:],selections)
        outputs=list(grip.unbind())+[p[...,h].mean() for h in range(2)]
        residual=[max(0.,1.1-float(grip[i].detach())) if s['category']==CONTEXTS[0] else 0.
                  for i,s in enumerate(selections)]+[0.,0.]
        gradients=[]
        for i,o in enumerate(outputs):
            gradients.append([g.detach() for g in torch.autograd.grad(o,params,retain_graph=i<len(outputs)-1)])
            print(json.dumps({'trainingJacobianRows':i+1,'total':len(outputs),'updatesPending':1}),flush=True)
        delta,linear=direction(gradients,residual)
        del p,grip,outputs,gradients
        for key in ('requestedOutputChange','predictedUnboundedOutputChange'):
            np.testing.assert_allclose(linear[key],published['linearProposal'][key],rtol=1e-4,atol=1e-5)
        scale=trust_scale(delta,CAPS)*chosen['backtrack']
        np.testing.assert_allclose(scale,chosen['scale'],rtol=1e-6,atol=1e-12)
        model.eval()
        with torch.no_grad():
            for param,original,d in zip(params,base,delta):param.copy_(original+scale*d)
            params[0].clamp_(-2.,2.);params[1].clamp_(-.1,.1)
            after=measure(frozen_predictions(model,x,state,64),y[64:],x[64:],motion,selections)
            same_measure(after,chosen['measured'])
            fit=measure_bank(model,bank,diagnostic_ids);same_measure(fit,chosen['diagnostic'])
            if (not preservation_gate(initial,after) or not preservation_gate(diagnostic,fit,0.)
                    or model.checkpoint_hash()!=chosen['temporaryParameterHash'] or model.checkpoint_hash()==before):
                raise ValueError('Original candidate or preservation guards did not reproduce')
            changes=[float((p-b).abs().max()) for p,b in zip(params,base)]
            if any(d>c+1e-7 for d,c in zip(changes,CAPS)):raise ValueError('Parameter trust bounds exceeded')
        committed=True
        return {'beforeHash':before,'afterHash':model.checkpoint_hash(),'backtrack':chosen['backtrack'],
                'scale':scale,'selectedProposalReproducedExactly':True,'windowIndexes':ids,'diagnosticWindowIndexes':diagnostic_ids,
                'initial':initial,'after':after,'initialDiagnostic':diagnostic,'afterDiagnostic':fit,
                'maximumParameterChanges':changes,'linearProposal':linear,'isServiceEvidence':False}
    finally:
        if not committed:
            with torch.no_grad():
                for p,b in zip(params,base):p.copy_(b)
        model.train(mode)
        if (model.fingerprint()!=fixed or any(p.grad is not None for p in params)
                or (not committed and model.checkpoint_hash()!=before)):
            raise AssertionError('Original fixed state or rollback failed')


def main(args):
    if args.out.exists():raise FileExistsError('Preserve canonical experiment')
    source=json.loads((args.probe/'manifest.json').read_text())
    r=json.loads((args.probe/'result.json').read_text())
    if ((args.probe/'failure.json').exists() or not r['parametersRestored'] or r['candidateSaved']
            or r['optimizerUsed'] or not r['sourceFilesUnchanged'] or not r['prefixExactAtInitialization']
            or not any(t['allTrainingGuards'] for t in r['trials']) or r['inputFileHashes']!=source['inputFileHashes']
            or r['sourceHashes']!=source['sourceHashes']):
        raise ValueError('Completed, unchanged and successful original read-only probe required')
    hashes={**r['inputFileHashes'],**{str((args.probe/n).resolve()):file_hash(args.probe/n)
            for n in ('manifest.json','result.json','initial-all-window-fit.json')}}
    for p,h in hashes.items():
        if file_hash(Path(p))!=h:raise ValueError('Original probe inputs changed')
    for n,h in r['sourceHashes'].items():
        if Path(n).name!=n or file_hash(Path(__file__).parent/n)!=h:raise ValueError('Original probe source changed')
    for p in (args.candidate,args.cache/'cache.npz',args.cache/'manifest.json',args.dataset/'manifest.json'):
        if hashes.get(str(p.resolve()))!=file_hash(p):raise ValueError('Wrong candidate/cache/dataset path')
    torch.set_num_threads(4);model=load_model(args.root,args.candidate).train()
    bank=OperationCache(args.cache,args.dataset,r['parentParameterHash'],model.fixed_hash,model.n)
    sources={**r['sourceHashes'],Path(__file__).name:file_hash(Path(__file__))}
    gains=model.log_gains.detach().cpu().numpy().copy();tonic=model.tonic.detach().cpu().numpy().copy()
    args.out.mkdir(parents=True)
    manifest={'algorithm':ALGORITHM,'maximumUpdates':1,'canonicalLearner':'Spark1 only','parentAdamConsumed':False,
        'optimizerState':'New stateless damped18-output Jacobian solve; deterministic original selections; no RNG',
        'parentParameterHash':model.checkpoint_hash(),'fixedHash':model.fixed_hash,'inputFileHashes':hashes,'sourceHashes':sources,
        'exactReLUDerivative':True,'gradientFrames':96,'prefixAgeUpdates':0,'motionBudget':list(MOTION_BUDGET),
        'loadedGripBudget':LOADED_GRIP_BUDGET,'coordinateCaps':list(CAPS),'teacherAtInference':False,'decoderTrained':False,
        'externalDecisionNetwork':False,'originalLabelsUnchanged':True,'dopamineLearning':False,'isServiceEvidence':False}
    atomic_json(args.out/'manifest.json',manifest)
    atomic_json(args.out/'status.json',{'finished':False,'acceptedUpdates':0,'parameterHash':model.checkpoint_hash()})
    began=time.perf_counter()
    try:
        row=apply_verified_step(model,bank,r)
        if any(file_hash(Path(p))!=h for p,h in hashes.items()) or any(file_hash(Path(__file__).parent/n)!=h for n,h in sources.items()):
            raise AssertionError('Original source or input changed')
        audit=model.audit(gains)
        save_model(args.out/'candidate-1.npz',model)
        atomic_json(args.out/'step-1.json',row)
        result={'algorithm':ALGORITHM,'acceptedUpdates':1,'finalParameterHash':model.checkpoint_hash(),
                'candidateFileHash':file_hash(args.out/'candidate-1.npz'),'audit':audit,
                'changedNeurons':int(np.count_nonzero(tonic!=model.tonic.detach().cpu().numpy())),
                'seconds':time.perf_counter()-began,'sourceFilesUnchanged':True,'requiresFrozenPhysicalVerification':True,
                'isServiceEvidence':False,'automaticContinuation':False}
        atomic_json(args.out/'result.json',result)
        atomic_json(args.out/'status.json',{'finished':True,'acceptedUpdates':1,'parameterHash':model.checkpoint_hash()})
        print('PICKUP_SINGLE_STEP_TRAINING_COMPLETE '+json.dumps(result),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','candidate','cache','dataset','probe','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
