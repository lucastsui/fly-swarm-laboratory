"""One current-data pickup step; save the exact measured, guarded snapshot.

No stale experiment identities, new decoder, optimizer continuation or automatic
deployment. The selected/diagnostic guards are unchanged. Full-bank measurements
are also published, but do not substitute for frozen closed-loop verification.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_context_step_probe import direction, trust_scale
from .layout_pickup_preservation_probe import measure, measure_bank, preservation_gate, CAPS, MOTION_BUDGET, LOADED_GRIP_BUDGET
from .layout_operation_prefix import KINDS, CONTEXTS
from .layout_operation_sampling import OperationCache
from .layout_operation_focus import focus_values
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_loaded_transition_train import same_measure
from .layout_guarded_snapshot_train import verify_saved_snapshot
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

ALGORITHM = 'single-current-onpolicy-pickup-snapshot-v1'


def apply_step(model, bank, published):
    before, fixed, mode = model.checkpoint_hash(), model.fingerprint(), model.training
    params = (model.log_gains, model.tonic)
    base = [p.detach().clone() for p in params]
    if (before != published['parentParameterHash'] or before != bank.manifest['parameterHash']
            or before != bank.manifest['behaviorParameterHash'] or fixed != published['fixedHash']
            or model.surrogate_training or any(p.grad is not None for p in params)):
        raise ValueError('Exact current on-policy parent, fixed state and derivative required')
    passing = [t for t in published['trials'] if t['allTrainingGuards']]
    if not passing:
        raise ValueError('No original guarded proposal; do not update')
    chosen = passing[0]
    ids = [bank.groups[k,c][0] for k in KINDS for c in CONTEXTS]
    diagnostic_ids = bank.diagnostic_indexes()
    selections = [bank.manifest['windows'][i] for i in ids]
    if (ids != published['windowIndexes'] or selections != published['selections']
            or diagnostic_ids != published['diagnosticWindowIndexes']):
        raise ValueError('Original selected physical histories required')
    committed = False
    try:
        model.eval()
        diagnostic = measure_bank(model, bank, diagnostic_ids)
        same_measure(diagnostic, published['initialDiagnostic'])
        x,y,state,motion = bank.batch(ids, model.tonic.device)
        model.train()
        predictions = recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        initial = measure(predictions,y[64:],x[64:],motion,selections)
        same_measure(initial,published['initial'])
        grip,_ = focus_values(predictions,y[64:],selections)
        outputs = list(grip.unbind())+[predictions[...,h].mean() for h in range(2)]
        residual = [max(0.,1.1-float(grip[i].detach())) if s['category']==CONTEXTS[0] else 0.
                    for i,s in enumerate(selections)]+[0.,0.]
        gradients = []
        for i,output in enumerate(outputs):
            gradients.append([g.detach() for g in torch.autograd.grad(output,params,retain_graph=i<len(outputs)-1)])
            print(json.dumps({'onPolicyTrainingJacobianRows':i+1,'total':len(outputs),'maximumUpdates':1}),flush=True)
        delta,linear = direction(gradients,residual)
        del predictions,grip,outputs,gradients
        for k in ('requestedOutputChange','predictedUnboundedOutputChange'):
            np.testing.assert_allclose(linear[k],published['linearProposal'][k],rtol=1e-4,atol=1e-5)
        scale = trust_scale(delta,CAPS)*chosen['backtrack']
        np.testing.assert_allclose(scale,chosen['scale'],rtol=1e-6,atol=1e-12)
        model.eval()
        with torch.no_grad():
            for p,b,d in zip(params,base,delta):p.copy_(b+scale*d)
            params[0].clamp_(-2.,2.);params[1].clamp_(-.1,.1)
            after = measure(frozen_predictions(model,x,state,64),y[64:],x[64:],motion,selections)
            fit = measure_bank(model,bank,diagnostic_ids)
        same_measure(after,chosen['measured']);same_measure(fit,chosen['diagnostic'])
        if not preservation_gate(initial,after) or not preservation_gate(diagnostic,fit,0.):
            raise ValueError('Original measured training guards failed')
        changes = [float((p.detach()-b).abs().max()) for p,b in zip(params,base)]
        if (any(c>limit+1e-7 for c,limit in zip(changes,CAPS)) or model.checkpoint_hash()==before
                or any(not torch.isfinite(p).all() for p in params)):
            raise ValueError('Original parameter bounds or finite changed snapshot required')
        committed = True
        return {'beforeHash':before,'afterHash':model.checkpoint_hash(),'backtrack':chosen['backtrack'],
                'scale':scale,'initial':initial,'after':after,'initialDiagnostic':diagnostic,'afterDiagnostic':fit,
                'windowIndexes':ids,'diagnosticWindowIndexes':diagnostic_ids,'linearProposal':linear,
                'maximumParameterChanges':changes,'allOriginalTrainingGuardsPassed':True,
                'previousProbeTemporaryHash':chosen['temporaryParameterHash'],
                'crossRunByteIdentityRequired':False,'isServiceEvidence':False}
    finally:
        if not committed:
            with torch.no_grad():
                for p,b in zip(params,base):p.copy_(b)
        model.train(mode)
        if (model.fingerprint()!=fixed or any(p.grad is not None for p in params)
                or (not committed and model.checkpoint_hash()!=before)):
            raise AssertionError('Fixed-state/gradient/rollback invariant failed')


def validate_probe(manifest,published):
    if (not published['parametersRestored']
            or published['candidateSaved'] or published['optimizerUsed'] or not published['sourceFilesUnchanged']
            or published['inputFileHashes']!=manifest['inputFileHashes']
            or published['sourceHashes']!=manifest['sourceHashes']
            or ('exactReLUDerivative' in published and published['exactReLUDerivative'] is not True)):
        raise ValueError('Complete original reversible probe required')
    # Original pickup probe records the exact recurrent horizon, not a surrogate.
    if not published['prefixExactAtInitialization'] or published['gradientFrames']!=96:
        raise ValueError('Exact current recurrent prefix and original gradient horizon required')


def main(args):
    if args.out.exists():raise FileExistsError('Preserve every original candidate')
    if (args.probe/'failure.json').exists():raise ValueError('Failed original probe')
    manifest = json.loads((args.probe/'manifest.json').read_text())
    published = json.loads((args.probe/'result.json').read_text())
    validate_probe(manifest,published)
    paths = {**published['inputFileHashes'],**{str((args.probe/n).resolve()):file_hash(args.probe/n)
              for n in ('manifest.json','result.json')}}
    names = set(published['sourceHashes']) | {Path(__file__).name,'layout_loaded_transition_train.py','layout_guarded_snapshot_train.py'}
    sources = {n:file_hash(Path(__file__).with_name(n)) for n in names}
    if any(sources[n]!=h for n,h in published['sourceHashes'].items()):
        raise ValueError('Original probe source changed')
    def unchanged():
        if (any(file_hash(Path(p))!=h for p,h in paths.items())
                or any(file_hash(Path(__file__).with_name(n))!=h for n,h in sources.items())):
            raise ValueError('Original source/input changed')
    unchanged()
    for p in (args.candidate,args.cache/'manifest.json',args.cache/'cache.npz',args.dataset/'manifest.json'):
        if paths.get(str(p.resolve()))!=file_hash(p):raise ValueError('Wrong original input path')
    torch.set_num_threads(4)
    model = load_model(args.root,args.candidate).train()
    bank = OperationCache(args.cache,args.dataset,published['parentParameterHash'],model.fixed_hash,model.n)
    gains,tonic = [p.detach().cpu().numpy().copy() for p in (model.log_gains,model.tonic)]
    args.out.mkdir(parents=True)
    record = {'algorithm':ALGORITHM,'maximumUpdates':1,'canonicalLearner':'Spark1 only','parentAdamConsumed':False,
              'optimizerState':'New stateless18-output solve; exact tested snapshot is saved; no RNG',
              'parentParameterHash':model.checkpoint_hash(),'fixedHash':model.fixed_hash,
              'inputFileHashes':paths,'sourceHashes':sources,'exactReLUDerivative':True,'gradientFrames':96,
              'prefixAgeUpdates':0,'onPolicy':True,'teacherAtInference':False,'decoderTrained':False,
              'externalDecisionNetwork':False,'dopamineLearning':False,'coordinateCaps':list(CAPS),
              'motionBudget':list(MOTION_BUDGET),'loadedGripBudget':LOADED_GRIP_BUDGET,
              'physicalAndTrainingGuardsUnchanged':True,'isServiceEvidence':False}
    atomic_json(args.out/'manifest.json',record)
    atomic_json(args.out/'status.json',{'finished':False,'acceptedUpdates':0})
    began = time.perf_counter()
    try:
        before_all = measure_bank(model,bank,list(range(len(bank.manifest['windows']))))
        row = apply_step(model,bank,published)
        after_all = measure_bank(model,bank,list(range(len(bank.manifest['windows']))))
        row.update(initialAllWindows=before_all,afterAllWindows=after_all,
                   fullBankPreservationDiagnostic=preservation_gate(before_all,after_all,0.))
        unchanged()
        path=args.out/'candidate-1.npz';save_model(path,model);verify_saved_snapshot(path,model)
        unchanged();atomic_json(args.out/'step-1.json',row)
        result={'algorithm':ALGORITHM,'acceptedUpdates':1,'finalParameterHash':model.checkpoint_hash(),
                'candidateFileHash':file_hash(path),'savedArraysExactlyMatchTestedSnapshot':True,
                'audit':model.audit(gains),'changedNeurons':int(np.count_nonzero(tonic!=model.tonic.detach().cpu().numpy())),
                'seconds':time.perf_counter()-began,'sourceFilesUnchanged':True,'isServiceEvidence':False,
                'requiresFrozenPhysicalVerification':True,'automaticContinuation':False,
                'fullBankPreservationDiagnostic':row['fullBankPreservationDiagnostic']}
        atomic_json(args.out/'result.json',result)
        atomic_json(args.out/'status.json',{'finished':True,'acceptedUpdates':1,'parameterHash':model.checkpoint_hash()})
        print('ONPOLICY_SNAPSHOT_COMPLETE '+json.dumps(result),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','candidate','cache','dataset','probe','out'):p.add_argument('--'+n,required=True,type=Path)
    main(p.parse_args())
