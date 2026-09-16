"""One canonical pooled update, with an additional candidate-full-history gate.

Existing box-solver, per-context/action/motion gates and physical criteria are
unchanged. A provisional update is rolled back unless its OWN recurrent history
also passes. Saving happens only after that check, never before it.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_operation_pool import OperationPool
from .layout_correction_prefix import load_histories
from .layout_box_constrained_train import attempt
from .layout_candidate_history_guard import candidate_history_guard
from .layout_margin_projected_train import margin_gate
from .layout_prefix_transfer_audit import prefix_difference
from .layout_recovery_brain import load_model, save_model
from .layout_guarded_snapshot_train import verify_saved_snapshot
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

ALGORITHM = 'one-pooled-box-update-with-candidate-full-history-gate-v1'


def guarded_attempt(model, bank, episodes, on_initial=lambda value:None,
                    on_provisional=lambda value:None, solver_iterations=1024):
    before = model.checkpoint_hash()
    params = (model.log_gains, model.tonic)
    base = [p.detach().clone() for p in params]
    committed = False
    try:
        proposal = attempt(model,bank,on_initial,solver_iterations=solver_iterations)
        on_provisional(proposal)
        history = candidate_history_guard(model,bank,episodes,proposal['initialFullBank']) if proposal['accepted'] else None
        committed = bool(proposal['accepted'] and history['gate']['passed'])
        return {'proposal':proposal,'candidateHistory':history,'accepted':committed,
                'beforeHash':before,'afterHash':model.checkpoint_hash() if committed else before,
                'isServiceEvidence':False}
    finally:
        if not committed:
            with torch.no_grad():
                for p,b in zip(params,base):p.copy_(b)
        if not committed and model.checkpoint_hash()!=before:
            raise AssertionError('History rejection did not restore original weights')


def main(args):
    if args.out.exists():raise FileExistsError('Preserve previous phases')
    # Require the original controlled evidence motivating this extra safeguard.
    audit_names=('manifest','result','parent-own-prefix','candidate-old-prefix','candidate-own-prefix')
    a,r,b,old,fresh=[json.loads((args.prefix_audit/(n+'.json')).read_text()) for n in audit_names]
    if any(file_hash(Path(__file__).with_name(n))!=h for n,h in a['sourceHashes'].items()):
        raise ValueError('Original diagnostic source changed')
    if (a['candidateParameterHash']!=args.expected_parameter_hash or not r['parametersUnchanged']
            or r['optimizerUsed'] or r['isServiceEvidence']
            or margin_gate(b,old,0.)!=r['originalFullBankGateWithOldPrefix']
            or margin_gate(b,fresh,0.)!=r['originalFullBankGateWithOwnPrefix']
            or prefix_difference(old,fresh)!=r['prefixOnlyOutputDifference']
            or not r['originalFullBankGateWithOldPrefix']['passed']
            or r['originalFullBankGateWithOwnPrefix']['parentMotionWithinBudget']):
        raise ValueError('Verified original prefix-drift diagnostic required')
    torch.set_num_threads(4)
    model=load_model(args.root,args.candidate).train()
    if model.checkpoint_hash()!=args.expected_parameter_hash or model.fixed_hash!=a['fixedHash']:
        raise ValueError('Wrong canonical parent')
    bank=OperationPool(args.caches,args.datasets,args.expected_parameter_hash,model.fixed_hash,model.n)
    episodes=[]
    for folder in args.datasets:
        items,_=load_histories(folder,replay=False);episodes.extend(items)
    paths=[args.candidate]+[args.prefix_audit/(n+'.json') for n in audit_names]
    paths += [c/n for c in args.caches for n in ('manifest.json','cache.npz')]
    for d,m in zip(args.datasets,bank.manifest['sourceBanks']):
        paths += [d/'manifest.json']+[d/e['file'] for e in m['datasetFiles']]
    hashes={str(p.resolve()):file_hash(p) for p in paths}
    names=('layout_history_guarded_train.py','layout_candidate_history_guard.py','layout_operation_pool.py',
           'layout_operation_sampling.py','layout_box_constrained_train.py','layout_box_solver.py',
           'layout_margin_projected_train.py','layout_contextual_operation_probe.py',
           'layout_context_step_probe.py','layout_operation_focus.py','layout_recovery_train.py',
           'layout_demonstration_step_probe.py','layout_guarded_snapshot_train.py','layout_recovery_protocol.py',
           'layout_decision_resume.py','layout_prefix_transfer_audit.py')
    sources={**a['sourceHashes'],**{n:file_hash(Path(__file__).with_name(n)) for n in names}}
    for m in bank.manifest['sourceBanks']:sources.update(m['sourceHashes'])
    def unchanged():
        if (any(file_hash(Path(p))!=h for p,h in hashes.items())
                or any(file_hash(Path(__file__).with_name(n))!=h for n,h in sources.items())):
            raise ValueError('Original source/input changed')
    unchanged()
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json',{'algorithm':ALGORITHM,'canonicalLearner':'Spark1 only',
        'maximumUpdates':1,'parentParameterHash':model.checkpoint_hash(),'fixedHash':model.fixed_hash,
        'inputFileHashes':hashes,'sourceHashes':sources,'pool':bank.manifest,
        'exactReLUDerivative':True,'existingTrainingGuardsUnchanged':True,'physicalReleaseCriteriaChanged':False,
        'candidateFullHistoryRequired':True,'isServiceEvidence':False,'teacherAtInference':False,
        'originalSensoryLabelsAndFixedDecoder':True,'newExternalDecisionNetwork':False})
    gains,tonic=[p.detach().cpu().numpy().copy() for p in (model.log_gains,model.tonic)]
    began=time.perf_counter()
    atomic_json(args.out/'status.json',{'finished':False,'acceptedUpdates':0})
    try:
        row=guarded_attempt(model,bank,episodes,
            lambda value:atomic_json(args.out/'initial-all-window-measure.json',value),
            lambda value:atomic_json(args.out/'provisional-step.json',value))
        unchanged();atomic_json(args.out/'step-1.json',row)
        result={'algorithm':ALGORITHM,'acceptedUpdates':int(row['accepted']),
                'finalParameterHash':model.checkpoint_hash(),'seconds':time.perf_counter()-began,
                'sourceFilesUnchanged':True,'isServiceEvidence':False,'automaticContinuation':False}
        if row['accepted']:
            audit=model.audit(gains)
            if not all(audit[k] for k in ('finite','signsPreserved','fixedGraphSensoryDecoderDynamicsUnchanged')):
                raise ValueError('Fixed signed brain audit failed')
            p=args.out/'candidate-1.npz';save_model(p,model);verify_saved_snapshot(p,model);unchanged()
            result.update(candidateFileHash=file_hash(p),savedArraysExactlyMatchTestedSnapshot=True,audit=audit,
                changedNeurons=int(np.count_nonzero(tonic!=model.tonic.detach().cpu().numpy())),
                candidateFullHistoryGuardPassed=True,requiresFrozenPhysicalVerification=True)
        else:
            result.update(candidateSaved=False,parametersRestored=True,
                          earlyStop='Original proposal or additional full-history gate rejected this update')
        atomic_json(args.out/'result.json',result)
        atomic_json(args.out/'status.json',{'finished':True,'acceptedUpdates':int(row['accepted']),
                                          'parameterHash':model.checkpoint_hash()})
        print('HISTORY_GUARDED_PHASE_COMPLETE '+json.dumps(result),flush=True)
    except BaseException as error:
        with torch.no_grad():
            model.log_gains.copy_(torch.as_tensor(gains,device=model.log_gains.device))
            model.tonic.copy_(torch.as_tensor(tonic,device=model.tonic.device))
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','candidate','prefix-audit','out'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('caches','datasets'):p.add_argument('--'+n,type=Path,nargs=2,required=True)
    p.add_argument('--expected-parameter-hash',required=True)
    main(p.parse_args())
