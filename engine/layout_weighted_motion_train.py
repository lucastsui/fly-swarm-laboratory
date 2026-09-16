"""One bounded update with stronger per-window movement TRAINING penalties.

V48 reduced own-history turning drift but still failed the unchanged budget.
This experiment gives its 32 movement rows 16x squared-residual weight (4x row
scale). All 16 grip targets, exact dynamics, and nonlinear gates are unchanged.
This is an optimizer revision, not a controller or proof of physical service.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_weighted_box_solver import bounded_direction
from .layout_contextual_operation_probe import measure, measure_bank, CAPS, MOTION_BUDGET, BACKTRACK
from .layout_operation_prefix import KINDS, CONTEXTS
from .layout_operation_focus import focus_values
from .layout_operation_pool import OperationPool
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_candidate_history_guard import candidate_history_guard
from .layout_correction_prefix import load_histories
from .layout_margin_projected_train import margin_gate
from .layout_recovery_brain import load_model, save_model
from .layout_guarded_snapshot_train import verify_saved_snapshot
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

ALGORITHM = 'one-pooled-weighted-window-motion-update-with-full-history-gate-v1'
MOTION_PENALTY_WEIGHT = 16.
PRIOR_HASHES = {
    "manifest": "f47993e6782c72c747d8c82efbc3167d13fe5b3ec2a04403cffb85d3265717a0",
    "initial-all-window-measure": "b9386de9f98f7c091255720280d389efd7be26e6135c9a643511f3efd564d506",
    "provisional-step": "74a26d8f099b2f26de39c7bee8818b5b3552e4da41789ec9ada1c635b7ec66c1",
    "step-1": "e84fe2d7d90c5d195008cbde18f2b6d77391deabff40b44bcbcc8e731a7dbe16",
    "result": "b708a02a48f85e11a9f3ab2a77f204a1f301db19be6c625c56fd09a0d17a59d6",
    "status": "bd41393c862d762d09850d66ad5ea3b8b8432e63af5429d62b7e6dad52458ce7"
}


def window_outputs(predictions, grip, selections):
    """16 grip rows, then (forward, turn) for each of the 16 actor windows."""
    if (predictions.shape != (32, 16, 3) or grip.shape != (16,)
            or [(s['kind'], s['category']) for s in selections] != [(k,c) for k in KINDS for c in CONTEXTS]):
        raise ValueError('Original ordered 16 family/context windows required')
    outputs = list(grip.unbind())
    residual = [max(0., 1.1-float(grip[i].detach())) if s['category'] == CONTEXTS[0]
                else min(0., .9-float(grip[i].detach())) if s['category'] == CONTEXTS[2] else 0.
                for i,s in enumerate(selections)]
    rows = [{'kind':'grip-focus', 'selectedWindow':i, 'head':2, 'frame':16} for i in range(16)]
    for i in range(16):
        for head in range(2):
            outputs.append(predictions[:,i,head].mean())
            residual.append(0.)
            rows.append({'kind':'window-motion-mean', 'selectedWindow':i, 'head':head, 'frames':32})
    if not np.isfinite(residual).all() or not torch.isfinite(torch.stack(outputs)).all():
        raise ValueError('Finite original output constraints required')
    return outputs, residual, rows


def active_rows(gradients, residual, rows):
    """Omit only identically zero motion rows with an already-zero target.

Such a row contributes a constant zero to the linear objective. Never omit a
grip correction, a tiny nonzero derivative, or an impossible nonzero target.
All window-wise nonlinear measurements and full-history guards still run.
"""
    if not gradients or not len(gradients) == len(residual) == len(rows):
        raise ValueError('Aligned output constraints required')
    active, dropped = [], []
    for i,(gradient,target,row) in enumerate(zip(gradients,residual,rows)):
        if len(gradient) != 2 or not np.isfinite(target) or any(not torch.isfinite(g).all() for g in gradient):
            raise ValueError('Finite two-block Jacobian required')
        if any(torch.count_nonzero(g).item() for g in gradient):
            active.append(i)
        elif row['kind'] == 'window-motion-mean' and target == 0.:
            dropped.append(i)
        else:
            raise ValueError('Zero grip/correction gradient cannot be discarded')
    if not active:
        raise ValueError('Nonzero trainable outputs required')
    return [gradients[i] for i in active], [residual[i] for i in active], {
        'activeRowIndexes':active, 'omittedExactZeroMotionRows':dropped,
        'rows':rows, 'originalRequestedOutputChange':residual,
        'gripRowsNeverOmitted':True, 'omissionChangesLinearObjective':False}


def attempt(model, bank, on_initial=lambda value:None, solver_iterations=1024):
    before, fixed, mode = model.checkpoint_hash(), model.fingerprint(), model.training
    params = (model.log_gains, model.tonic)
    if (before != bank.manifest['parameterHash'] or before != bank.manifest['behaviorParameterHash']
            or model.surrogate_training or any(p.grad is not None for p in params)):
        raise ValueError('Exact current on-policy parent and derivative required')
    base = [p.detach().clone() for p in params]
    ids = [bank.groups[k,c][0] for k in KINDS for c in CONTEXTS]
    selections = [bank.manifest['windows'][i] for i in ids]
    all_ids = list(range(len(bank.manifest['windows'])))
    committed = False
    try:
        model.eval(); full_before = measure_bank(model,bank,all_ids); on_initial(full_before)
        x,y,state,motion = bank.batch(ids,model.tonic.device)
        model.train()
        p = recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        initial = measure(p,y[64:],x[64:],motion,selections)
        grip,_ = focus_values(p,y[64:],selections)
        outputs,residual,rows = window_outputs(p,grip,selections)
        gradients = []
        for i,output in enumerate(outputs):
            gradients.append([g.detach() for g in torch.autograd.grad(output,params,retain_graph=i<len(outputs)-1)])
            print(json.dumps({'windowMotionJacobianRows':i+1,'total':len(outputs),'maximumUpdates':1}),flush=True)
        del p,grip,outputs
        gradients,residual,constraints = active_rows(gradients,residual,rows)
        row_weights = [MOTION_PENALTY_WEIGHT if constraints['rows'][i]['kind'] == 'window-motion-mean' else 1.
                       for i in constraints['activeRowIndexes']]
        delta,linear = bounded_direction(gradients,residual,base,row_weights=row_weights,iterations=solver_iterations,
            on_progress=lambda value:print(json.dumps(value),flush=True))
        del gradients
        model.eval(); trials = []
        with torch.no_grad():
            for scale in BACKTRACK:
                for param,original,d in zip(params,base,delta):param.copy_(original+scale*d)
                params[0].clamp_(-2.,2.); params[1].clamp_(-.1,.1)
                measured = measure(frozen_predictions(model,x,state,64),y[64:],x[64:],motion,selections)
                selected_gate = margin_gate(initial,measured)
                full_after = measure_bank(model,bank,all_ids) if selected_gate['passed'] else None
                full_gate = margin_gate(full_before,full_after,0.) if full_after is not None else None
                changes = [float((param-original).abs().max()) for param,original in zip(params,base)]
                passed = bool(selected_gate['passed'] and full_gate['passed'])
                if any(change > cap+1e-7 for change,cap in zip(changes,CAPS)):
                    raise ValueError('Coordinate trust bound exceeded')
                row = {'backtrack':scale,'selectedGate':selected_gate,'fullGate':full_gate,
                       'measured':measured,'fullBank':full_after,'accepted':passed,
                       'maximumParameterChanges':changes,'parameterHash':model.checkpoint_hash()}
                trials.append(row)
                print('WINDOW_MOTION_TRIAL '+json.dumps({k:row[k] for k in
                      ('backtrack','selectedGate','fullGate','accepted','maximumParameterChanges')}),flush=True)
                if passed:
                    if row['parameterHash'] == before or model.fingerprint() != fixed:
                        raise ValueError('Finite changed brain on original fixed graph required')
                    committed = True
                    break
        return {'algorithm':ALGORITHM,'beforeHash':before,'afterHash':model.checkpoint_hash() if committed else before,
                'accepted':committed,'windowIndexes':ids,'selections':selections,'initial':initial,
                'initialFullBank':full_before,'boundedLinearSolve':linear,'outputConstraints':constraints,'trials':trials,
                'motionSquaredResidualWeight':MOTION_PENALTY_WEIGHT,
                'fixedHash':fixed,'isServiceEvidence':False,'teacherAtInference':False,
                'v40TrainingGuardsUnchanged':True,'physicalReleaseCriteriaChanged':False,
                'candidateFullHistoryPrefix':False,'sourceLabelsAndSensoryHistoriesUnchanged':True}
    except BaseException:
        committed = False
        raise
    finally:
        if not committed:
            with torch.no_grad():
                for param,original in zip(params,base):param.copy_(original)
        model.train(mode)
        if model.fingerprint() != fixed or any(p.grad is not None for p in params) or (not committed and model.checkpoint_hash() != before):
            raise AssertionError('Fixed-state/gradient/rollback invariant failed')


def guarded_attempt(model,bank,episodes,on_initial=lambda value:None,on_provisional=lambda value:None,solver_iterations=1024):
    before = model.checkpoint_hash()
    params = (model.log_gains,model.tonic)
    base = [p.detach().clone() for p in params]
    committed = False
    try:
        proposal = attempt(model,bank,on_initial,solver_iterations=solver_iterations)
        on_provisional(proposal)
        history = candidate_history_guard(model,bank,episodes,proposal['initialFullBank']) if proposal['accepted'] else None
        committed = bool(proposal['accepted'] and history['gate']['passed'])
        return {'proposal':proposal,'candidateHistory':history,'accepted':committed,
                'beforeHash':before,'afterHash':model.checkpoint_hash() if committed else before,'isServiceEvidence':False}
    finally:
        if not committed:
            with torch.no_grad():
                for param,original in zip(params,base):param.copy_(original)
        if not committed and model.checkpoint_hash() != before:
            raise AssertionError('Full-history rejection did not restore original weights')


def validate_rejection(docs,expected):
    m,initial,p,s,r,status = [docs[n] for n in PRIOR_HASHES]
    if (r['algorithm'] != 'one-pooled-window-motion-update-with-full-history-gate-v1'
            or r['acceptedUpdates'] != 0 or not r['parametersRestored'] or r['candidateSaved']
            or not r['sourceFilesUnchanged'] or r['isServiceEvidence'] or not status['finished']
            or status['acceptedUpdates'] != 0 or s['accepted'] or not p['accepted'] or s['proposal'] != p
            or p['initialFullBank'] != initial or p['beforeHash'] != expected
            or any(v != expected for v in (m['parentParameterHash'],s['beforeHash'],s['afterHash'],
                                           r['finalParameterHash'],status['parameterHash']))):
        raise ValueError('Original completed V48 rejection with exact rollback required')
    accepted = []
    for trial in p['trials']:
        selected = margin_gate(p['initial'],trial['measured'])
        full = margin_gate(initial,trial['fullBank'],0.) if trial['fullBank'] is not None else None
        if selected != trial['selectedGate'] or full != trial['fullGate'] or trial['accepted'] != bool(selected['passed'] and full['passed']):
            raise ValueError('Prior proposal gate changed')
        if trial['accepted']:accepted.append(trial)
    if len(accepted) != 1 or accepted[0] != p['trials'][-1] or p['afterHash'] != accepted[0]['parameterHash']:
        raise ValueError('One exact prior provisional update required')
    history = s['candidateHistory']
    if (history['candidateParameterHash'] != p['afterHash'] or history['parentParameterHash'] != expected
            or not history['candidateFullHistoryPrefix'] or not history['originalMotionTargetsRetained']
            or not history['originalGuardThresholdsUnchanged'] or history['optimizerUsed'] or history['isServiceEvidence']
            or margin_gate(initial,history['measurement'],0.) != history['gate']
            or history['gate']['passed'] or history['gate']['parentMotionWithinBudget']):
        raise ValueError('Original measured full-history movement rejection required')


def main(args):
    if args.out.exists():raise FileExistsError('Preserve previous phases')
    if (args.prior_phase/'failure.json').exists() or (args.prior_phase/'candidate-1.npz').exists():
        raise ValueError('Clean rejection without saved candidate required')
    docs = {}
    for name,expected in PRIOR_HASHES.items():
        path = args.prior_phase/(name+'.json')
        if file_hash(path) != expected:raise ValueError('Original V48 receipt hash mismatch: '+name)
        docs[name] = json.loads(path.read_text())
    validate_rejection(docs,args.expected_parameter_hash)
    prior = docs['manifest']
    hashes = {**prior['inputFileHashes'], **{str((args.prior_phase/(n+'.json')).resolve()):h for n,h in PRIOR_HASHES.items()}}
    names = ('layout_weighted_motion_train.py','test_layout_weighted_motion_train.py',
             'layout_weighted_box_solver.py','test_layout_weighted_box_solver.py')
    sources = {**prior['sourceHashes'],**{n:file_hash(Path(__file__).with_name(n)) for n in names}}
    def unchanged():
        if (any(file_hash(Path(p)) != h for p,h in hashes.items())
                or any(file_hash(Path(__file__).with_name(n)) != h for n,h in sources.items())):
            raise ValueError('Original source/input changed')
    unchanged()
    paths = [args.candidate]+[c/n for c in args.caches for n in ('manifest.json','cache.npz')]+[d/'manifest.json' for d in args.datasets]
    if any(hashes.get(str(p.resolve())) != file_hash(p) for p in paths):
        raise ValueError('Same original checkpoint, caches and physical datasets required')
    torch.set_num_threads(4)
    model = load_model(args.root,args.candidate).train()
    if model.checkpoint_hash() != args.expected_parameter_hash or model.fixed_hash != prior['fixedHash']:
        raise ValueError('Wrong canonical parent')
    bank = OperationPool(args.caches,args.datasets,args.expected_parameter_hash,model.fixed_hash,model.n)
    if bank.manifest != prior['pool']:raise ValueError('Original explicit pool changed')
    episodes = []
    for folder in args.datasets:
        items,_ = load_histories(folder,replay=False); episodes.extend(items)
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json',{'algorithm':ALGORITHM,'canonicalLearner':'Spark1 only','maximumUpdates':1,
        'parentParameterHash':model.checkpoint_hash(),'fixedHash':model.fixed_hash,'inputFileHashes':hashes,
        'sourceHashes':sources,'pool':bank.manifest,'exactReLUDerivative':True,'existingTrainingGuardsUnchanged':True,
        'physicalReleaseCriteriaChanged':False,'candidateFullHistoryRequired':True,'isServiceEvidence':False,
        'teacherAtInference':False,'originalSensoryLabelsAndFixedDecoder':True,'newExternalDecisionNetwork':False,
        'maximumJacobianRows':48,'gripFocusRows':16,'individualWindowMotionRows':32,
        'motionSquaredResidualWeight':MOTION_PENALTY_WEIGHT,'gripSquaredResidualWeight':1.,
        'motionRows':'one 32-frame mean per selected actor window and motor head; no across-actor averaging',
        'zeroRowPolicy':'only exactly zero motion Jacobian with zero target may be omitted; reported explicitly',
        'coordinateCaps':list(CAPS),'motionBudget':list(MOTION_BUDGET),'minimumSelectedImprovement':.01,
        'gradientFrames':96,'scoredFrames':32,'prefixGradientDetached':True,'solverIterationLimit':1024,
        'solverStationarityTolerance':1e-5,'statelessDirection':True,'oldAdamConsumed':False})
    gains,tonic = [p.detach().cpu().numpy().copy() for p in (model.log_gains,model.tonic)]
    began = time.perf_counter()
    atomic_json(args.out/'status.json',{'finished':False,'acceptedUpdates':0})
    try:
        row = guarded_attempt(model,bank,episodes,
            lambda value:atomic_json(args.out/'initial-all-window-measure.json',value),
            lambda value:atomic_json(args.out/'provisional-step.json',value))
        unchanged(); atomic_json(args.out/'step-1.json',row)
        result = {'algorithm':ALGORITHM,'acceptedUpdates':int(row['accepted']),'finalParameterHash':model.checkpoint_hash(),
                  'seconds':time.perf_counter()-began,'sourceFilesUnchanged':True,'isServiceEvidence':False,'automaticContinuation':False}
        if row['accepted']:
            audit = model.audit(gains)
            if (not all(audit[k] for k in ('finite','signsPreserved','fixedGraphSensoryDecoderDynamicsUnchanged'))
                    or model.checkpoint_hash() != row['afterHash']):
                raise ValueError('Same-pass fixed signed brain audit failed')
            path = args.out/'candidate-1.npz'; save_model(path,model); verify_saved_snapshot(path,model); unchanged()
            result.update(candidateSaved=True,candidateFileHash=file_hash(path),savedArraysExactlyMatchTestedSnapshot=True,
                audit=audit,changedNeurons=int(np.count_nonzero(tonic!=model.tonic.detach().cpu().numpy())),
                candidateFullHistoryGuardPassed=True,requiresFrozenPhysicalVerification=True)
        else:
            result.update(candidateSaved=False,parametersRestored=True,
                          earlyStop='Original proposal or unchanged candidate-full-history guard rejected this update')
        atomic_json(args.out/'result.json',result)
        atomic_json(args.out/'status.json',{'finished':True,'acceptedUpdates':int(row['accepted']),'parameterHash':model.checkpoint_hash()})
        print('WINDOW_MOTION_PHASE_COMPLETE '+json.dumps(result),flush=True)
    except BaseException as error:
        with torch.no_grad():
            model.log_gains.copy_(torch.as_tensor(gains,device=model.log_gains.device))
            model.tonic.copy_(torch.as_tensor(tonic,device=model.tonic.device))
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root','candidate','prior-phase','out'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('caches','datasets'):parser.add_argument('--'+name,type=Path,nargs=2,required=True)
    parser.add_argument('--expected-parameter-hash',required=True)
    main(parser.parse_args())
