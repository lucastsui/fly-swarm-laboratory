"""One bounded brain-only step with constraints inside the Jacobian solve.

V40's action-margin, per-context and motion guards are retained verbatim.
No new sensory information, decoder, neural dynamics or physical assistance.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_box_solver import bounded_direction
from .layout_margin_projected_train import margin_gate
from .layout_contextual_operation_probe import measure, measure_bank, CAPS, MOTION_BUDGET, BACKTRACK
from .layout_operation_prefix import KINDS, CONTEXTS
from .layout_operation_sampling import OperationCache
from .layout_operation_focus import focus_values
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model, save_model
from .layout_guarded_snapshot_train import verify_saved_snapshot
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

ALGORITHM = 'single-box-constrained-context-step-v1'


def attempt(model, bank, on_initial=lambda value: None, solver_iterations=1024):
    before, fixed, mode = model.checkpoint_hash(), model.fingerprint(), model.training
    params = (model.log_gains, model.tonic)
    if (before != bank.manifest['parameterHash'] or before != bank.manifest['behaviorParameterHash']
            or model.surrogate_training or any(p.grad is not None for p in params)):
        raise ValueError('Exact current on-policy parent and derivative required')
    base = [p.detach().clone() for p in params]
    ids = [bank.groups[k, c][0] for k in KINDS for c in CONTEXTS]
    selections = [bank.manifest['windows'][i] for i in ids]
    all_ids = list(range(len(bank.manifest['windows'])))
    committed = False
    try:
        model.eval(); full_before = measure_bank(model, bank, all_ids); on_initial(full_before)
        x, y, state, motion = bank.batch(ids, model.tonic.device)
        model.train()
        p = recurrent_predictions(model, x, state, 64, model.weights(), gradient_start=0)
        initial = measure(p, y[64:], x[64:], motion, selections)
        grip, _ = focus_values(p, y[64:], selections)
        outputs = list(grip.unbind()) + [p[..., h].mean() for h in range(2)]
        residual = [max(0., 1.1-float(grip[i].detach())) if s['category'] == CONTEXTS[0]
                    else min(0., .9-float(grip[i].detach())) if s['category'] == CONTEXTS[2] else 0.
                    for i, s in enumerate(selections)] + [0., 0.]
        gradients = []
        for i, output in enumerate(outputs):
            gradients.append([g.detach() for g in torch.autograd.grad(output, params, retain_graph=i<len(outputs)-1)])
            print(json.dumps({'boxStepJacobianRows': i+1, 'total': len(outputs), 'maximumUpdates': 1}), flush=True)
        del p, grip, outputs
        delta, linear = bounded_direction(gradients, residual, base, iterations=solver_iterations,
            on_progress=lambda value: print(json.dumps(value), flush=True))
        del gradients
        model.eval(); trials = []
        with torch.no_grad():
            for scale in BACKTRACK:
                for param, original, d in zip(params, base, delta):
                    param.copy_(original+scale*d)
                # The solver includes these global bounds; clamp only float32 roundoff.
                params[0].clamp_(-2., 2.); params[1].clamp_(-.1, .1)
                measured = measure(frozen_predictions(model, x, state, 64), y[64:], x[64:], motion, selections)
                selected_gate = margin_gate(initial, measured)
                full_after = measure_bank(model, bank, all_ids) if selected_gate['passed'] else None
                full_gate = margin_gate(full_before, full_after, 0.) if full_after is not None else None
                changes = [float((param-original).abs().max()) for param, original in zip(params, base)]
                passed = selected_gate['passed'] and full_gate['passed']
                if any(change > cap+1e-7 for change,cap in zip(changes,CAPS)):
                    raise ValueError('Coordinate trust bound exceeded')
                row = {'backtrack': scale, 'selectedGate': selected_gate, 'fullGate': full_gate,
                       'measured': measured, 'fullBank': full_after, 'accepted': bool(passed),
                       'maximumParameterChanges': changes, 'parameterHash': model.checkpoint_hash()}
                trials.append(row)
                print('BOX_CONTEXT_TRIAL '+json.dumps({k:row[k] for k in
                      ('backtrack', 'selectedGate', 'fullGate', 'accepted', 'maximumParameterChanges')}), flush=True)
                if passed:
                    if row['parameterHash'] == before or model.fingerprint() != fixed:
                        raise ValueError('Finite changed brain on the original fixed graph required')
                    committed = True
                    break
        return {'algorithm': ALGORITHM, 'beforeHash': before, 'afterHash': model.checkpoint_hash() if committed else before,
                'accepted': committed, 'windowIndexes': ids, 'selections': selections,
                'initial': initial, 'initialFullBank': full_before, 'boundedLinearSolve': linear, 'trials': trials,
                'fixedHash': fixed, 'isServiceEvidence': False, 'teacherAtInference': False,
                'v40TrainingGuardsUnchanged': True, 'physicalReleaseCriteriaChanged': False,
                'candidateFullHistoryPrefix': False, 'sourceLabelsAndSensoryHistoriesUnchanged': True}
    except BaseException:
        committed = False
        raise
    finally:
        if not committed:
            with torch.no_grad():
                for param, original in zip(params, base):
                    param.copy_(original)
        model.train(mode)
        if model.fingerprint() != fixed or any(p.grad is not None for p in params) or (not committed and model.checkpoint_hash() != before):
            raise AssertionError('Fixed-state/gradient/rollback invariant failed')


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve every previous experiment')
    if (args.prior_phase/'failure.json').exists():
        raise ValueError('Failed prior execution cannot support this experiment')
    prior = json.loads((args.prior_phase/'result.json').read_text())
    prior_manifest = json.loads((args.prior_phase/'manifest.json').read_text())
    step = json.loads((args.prior_phase/'step-1.json').read_text())
    if (prior['algorithm'] != 'single-coordinate-projected-context-action-margin-step-v1'
            or prior['acceptedUpdates'] != 0 or not prior['parametersRestored'] or prior['candidateSaved']
            or not prior['sourceFilesUnchanged'] or prior['isServiceEvidence'] or step['accepted']
            or prior['finalParameterHash'] != step['beforeHash'] or step['beforeHash'] != step['afterHash']):
        raise ValueError('Original completed V40 rejection required')
    # Recompute every old gate; never reinterpret its rejection as acceptance.
    for trial in step['trials']:
        if margin_gate(step['initial'], trial['measured']) != trial['selectedGate'] or trial['accepted']:
            raise ValueError('Original V40 result is inconsistent')
    hashes = {**prior_manifest['inputFileHashes'], **{str((args.prior_phase/name).resolve()): file_hash(args.prior_phase/name)
              for name in ('manifest.json', 'result.json', 'status.json', 'step-1.json')}}
    names = {Path(__file__).name, 'layout_box_solver.py'}
    sources = {**prior_manifest['sourceHashes'], **{n:file_hash(Path(__file__).with_name(n)) for n in names}}
    def unchanged():
        if any(file_hash(Path(p)) != h for p,h in hashes.items()) or any(file_hash(Path(__file__).with_name(n)) != h for n,h in sources.items()):
            raise ValueError('Original inputs or sources changed')
    unchanged()
    for p in (args.candidate, args.cache/'manifest.json', args.cache/'cache.npz', args.dataset/'manifest.json'):
        if hashes.get(str(p.resolve())) != file_hash(p):
            raise ValueError('Wrong original input path')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).train()
    if model.checkpoint_hash() != prior['finalParameterHash'] or model.fixed_hash != prior_manifest['fixedHash']:
        raise ValueError('Wrong current parent')
    bank = OperationCache(args.cache, args.dataset, model.checkpoint_hash(), model.fixed_hash, model.n)
    gains, tonic = [p.detach().cpu().numpy().copy() for p in (model.log_gains, model.tonic)]
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json', {'algorithm': ALGORITHM, 'maximumUpdates': 1,
        'parentParameterHash': model.checkpoint_hash(), 'fixedHash': model.fixed_hash,
        'inputFileHashes': hashes, 'sourceHashes': sources, 'exactReLUDerivative': True,
        'gradientFrames': 96, 'scoredFrames': 32, 'prefixGradientDetached': True,
        'coordinateCaps': list(CAPS), 'motionBudget': list(MOTION_BUDGET), 'positiveGripMargin': 1.1,
        'v40TrainingGuardsUnchanged': True, 'minimumSelectedImprovement': .01,
        'originalPhysicsSensesDecoderAndPhysicalReleaseGateUnchanged': True,
        'solver': 'row-normalized ridge least-squares with coordinate/global bounds inside accelerated projected-gradient solve',
        'solverIterationLimit': 1024, 'solverStationarityTolerance': 1e-5,
        'statelessDirection': True, 'oldAdamConsumed': False, 'canonicalLearner': 'Spark1 only'})
    atomic_json(args.out/'status.json', {'finished': False, 'acceptedUpdates': 0})
    began = time.perf_counter()
    try:
        row = attempt(model, bank, lambda value: atomic_json(args.out/'initial-all-window-measure.json', value))
        unchanged(); atomic_json(args.out/'step-1.json', row)
        result = {'algorithm': ALGORITHM, 'acceptedUpdates': int(row['accepted']),
                  'finalParameterHash': model.checkpoint_hash(), 'seconds': time.perf_counter()-began,
                  'sourceFilesUnchanged': True, 'isServiceEvidence': False, 'automaticContinuation': False}
        if row['accepted']:
            audit = model.audit(gains)
            if (not all(audit[k] for k in ('finite', 'signsPreserved', 'fixedGraphSensoryDecoderDynamicsUnchanged'))
                    or model.checkpoint_hash() != row['afterHash']):
                raise ValueError('Same-pass saved brain audit failed')
            path = args.out/'candidate-1.npz'
            save_model(path, model); verify_saved_snapshot(path, model); unchanged()
            result.update(candidateFileHash=file_hash(path), savedArraysExactlyMatchTestedSnapshot=True,
                          audit=audit, changedNeurons=int(np.count_nonzero(tonic!=model.tonic.detach().cpu().numpy())),
                          requiresFrozenPhysicalVerification=True)
        else:
            result.update(candidateSaved=False, parametersRestored=True, earlyStop='No bounded solve proposal passed unchanged V40 training guards')
        atomic_json(args.out/'result.json', result)
        atomic_json(args.out/'status.json', {'finished': True, 'acceptedUpdates': int(row['accepted']), 'parameterHash': model.checkpoint_hash()})
        print('BOX_CONSTRAINED_PHASE_COMPLETE '+json.dumps(result), flush=True)
    except BaseException as error:
        with torch.no_grad():
            model.log_gains.copy_(torch.as_tensor(gains,device=model.log_gains.device))
            model.tonic.copy_(torch.as_tensor(tonic,device=model.tonic.device))
        atomic_json(args.out/'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'dataset', 'prior-phase', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
