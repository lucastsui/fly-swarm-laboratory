"""One bounded context update with coordinate projection and action margins.

Explicit new TRAINING guards protect correct decisions, not raw grip amplitude.
Compare the globally scaled direction with a coordinate-clipped direction;
retain only a same-pass, full-bank guarded snapshot. Physical release tests,
graph/signs, sensory projection, decoder and neural forward dynamics are fixed.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_context_step_probe import direction, trust_scale
from .layout_contextual_operation_probe import measure, measure_bank, contextual_gate, CAPS, MOTION_BUDGET, BACKTRACK
from .layout_operation_prefix import KINDS, CONTEXTS
from .layout_operation_sampling import OperationCache
from .layout_operation_focus import focus_values
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model, save_model
from .layout_guarded_snapshot_train import verify_saved_snapshot
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

ALGORITHM = 'single-coordinate-projected-context-action-margin-step-v1'


def margin_gate(before, after, minimum_improvement=.01):
    if not np.isfinite(minimum_improvement) or not 0 <= minimum_improvement < 1:
        raise ValueError('Bounded improvement fraction required')
    if before['positiveFocus'] != after['positiveFocus'] or before['protectedLoadedMask'] != after['protectedLoadedMask']:
        raise ValueError('Original sensory/label masks changed')
    for key in ('rawGrip', 'protectedLoadedGrip', 'parentMotionMSE'):
        a, b = np.asarray(before[key]), np.asarray(after[key])
        if not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Finite aligned measurements required')
    for key in ('balancedHinge', 'opposedReturnHinge'):
        if not np.isfinite([before[key], after[key]]).all() or min(before[key], after[key]) < 0:
            raise ValueError('Finite nonnegative losses required')
    expected = [(k, c) for k in KINDS for c in CONTEXTS]
    for value in (before, after):
        if [(g['kind'], g['context']) for g in value['groups']] != expected:
            raise ValueError('Every original family/context required in order')
    groups_ok = True
    for a, b in zip(before['groups'], after['groups']):
        if a['windows'] != b['windows'] or a['windows'] <= 0:
            raise ValueError('Selection coverage changed')
        if not np.isfinite([a['hinge'], b['hinge']]).all():
            raise ValueError('Nonfinite contextual loss')
        groups_ok &= b['hinge'] <= a['hinge']+1e-9 and b['correct'] >= a['correct']
    raw, new = np.asarray(before['rawGrip']), np.asarray(after['rawGrip'])
    desired = np.asarray(before['positiveFocus'], dtype=bool)
    if desired.shape != raw.shape:
        raise ValueError('Original aligned focus decisions required')
    loaded, changed = np.asarray(before['protectedLoadedGrip']), np.asarray(after['protectedLoadedGrip'])
    was_correct = (raw > 1.) == desired
    details = {
        'contextGroupsNonRegressing': bool(groups_ok),
        'balancedImprovement': 1-after['balancedHinge']/before['balancedHinge'] if before['balancedHinge'] > 0 else None,
        'returnImprovement': 1-after['opposedReturnHinge']/before['opposedReturnHinge'] if before['opposedReturnHinge'] > 0 else None,
        'correctFocusPreserved': bool(np.all(((new > 1.) == desired)[was_correct])),
        'correctLoadedDecisionsPreserved': bool(np.all(changed[loaded > 1.] > 1.)),
        # A supported output already above1.1 may change amplitude, but cannot
        # lose its safety margin. Lower supported outputs must not decrease.
        'positiveLoadedMarginPreserved': bool(np.all(changed >= np.minimum(loaded, 1.1)-1e-7)),
        'positiveLoadedMaxDrift': float(np.max(np.abs(loaded-changed))),
        'parentMotionWithinBudget': bool(np.all(np.asarray(after['parentMotionMSE']) <= np.asarray(MOTION_BUDGET))),
        'minimumImprovement': minimum_improvement,
    }
    details['passed'] = bool(before['balancedHinge'] > 0 and before['opposedReturnHinge'] > 0
        and after['balancedHinge'] <= (1-minimum_improvement)*before['balancedHinge']+1e-9
        and after['opposedReturnHinge'] <= (1-minimum_improvement)*before['opposedReturnHinge']+1e-9
        and all(details[k] for k in ('contextGroupsNonRegressing', 'correctFocusPreserved',
             'correctLoadedDecisionsPreserved', 'positiveLoadedMarginPreserved', 'parentMotionWithinBudget')))
    return details


def project_coordinates(delta, caps=CAPS):
    if len(delta) != 2 or len(caps) != 2 or any(not np.isfinite(c) or c <= 0 for c in caps):
        raise ValueError('Two finite positive coordinate bounds required')
    if any(not torch.isfinite(d).all() for d in delta):
        raise ValueError('Finite proposal required')
    return [d.clamp(-cap, cap) for d, cap in zip(delta, caps)]


def attempt(model, bank, on_initial=lambda value: None):
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
            print(json.dumps({'projectedStepJacobianRows': i+1, 'total': len(outputs), 'maximumUpdates': 1}), flush=True)
        delta, linear = direction(gradients, residual)
        projected = project_coordinates(delta)
        global_scale = trust_scale(delta, CAPS)
        projection = {'globalScaleControl': global_scale,
                      'clippedCoordinates': [int((d.abs()>cap).sum()) for d, cap in zip(delta, CAPS)],
                      'predictedProjectedOutputChange': [sum(float((g*d).sum()) for g,d in zip(row,projected)) for row in gradients]}
        del p, grip, outputs, gradients
        model.eval(); trials = []; control = None
        with torch.no_grad():
            # Matched global-scale control is measured but never committed.
            for param, original, d in zip(params, base, delta):
                param.copy_(original+global_scale*d)
            params[0].clamp_(-2., 2.); params[1].clamp_(-.1, .1)
            measured = measure(frozen_predictions(model, x, state, 64), y[64:], x[64:], motion, selections)
            control = {'measured': measured, 'newMarginGate': margin_gate(initial, measured),
                       'oldAmplitudeGate': contextual_gate(initial, measured), 'permanentUpdate': False}
            for scale in BACKTRACK:
                for param, original, d in zip(params, base, projected):
                    param.copy_(original+scale*d)
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
                print('PROJECTED_CONTEXT_TRIAL '+json.dumps({k:row[k] for k in
                      ('backtrack', 'selectedGate', 'fullGate', 'accepted', 'maximumParameterChanges')}), flush=True)
                if passed:
                    if row['parameterHash'] == before or model.fingerprint() != fixed:
                        raise ValueError('Finite changed brain on the original fixed graph required')
                    committed = True
                    break
        return {'algorithm': ALGORITHM, 'beforeHash': before, 'afterHash': model.checkpoint_hash() if committed else before,
                'accepted': committed, 'windowIndexes': ids, 'selections': selections,
                'initial': initial, 'initialFullBank': full_before, 'linearProposal': linear,
                'projection': projection, 'globalControl': control, 'trials': trials,
                'fixedHash': fixed, 'isServiceEvidence': False, 'teacherAtInference': False,
                'originalAmplitudeGuardSupersededExplicitly': True, 'physicalReleaseCriteriaChanged': False,
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
        raise FileExistsError('Preserve every previous candidate/experiment')
    if (args.prior_probe/'failure.json').exists():
        raise ValueError('Failed prior probe cannot support this experiment')
    prior = json.loads((args.prior_probe/'result.json').read_text())
    prior_manifest = json.loads((args.prior_probe/'manifest.json').read_text())
    if (prior.get('schema') != 'individual-pickup-return-context-probe-v1'
            or not prior['parametersRestored'] or prior['candidateSaved'] or prior['optimizerUsed']
            or prior['isServiceEvidence'] or not prior['sourceFilesUnchanged']
            or prior['inputFileHashes'] != prior_manifest['inputFileHashes']
            or prior['sourceHashes'] != prior_manifest['sourceHashes']
            or prior['gradientFrames'] != 96 or not prior['prefixExactAtInitialization']):
        raise ValueError('Original completed contextual probe required')
    hashes = {**prior['inputFileHashes'], **{str((args.prior_probe/name).resolve()): file_hash(args.prior_probe/name)
              for name in ('manifest.json', 'result.json')}}
    names = {Path(__file__).name, 'layout_guarded_snapshot_train.py'}
    sources = {**prior['sourceHashes'], **{n:file_hash(Path(__file__).with_name(n)) for n in names}}
    def unchanged():
        if any(file_hash(Path(p)) != h for p,h in hashes.items()) or any(file_hash(Path(__file__).with_name(n)) != h for n,h in sources.items()):
            raise ValueError('Original inputs or sources changed')
    unchanged()
    for p in (args.candidate, args.cache/'manifest.json', args.cache/'cache.npz', args.dataset/'manifest.json'):
        if hashes.get(str(p.resolve())) != file_hash(p):
            raise ValueError('Wrong original input path')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).train()
    if model.checkpoint_hash() != prior['parentParameterHash'] or model.fixed_hash != prior['fixedHash']:
        raise ValueError('Wrong current parent')
    bank = OperationCache(args.cache, args.dataset, model.checkpoint_hash(), model.fixed_hash, model.n)
    gains, tonic = [p.detach().cpu().numpy().copy() for p in (model.log_gains, model.tonic)]
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json', {'algorithm': ALGORITHM, 'maximumUpdates': 1,
        'parentParameterHash': model.checkpoint_hash(), 'fixedHash': model.fixed_hash,
        'inputFileHashes': hashes, 'sourceHashes': sources, 'exactReLUDerivative': True,
        'gradientFrames': 96, 'scoredFrames': 32, 'prefixGradientDetached': True,
        'coordinateCaps': list(CAPS), 'motionBudget': list(MOTION_BUDGET), 'positiveGripMargin': 1.1,
        'oldAmplitudeGuardSupersededExplicitly': True, 'minimumSelectedImprovement': .01,
        'originalPhysicsSensesDecoderAndPhysicalReleaseGateUnchanged': True,
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
            result.update(candidateSaved=False, parametersRestored=True, earlyStop='No proposal passed the new selected/full-bank action-margin guards')
        atomic_json(args.out/'result.json', result)
        atomic_json(args.out/'status.json', {'finished': True, 'acceptedUpdates': int(row['accepted']), 'parameterHash': model.checkpoint_hash()})
        print('MARGIN_PROJECTED_PHASE_COMPLETE '+json.dumps(result), flush=True)
    except BaseException as error:
        with torch.no_grad():
            model.log_gains.copy_(torch.as_tensor(gains,device=model.log_gains.device))
            model.tonic.copy_(torch.as_tensor(tonic,device=model.tonic.device))
        atomic_json(args.out/'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'dataset', 'prior-probe', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
