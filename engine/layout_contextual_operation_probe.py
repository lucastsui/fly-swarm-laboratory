"""Reversible context-specific pickup/return proposal on real brain histories.

This deliberately changes the TRAINING objective: teacher-opposed returns may
change, unlike the pickup-only experiment. Original labels, observations,
forward dynamics, motor decoder and physical release criteria are unchanged.
No candidate is saved and all temporary weights are restored, including errors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_context_step_probe import direction, trust_scale
from .layout_operation_sampling import OperationCache
from .layout_operation_prefix import KINDS, CONTEXTS
from .layout_operation_focus import focus_values
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

VERSION = 'individual-pickup-return-context-probe-v1'
CAPS = (.03, 1e-4)
MOTION_BUDGET = (2.5e-5, 2.5e-4)
POSITIVE_LOADED_BUDGET = .002
BACKTRACK = (1., .5, .25, .125, .0625, .03125)


def measure(predictions, labels, observations, motion, selections):
    grip, target = focus_values(predictions, labels, selections)
    if observations.shape != (32, len(selections), 297) or motion.shape != (32, len(selections), 2):
        raise ValueError('Original scored observations and motion required')
    if any(not torch.isfinite(a).all() for a in (predictions, labels, observations, motion)):
        raise ValueError('Finite original histories required')
    cargo = observations[..., 126:129]
    if not torch.all((cargo == 0) | (cargo == 1)) or not torch.all(cargo.sum(-1) <= 1):
        raise ValueError('Original one-hot cargo required')
    loaded = cargo.sum(-1) > 0
    pickup = torch.as_tensor([s['category'] == CONTEXTS[0] for s in selections], device=grip.device)
    if not torch.equal(loaded[16], ~pickup):
        raise ValueError('Selected physical operation and cargo disagree')
    positive = target > 1.
    loss = torch.where(positive, torch.relu(1.1 - grip), torch.relu(grip - .9)).square()
    groups = []
    for kind in KINDS:
        for context in CONTEXTS:
            ids = [i for i, s in enumerate(selections) if s['kind'] == kind and s['category'] == context]
            if not ids:
                raise ValueError('Every actual family/context required')
            groups.append({'kind': kind, 'context': context, 'windows': len(ids),
                           'hinge': float(loss[ids].mean().detach()),
                           'meanGrip': float(grip[ids].mean().detach()),
                           'correct': int(((grip[ids] > 1.) == positive[ids]).sum())})
    protected = loaded & (labels[..., 2] > 1.)
    if not protected.any():
        raise ValueError('Real supported loaded operations required')
    return {'groups': groups, 'balancedHinge': float(np.mean([r['hinge'] for r in groups])),
            'opposedReturnHinge': float(np.mean([r['hinge'] for r in groups if r['context'] == CONTEXTS[2]])),
            'rawGrip': grip.detach().cpu().tolist(), 'positiveFocus': positive.cpu().tolist(),
            'protectedLoadedGrip': predictions[..., 2][protected].detach().cpu().tolist(),
            'protectedLoadedMask': protected.cpu().tolist(),
            'parentMotionMSE': (predictions[..., :2] - motion).square().mean((0, 1)).detach().cpu().tolist()}


def contextual_gate(before, after, minimum_improvement=.01):
    if not np.isfinite(minimum_improvement) or not 0 <= minimum_improvement < 1:
        raise ValueError('Bounded improvement fraction required')
    if (before['positiveFocus'] != after['positiveFocus']
            or before['protectedLoadedMask'] != after['protectedLoadedMask']):
        raise ValueError('Original labels/cargo masks changed')
    for key in ('rawGrip', 'protectedLoadedGrip', 'parentMotionMSE'):
        old, new = np.asarray(before[key]), np.asarray(after[key])
        if old.shape != new.shape or not old.size or not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError('Finite aligned measured outputs required')
    for key in ('balancedHinge', 'opposedReturnHinge'):
        if not np.isfinite([before[key], after[key]]).all() or min(before[key], after[key]) < 0:
            raise ValueError('Finite nonnegative loss required')
    if len(before['groups']) != 16 or len(after['groups']) != 16:
        raise ValueError('Every family/context required')
    for old, new in zip(before['groups'], after['groups']):
        if any(old[k] != new[k] for k in ('kind', 'context', 'windows')):
            raise ValueError('Changed selection coverage')
        if not np.isfinite([old['hinge'], new['hinge'], old['meanGrip'], new['meanGrip']]).all():
            raise ValueError('Finite contextual measurements required')
        if new['hinge'] > old['hinge'] + 1e-9 or new['correct'] < old['correct']:
            return False
    raw, changed = np.asarray(before['rawGrip']), np.asarray(after['rawGrip'])
    desired = np.asarray(before['positiveFocus'], dtype=bool)
    if raw.shape != desired.shape:
        raise ValueError('Aligned desired focus decisions required')
    was_correct = (raw > 1.) == desired
    loaded, new_loaded = np.asarray(before['protectedLoadedGrip']), np.asarray(after['protectedLoadedGrip'])
    # Preserve each previously correct focus, not merely the net success count.
    # Only teacher-positive loaded frames have the old tight drift protection;
    # negative return focuses are now explicit correction targets.
    return bool(before['balancedHinge'] > 0 and before['opposedReturnHinge'] > 0
        and after['balancedHinge'] <= (1-minimum_improvement)*before['balancedHinge'] + 1e-9
        and after['opposedReturnHinge'] <= (1-minimum_improvement)*before['opposedReturnHinge'] + 1e-9
        and np.all(((changed > 1.) == desired)[was_correct])
        and np.array_equal(loaded > 1., new_loaded > 1.)
        and np.max(np.abs(loaded-new_loaded)) <= POSITIVE_LOADED_BUDGET
        and np.all(np.asarray(after['parentMotionMSE']) <= np.asarray(MOTION_BUDGET)))


@torch.no_grad()
def measure_bank(model, bank, ids):
    pieces = []
    for first in range(0, len(ids), 16):
        x, y, state, motion = bank.batch(ids[first:first+16], model.tonic.device)
        pieces.append((frozen_predictions(model, x, state, 64), y[64:], x[64:], motion))
    p, y, x, m = (torch.cat([part[i] for part in pieces], dim=1) for i in range(4))
    return measure(p, y, x, m, [bank.manifest['windows'][i] for i in ids])


def probe(model, bank, on_initial=lambda value: None):
    before, fixed, mode = model.checkpoint_hash(), model.fingerprint(), model.training
    params = (model.log_gains, model.tonic)
    if before != bank.manifest['parameterHash'] or model.surrogate_training or any(p.grad is not None for p in params):
        raise ValueError('Exact derivative/current full-history parent required')
    base = [p.detach().clone() for p in params]
    ids = [bank.groups[k, c][0] for k in KINDS for c in CONTEXTS]
    selections = [bank.manifest['windows'][i] for i in ids]
    all_ids = list(range(len(bank.manifest['windows'])))
    x, y, state, motion = bank.batch(ids, model.tonic.device)
    try:
        model.eval()
        full_before = measure_bank(model, bank, all_ids)
        on_initial(full_before)
        model.train()
        p = recurrent_predictions(model, x, state, 64, model.weights(), gradient_start=0)
        initial = measure(p, y[64:], x[64:], motion, selections)
        if initial['opposedReturnHinge'] <= 0:
            raise ValueError('No actual opposed-return deficit in selected histories')
        grip, _ = focus_values(p, y[64:], selections)
        outputs = list(grip.unbind()) + [p[..., h].mean() for h in range(2)]
        residual = [max(0., 1.1-float(grip[i].detach())) if s['category'] == CONTEXTS[0]
                    else min(0., .9-float(grip[i].detach())) if s['category'] == CONTEXTS[2] else 0.
                    for i, s in enumerate(selections)] + [0., 0.]
        gradients = []
        for i, output in enumerate(outputs):
            gradients.append([g.detach() for g in torch.autograd.grad(output, params, retain_graph=i<len(outputs)-1)])
            print(json.dumps({'contextualJacobianRows': i+1, 'total': len(outputs), 'permanentUpdates': False}), flush=True)
        delta, linear = direction(gradients, residual)
        del p, grip, outputs, gradients
        bounded = trust_scale(delta, CAPS)
        trials = []
        model.eval()
        with torch.no_grad():
            for backtrack in BACKTRACK:
                for param, original, d in zip(params, base, delta):
                    param.copy_(original + bounded*backtrack*d)
                params[0].clamp_(-2., 2.); params[1].clamp_(-.1, .1)
                measured = measure(frozen_predictions(model, x, state, 64), y[64:], x[64:], motion, selections)
                batch_pass = contextual_gate(initial, measured)
                full_after = measure_bank(model, bank, all_ids) if batch_pass else None
                full_pass = batch_pass and contextual_gate(full_before, full_after, minimum_improvement=0.)
                row = {'backtrack': backtrack, 'scale': bounded*backtrack, 'batchGuardPassed': batch_pass,
                       'allTrainingGuards': full_pass, 'measured': measured, 'fullBank': full_after,
                       'temporaryParameterHash': model.checkpoint_hash(),
                       'maximumParameterChanges': [float((param-original).abs().max()) for param, original in zip(params, base)]}
                trials.append(row)
                print('CONTEXTUAL_OPERATION_TRIAL '+json.dumps({k: row[k] for k in
                      ('backtrack', 'batchGuardPassed', 'allTrainingGuards', 'maximumParameterChanges')}), flush=True)
        return {'schema': VERSION, 'parentParameterHash': before, 'fixedHash': fixed,
                'windowIndexes': ids, 'selections': selections, 'initial': initial,
                'initialFullBank': full_before, 'linearProposal': linear, 'trials': trials,
                'parametersRestored': True, 'candidateSaved': False, 'optimizerUsed': False,
                'isServiceEvidence': False, 'prefixExactAtInitialization': True,
                'candidateFullHistoryPrefix': False, 'gradientFrames': 96, 'scoredFrames': 32,
                'prefixGradientDetached': True, 'originalLabelsUnchanged': True,
                'teacherAtInference': False, 'physicalActionsExecuted': False,
                'teacherOpposedReturnsAreCorrectionTargets': True,
                'teacherPreferencesAreNotProofOfOptimality': True,
                'positiveLoadedGripBudget': POSITIVE_LOADED_BUDGET, 'motionBudget': list(MOTION_BUDGET),
                'qualification': 'New contextual training objective; temporary counterfactual replay, NOT learned physical service'}
    finally:
        with torch.no_grad():
            for param, original in zip(params, base):
                param.copy_(original)
        model.train(mode)
        if model.checkpoint_hash() != before or model.fingerprint() != fixed or any(p.grad is not None for p in params):
            raise AssertionError('Original model was not restored')


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve every original experiment')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).train()
    if model.checkpoint_hash() != args.expected_parameter_hash:
        raise ValueError('Wrong original parent')
    bank = OperationCache(args.cache, args.dataset, args.expected_parameter_hash, model.fixed_hash, model.n)
    m = bank.manifest
    if m['candidateFileHash'] != file_hash(args.candidate) or m['behaviorParameterHash'] != model.checkpoint_hash():
        raise ValueError('Exact original on-policy candidate required')
    paths = [args.candidate, args.cache/'manifest.json', args.cache/'cache.npz', args.dataset/'manifest.json']
    paths += [args.dataset/e['file'] for e in m['datasetFiles']]
    hashes = {str(p.resolve()): file_hash(p) for p in paths}
    names = {Path(__file__).name, 'layout_operation_sampling.py', 'layout_operation_focus.py',
             'layout_context_step_probe.py', 'layout_context_credit_probe.py', 'layout_recovery_train.py',
             'layout_demonstration_step_probe.py', 'layout_grip_objective.py', 'layout_recovery_protocol.py'}
    sources = {**m['sourceHashes'], **{n: file_hash(Path(__file__).parent/n) for n in names}}
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json', {'schema': VERSION, 'parentParameterHash': model.checkpoint_hash(),
                'fixedHash': model.fixed_hash, 'inputFileHashes': hashes, 'sourceHashes': sources,
                'canonicalPrefixRun': m['prefixCanonicalRun'], 'canonicalPrefixVersion': m['prefixCanonicalVersion'],
                'behaviorParameterHash': m['behaviorParameterHash'], 'optimizerUsed': False,
                'purpose': 'Reversible contextual correction probe; NOT pickup-only preservation or physical service'})
    try:
        def initial(value):
            atomic_json(args.out/'initial-all-window-measure.json', value)
            print('INITIAL_CONTEXTUAL_FIT '+json.dumps({k: value[k] for k in
                  ('balancedHinge', 'opposedReturnHinge', 'parentMotionMSE')}), flush=True)
        result = probe(model, bank, initial)
        if any(file_hash(Path(p)) != h for p, h in hashes.items()) or any(file_hash(Path(__file__).parent/n) != h for n, h in sources.items()):
            raise AssertionError('Original inputs or source changed')
        result.update(inputFileHashes=hashes, sourceHashes=sources, sourceFilesUnchanged=True)
        atomic_json(args.out/'result.json', result)
        print('CONTEXTUAL_OPERATION_COMPLETE '+json.dumps({'passingBacktracks': [r['backtrack'] for r in result['trials'] if r['allTrainingGuards']],
              'parametersRestored': True, 'candidateSaved': False}), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'dataset', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-parameter-hash', required=True)
    main(parser.parse_args())
