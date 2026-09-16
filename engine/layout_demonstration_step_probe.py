"""Restored update-direction diagnostic on real physical TRAINING windows.

No optimizer.step, learned candidate, physical actions or inference assistance.
Full-window means 96 frames, NOT differentiation through the earlier cached
episode prefix. All comparisons use the same forward inputs, states and loss.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .layout_adam_step_probe import adam_direction, training_derivative
from .layout_recovery_step_probe import temporary_step
from .layout_recovery_horizon_probe import cosine
from .layout_recovery_brain import load_model
from .layout_recovery_train import recurrent_predictions, motor_head_errors
from .layout_demonstration_train import DemonstrationCache, training_scores
from .layout_demonstration_resume import prepare_continuation, restore_continuation_optimizer
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json
from .layout_recovery_teacher import KINDS
from .supervised_joint import raw_readout


def select_windows(bank, category='pickup-1'):
    indexes = []
    for kind in KINDS:
        candidates = []
        for index in sorted(bank.groups[kind].get(category, [])):
            on = bank.y[index, bank.manifest['burn']:, :, 2] > 1.
            if on.any() and (~on).any():
                candidates.append(index)
        if not candidates:
            raise ValueError('Missing mixed grip-on/off physical training window in '+kind)
        indexes.append(min(candidates))
    return indexes


def moment_hash(optimizer):
    h = hashlib.sha256()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                h.update(optimizer.state[parameter][key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def frozen_predictions(model, x, state, burn):
    weights, predictions = model.weights(), []
    state = state.detach()
    for frame in range(len(x)):
        _, state = model(x[frame], 4, state, weights)
        if frame >= burn:
            predictions.append(raw_readout(model, state))
    return torch.stack(predictions)


def objective(prediction, labels):
    heads = motor_head_errors(prediction, labels, balanced_grip=True)
    return (heads*heads.new_tensor([4., 2., 2.])).sum()


def run_probe(model, x, y, state, optimizer, burn, scales=(.1, 1., 4.)):
    if not scales or any(not 0 < s <= 10 for s in scales):
        raise ValueError('Bounded positive diagnostic step scales required')
    before = model.checkpoint_hash()
    moments_before = moment_hash(optimizer)
    parameters = [model.log_gains, model.tonic]
    originals = [p.detach().clone() for p in parameters]
    current = frozen_predictions(model, x, state, burn)
    labels = y[burn:]
    initial = {'loss': float(objective(current, labels)), **training_scores(current, labels)}
    gradients, trials, reference = [], [], None
    for mode, gradient_start, surrogate in (
            ('exact-full-window', 0, False), ('exact-truncated', burn, False),
            ('surrogate-full-window', 0, True)):
        with training_derivative(model, surrogate):
            prediction = recurrent_predictions(model, x, state, burn, model.weights(), gradient_start)
            torch.testing.assert_close(prediction.detach(), current, rtol=1e-5, atol=1e-5)
            loss = objective(prediction, labels)
            grad = [g.detach() for g in torch.autograd.grad(loss, parameters)]
            del prediction, loss
        if any(not torch.isfinite(g).all() for g in grad):
            raise FloatingPointError('Nonfinite probe gradient')
        if reference is None:
            reference = grad
        gradient_row = {'mode': mode, 'gradientStart': gradient_start,
                        'norms': [float(g.norm()) for g in grad],
                        'cosinesToExactFullWindow': [cosine(g, r) for g, r in zip(grad, reference)]}
        gradients.append(gradient_row)
        print('GRADIENT '+json.dumps(gradient_row), flush=True)
        for moment_mode in ('preserved', 'fresh'):
            directions = []
            for p, g, group in zip(parameters, grad, optimizer.param_groups):
                adam_state = optimizer.state[p] if moment_mode == 'preserved' else {
                    'step': torch.tensor(0.), 'exp_avg': torch.zeros_like(p), 'exp_avg_sq': torch.zeros_like(p)}
                directions.append(adam_direction(g, adam_state, group))
            slope = sum(float((g*d).sum()) for g, d in zip(reference, directions))
            for scale in scales:
                with temporary_step(model, originals, [d*scale for d in directions]):
                    prediction = frozen_predictions(model, x, state, burn)
                    row = {'mode': mode, 'moments': moment_mode, 'scale': scale,
                           'exactFullWindowLocalSlope': slope,
                           'loss': float(objective(prediction, labels)),
                           **training_scores(prediction, labels)}
                assert model.checkpoint_hash() == before
                trials.append(row)
                print('TRIAL '+json.dumps(row), flush=True)
            del directions
        del grad
    assert model.checkpoint_hash() == before and moment_hash(optimizer) == moments_before
    return {'initial': initial, 'gradients': gradients, 'trials': trials,
            'parametersRestoredExactly': True, 'optimizerMomentsUnchanged': True,
            'momentHash': moments_before}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve previous diagnostics')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate, surrogate=False)
    model.eval()
    before, fixed = model.checkpoint_hash(), model.fixed_hash
    bank = DemonstrationCache(args.cache, before, fixed, args.burn, args.gradient_frames, model.n)
    optimizer = torch.optim.Adam([{'params': [model.log_gains], 'lr': args.lr, 'eps': 1e-14},
                                  {'params': [model.tonic], 'lr': args.tonic_lr, 'eps': 1e-10}])
    # Reuse the actual source/checkpoint/Adam/new-data audit, but execute no step.
    continuation = prepare_continuation(args, model, bank)
    restore_continuation_optimizer(optimizer, continuation)
    indexes = select_windows(bank, args.category)
    x, y, state = bank.batch(indexes, model.tonic.device)
    print('READY '+json.dumps({'checkpoint': before, 'windows': indexes,
                              'category': args.category, 'savedUpdate': continuation['start']}), flush=True)
    result = run_probe(model, x, y, state, optimizer, args.burn)
    assert model.checkpoint_hash() == before and model.fingerprint() == fixed
    for filename, expected in continuation['record']['sourceFileSHA256'].items():
        source = Path(continuation['record']['sourceRun'])
        path = source.parent/filename if filename.endswith('.log') else source/filename
        assert file_hash(path) == expected
    result.update(checkpointHash=before, fixedHash=fixed, continuation=continuation['record'],
                  windows=[bank.manifest['windows'][i] for i in indexes],
                  cacheManifestHash=file_hash(args.cache/'manifest.json'),
                  sourceHash=file_hash(Path(__file__)), optimizerStepExecuted=False,
                  candidateSaved=False, isServiceEvidence=False, physicalTrainingWindowsOnly=True,
                  objective='4*speedMSE + 2*turnMSE + 2*class-balanced-grip-MSE-and-contrast',
                  fullWindowGradientFrames=args.burn+args.gradient_frames,
                  lossFrames=args.gradient_frames, earlierPrefixDifferentiated=False,
                  sourceHashes={name: file_hash(Path(__file__).parent/name) for name in
                    ('layout_recovery_train.py', 'layout_adam_step_probe.py', 'layout_recovery_step_probe.py',
                     'layout_recovery_brain.py', 'layout_excitability.py', 'supervised_steering.py')})
    atomic_json(args.out, result)
    print('PHYSICAL_WINDOW_PROBE_COMPLETE '+json.dumps({'initialLoss': result['initial']['loss'],
          'bestTrialLoss': min(r['loss'] for r in result['trials']), 'parametersRestoredExactly': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'out', 'resume-demonstration-run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume-new-demonstration-dataset', type=Path)
    parser.add_argument('--resume-checkpoint-update', type=int)
    parser.add_argument('--burn', type=int, default=64)
    parser.add_argument('--gradient-frames', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--tonic-lr', type=float, default=.000001)
    parser.add_argument('--seed', type=int, default=9380001)
    parser.add_argument('--recent-corrections', action='store_true')
    parser.add_argument('--require-stratified-corrections', action='store_true')
    parser.add_argument('--category', default='pickup-1')
    main(parser.parse_args())
