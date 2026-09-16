"""Bounded, restored finite-step diagnostic. No optimizer or saved candidate.

Synthetic TRAINING scenes only. Compare actual post-perturbation forward losses
to local gradients; this is not service, generalization or a new learned model.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .layout_grip_calibration import matched_examples
from .layout_grip_objective import threshold_grip_loss, matched_grip_ranking
from .layout_recovery_brain import load_model
from .layout_recovery_protocol import atomic_json
from .layout_recovery_train import recurrent_predictions
from .supervised_joint import raw_readout


def first_step_direction(gradient, rate, eps):
    """Fresh Adam direction, with the v5 independent norm clip and no momentum."""
    if not rate > 0 or not eps > 0 or not torch.isfinite(gradient).all():
        raise ValueError('Finite gradients and positive controls required')
    scale = min(1., 1./(float(gradient.norm())+1e-6))
    g = gradient*scale
    return -rate*g/(g.abs()+eps)


@contextlib.contextmanager
def temporary_step(model, originals, directions):
    """Restore exactly even if the diagnostic raises; never touch an optimizer."""
    try:
        with torch.no_grad():
            model.log_gains.copy_((originals[0]+directions[0]).clamp(-2., 2.))
            model.tonic.copy_((originals[1]+directions[1]).clamp(-.1, .1))
        yield
    finally:
        with torch.no_grad():
            model.log_gains.copy_(originals[0])
            model.tonic.copy_(originals[1])


@torch.no_grad()
def forward_predictions(model, observations, burn):
    weights = model.weights()
    state = None
    result = []
    for frame, x in enumerate(observations):
        _, state = model(x, 4, state, weights)
        if frame >= burn:
            result.append(raw_readout(model, state))
    return torch.stack(result)


def grip_terms(p, y):
    return threshold_grip_loss(p[..., 2], y[..., 2], .1), matched_grip_ranking(
        p[..., 2].reshape(-1, 4), y[..., 2].reshape(-1, 4), .1)


def measurements(p, y, parent):
    positive, response = y[..., 2] > 1., p[..., 2] > 1.
    hinge, ranking = grip_terms(p, y)
    return {'hinge': float(hinge), 'ranking': float(ranking),
            'recall': float(response[positive].float().mean()),
            'falsePositive': float(response[~positive].float().mean()),
            'parentMotionMSE': (p[..., :2]-parent[..., :2]).square().mean((0, 1)).tolist(),
            'lastFrameGripBySceneCargo': p[-1, :, 2].reshape(-1, 4).tolist()}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostic')
    if not all(np.isfinite(v) and v > 0 for v in args.ranking_weights+args.gain_rates):
        raise ValueError('Positive finite diagnostic controls required')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate, surrogate=False)
    model.train()
    before, fixed = model.checkpoint_hash(), model.fingerprint()
    originals = [p.detach().clone() for p in (model.log_gains, model.tonic)]
    x, y, meta = matched_examples(scenes=16)
    selected = [i for i in range(16) if (y[-1, 4*i:4*i+4, 2] > 1).any()
                and (y[-1, 4*i:4*i+4, 2] <= 1).any()][:4]
    indices = (4*np.asarray(selected)[:, None]+np.arange(4)).ravel()
    x = torch.as_tensor(x[:, indices], device='cuda')
    y = torch.as_tensor(y[:, indices], device='cuda')
    parent = forward_predictions(model, x, 64)
    trials = []
    for ranking_weight in args.ranking_weights:
        weights = model.weights()
        p = recurrent_predictions(model, x, torch.zeros((model.n, len(indices)), device='cuda'), 64, weights)
        hinge, ranking = grip_terms(p, y)
        loss = hinge+ranking_weight*ranking
        gradients = [g.detach() for g in torch.autograd.grad(loss, (model.log_gains, model.tonic))]
        del p, loss, hinge, ranking, weights
        for gain_lr in args.gain_rates:
            directions = [first_step_direction(g, rate, eps) for g, rate, eps in
                          zip(gradients, (gain_lr, .00001), (1e-14, 1e-10))]
            with temporary_step(model, originals, directions):
                p = forward_predictions(model, x, 64)
                trials.append({'rankingWeight': ranking_weight, 'gainLR': gain_lr, 'tonicLR': .00001,
                               'gradientNorms': [float(g.norm()) for g in gradients],
                               **measurements(p, y, parent)})
            assert model.checkpoint_hash() == before
            del directions, p
        del gradients
    assert model.checkpoint_hash() == before and model.fingerprint() == fixed
    result = {'checkpointHash': before, 'fixedHash': fixed, 'parametersRestoredExactly': True,
              'optimizerUsed': False, 'candidateSaved': False, 'isServiceEvidence': False,
              'syntheticTrainingContexts': True, 'freshStepNotSavedMomentum': True,
              'sourceHash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scenes': [meta[i] for i in selected], 'initial': measurements(parent, y, parent),
              'trials': trials}
    atomic_json(args.out, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--ranking-weights', nargs='+', type=float, default=[1., 100.])
    parser.add_argument('--gain-rates', nargs='+', type=float, default=[.001, .01, .03])
    main(parser.parse_args())
