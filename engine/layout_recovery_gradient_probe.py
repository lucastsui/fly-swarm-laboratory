"""Read-only comparison of exact and surrogate grip-training gradients.

No optimizer, parameter update, action override or checkpoint writing. Reports
are diagnostics on synthetic training contexts, NOT assembly-line performance.
"""
import argparse
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


def vector_comparison(exact, surrogate, clip_scale=1., eps=1e-10):
    a, b = exact.detach().reshape(-1), surrogate.detach().reshape(-1)
    denominator = a.norm()*b.norm()
    nz = b != 0
    return {'exactL2': float(a.norm()), 'surrogateL2': float(b.norm()),
            'cosine': float(torch.dot(a, b)/denominator) if denominator > 0 else None,
            'exactNonzero': int(torch.count_nonzero(a)), 'surrogateNonzero': int(nz.sum()),
            'surrogateRawBelowAdamEps': int(((b.abs() < eps) & nz).sum()),
            'surrogateClippedBelowAdamEps': int(((b.abs()*clip_scale < eps) & nz).sum()),
            'surrogateClippedBelowSmallerEps': int(((b.abs()*clip_scale < 1e-14) & nz).sum()),
            'surrogateClippedMax': float(b.abs().max()*clip_scale)}


def probe(model, observations, labels, burn=64, margin=.1, objective='combined'):
    if objective not in ('combined', 'ranking'):
        raise ValueError('Unknown diagnostic objective')
    before = model.checkpoint_hash()
    fixed_before = model.fingerprint()
    saved_mode = model.surrogate_training
    gradients, losses, predictions = {}, {}, {}
    try:
        for mode in ('exact', 'surrogate'):
            model.surrogate_training = mode == 'surrogate'
            weights = model.weights()
            state = torch.zeros((model.n, observations.shape[1]), device=observations.device)
            prediction = recurrent_predictions(model, observations, state, burn, weights)
            loss = matched_grip_ranking(prediction[..., 2].reshape(-1, 4),
                                       labels[..., 2].reshape(-1, 4), margin)
            if objective == 'combined':
                loss = loss+threshold_grip_loss(prediction[..., 2], labels[..., 2], margin)
            gradients[mode] = [g.detach() for g in torch.autograd.grad(loss, (model.log_gains, model.tonic))]
            losses[mode] = float(loss.detach())
            predictions[mode] = prediction.detach().cpu().numpy()
            del prediction, loss, weights, state
    finally:
        model.surrogate_training = saved_mode
    norm = torch.sqrt(sum(g.square().sum() for g in gradients['surrogate']))
    scale = min(1., 1./(float(norm)+1e-6))
    comparisons = {name: vector_comparison(a, b, scale) for name, a, b in
                   zip(('synapticGains', 'tonic'), gradients['exact'], gradients['surrogate'])}
    # This is the local derivative for a hypothetical FRESH Adam first step,
    # not a claim about a saved optimizer's momentum or an executed update.
    slope = 0.
    for rate, exact, surrogate in zip((.001, .00001), gradients['exact'], gradients['surrogate']):
        clipped = surrogate*scale
        slope -= float((exact*(rate*clipped/(clipped.abs()+1e-10))).sum())
    if model.checkpoint_hash() != before or model.fingerprint() != fixed_before:
        raise AssertionError('Gradient diagnostic changed the brain')
    return {'checkpointHash': before, 'fixedHash': fixed_before, 'learning': False,
            'optimizerUsed': False, 'syntheticTrainingContexts': True, 'isServiceEvidence': False,
            'objective': objective,
            'gripLoss': losses, 'forwardMaxDifference': float(np.max(np.abs(predictions['exact']-predictions['surrogate']))),
            'surrogateTotalGradientL2': float(norm), 'clipScale': scale,
            'gradientComparisons': comparisons, 'hypotheticalFreshAdamLocalSlope': slope,
            'prediction': predictions['exact'].tolist(), 'target': labels.detach().cpu().tolist()}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve gradient evidence')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate)
    model.train()
    x, y, meta = matched_examples(scenes=16)
    selected = [i for i in range(16) if (y[-1, i*4:i*4+4, 2] > 1).any()
                and (y[-1, i*4:i*4+4, 2] <= 1).any()][:4]
    indices = (4*np.asarray(selected)[:, None]+np.arange(4)).ravel()
    report = probe(model, torch.as_tensor(x[:, indices], device='cuda'),
                   torch.as_tensor(y[:, indices], device='cuda'), objective=args.objective)
    report.update(scenes=[meta[i] for i in selected], device=torch.cuda.get_device_name(),
                  sourceHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('prediction', 'target')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--objective', choices=('combined', 'ranking'), default='combined')
    main(parser.parse_args())
