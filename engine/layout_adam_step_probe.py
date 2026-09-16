"""Restored exact-vs-surrogate step probe using saved Adam moments.

Only synthetic TRAINING observations and original parent-motion labels. No
optimizer.step, checkpoint saving, physical actions, or enduring mutation.
"""
import argparse
import contextlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .layout_recovery_brain import load_model
from .layout_recovery_protocol import atomic_json
from .layout_recovery_train import recurrent_predictions
from .layout_recovery_step_probe import temporary_step, forward_predictions, grip_terms, measurements
from .layout_microfit_resume import file_hash


def adam_direction(gradient, state, group):
    """Out-of-place next-step direction with the learner's per-block clipping."""
    if group.get('weight_decay', 0) or group.get('amsgrad', False) or group.get('maximize', False):
        raise ValueError('Unsupported Adam variant')
    if not torch.isfinite(gradient).all():
        raise ValueError('Nonfinite gradient')
    g = gradient*min(1., 1./(float(gradient.norm())+1e-6))
    beta1, beta2 = group['betas']
    step = int(float(state['step']))+1
    m = beta1*state['exp_avg']+(1-beta1)*g
    v = beta2*state['exp_avg_sq']+(1-beta2)*g.square()
    return -(group['lr']/(1-beta1**step))*m/(v.sqrt()/math.sqrt(1-beta2**step)+group['eps'])


@contextlib.contextmanager
def training_derivative(model, surrogate):
    saved_training, saved_surrogate = model.training, model.surrogate_training
    try:
        model.train()
        model.surrogate_training = surrogate
        yield
    finally:
        model.train(saved_training)
        model.surrogate_training = saved_surrogate


def objective(p, y):
    hinge, ranking = grip_terms(p, y)
    motion = (p[..., :2]-y[..., :2]).square().mean((0, 1))
    return 8*motion[0]+4*motion[1]+8*(hinge+1000*ranking)


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostic')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate, surrogate=False)
    before, fixed = model.checkpoint_hash(), model.fingerprint()
    parameters = [model.log_gains, model.tonic]
    originals = [p.detach().clone() for p in parameters]
    with np.load(args.bank, allow_pickle=False) as data:
        x = torch.as_tensor(data['observations'].copy(), device=model.tonic.device)
        y = torch.as_tensor(data['targets'].copy(), device=model.tonic.device)
        original_parent = str(data['parent_hash'])
    if x.shape != (96, 16, 297) or y.shape != (32, 16, 3) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Expected the unchanged four-scene training bank')
    # Task-owned local file, never accepted from the network's experience API.
    payload = torch.load(args.optimizer, map_location=model.tonic.device, weights_only=True)
    groups, states = payload['optimizer']['param_groups'], payload['optimizer']['state']
    if len(groups) != 2 or any(len(g['params']) != 1 for g in groups):
        raise ValueError('Unexpected canonical Adam groups')
    ordered = [states[g['params'][0]] for g in groups]
    for p, state in zip(parameters, ordered):
        if (float(state['step']) != payload['updates'] or state['exp_avg'].shape != p.shape
                or state['exp_avg_sq'].shape != p.shape
                or any(not torch.isfinite(state[k]).all() for k in ('exp_avg', 'exp_avg_sq'))):
            raise ValueError('Adam state does not match brain')
    current = forward_predictions(model, x, 64)
    initial = {'loss': float(objective(current, y)), **measurements(current, y, y)}
    print('READY '+json.dumps({'hash': before, 'update': payload['updates'], 'loss': initial['loss']}), flush=True)
    trials, exact_gradients, comparison = [], None, None
    for mode in ('exact', 'surrogate'):
        with training_derivative(model, mode == 'surrogate'):
            p = recurrent_predictions(model, x, torch.zeros((model.n, 16), device=x.device),
                                      64, model.weights(), gradient_start=0)
            torch.testing.assert_close(p.detach(), current, rtol=1e-5, atol=1e-5)
            loss = objective(p, y)
            gradients = [g.detach() for g in torch.autograd.grad(loss, parameters)]
            del p, loss
        if mode == 'exact':
            exact_gradients = gradients
        else:
            comparison = [float(torch.dot(a.flatten(), b.flatten())/(a.norm()*b.norm()).clamp_min(1e-30))
                          for a, b in zip(exact_gradients, gradients)]
        directions = [adam_direction(g, state, group) for g, state, group in zip(gradients, ordered, groups)]
        slope = sum(float((g*d).sum()) for g, d in zip(exact_gradients, directions))
        print('DIRECTION '+json.dumps({'mode': mode, 'exactLocalSlope': slope,
                                       'gradientNorms': [float(g.norm()) for g in gradients]}), flush=True)
        for scale in (.1, .3, 1.):
            with temporary_step(model, originals, [d*scale for d in directions]):
                p = forward_predictions(model, x, 64)
                row = {'derivative': mode, 'stepScale': scale, 'exactLocalSlope': slope,
                       'loss': float(objective(p, y)), **measurements(p, y, y)}
            assert model.checkpoint_hash() == before
            trials.append(row)
            print('TRIAL '+json.dumps(row), flush=True)
        del directions, gradients, p
    assert model.checkpoint_hash() == before and model.fingerprint() == fixed
    result = {'checkpointHash': before, 'fixedHash': fixed, 'savedAdamUpdate': payload['updates'],
              'originalCalibrationParentHash': original_parent, 'parametersRestoredExactly': True,
              'optimizerStepExecuted': False, 'savedAdamMomentsUsed': True, 'candidateSaved': False,
              'isServiceEvidence': False, 'syntheticTrainingOnly': True,
              'objective': '8*speedMSE + 4*turnMSE + 8*(balanced grip hinge + 1000*matched cargo ranking)',
              'sourceHash': file_hash(__file__), 'inputFileHashes': {p.name: file_hash(p) for p in
                (args.candidate, args.optimizer, args.bank)},
              'gradientCosinesGainTonic': comparison, 'initial': initial, 'trials': trials}
    atomic_json(args.out, result)
    print('COMPLETE '+json.dumps({k: v for k, v in result.items() if k not in ('initial', 'trials')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'optimizer', 'bank', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
