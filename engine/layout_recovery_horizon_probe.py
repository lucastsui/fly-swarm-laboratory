"""Restored, optimizer-free gradient-horizon diagnostic on TRAINING contexts.

Compare the SAME forward sequence/loss with or without differentiating its
burn-in. No candidate, teacher actions, new decoder or sensory mapping is made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .layout_grip_calibration import matched_examples
from .layout_microfit import select_microfit_scenes
from .layout_recovery_brain import load_model
from .layout_recovery_protocol import atomic_json
from .layout_recovery_step_probe import (first_step_direction, temporary_step,
                                         forward_predictions, grip_terms, measurements)
from .supervised_joint import raw_readout


def horizon_predictions(model, observations, output_start, gradient_start):
    if not 0 <= gradient_start <= output_start < len(observations):
        raise ValueError('Gradient start must precede retained outputs')
    weights, state, result = model.weights(), None, []
    for frame, x in enumerate(observations):
        with torch.set_grad_enabled(frame >= gradient_start):
            _, state = model(x, 4, state, weights)
            if frame >= output_start:
                result.append(raw_readout(model, state))
    return torch.stack(result)


def objective(p, y, parent, rank_weight):
    hinge, rank = grip_terms(p, y)
    motion = (p[..., :2]-parent[..., :2]).square().mean((0, 1))
    return 8*motion[0]+4*motion[1]+8*(hinge+rank_weight*rank)


def cosine(a, b):
    return float(torch.dot(a.flatten(), b.flatten())/(a.norm()*b.norm()).clamp_min(1e-30))


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostic')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate, surrogate=False)
    model.train()
    before, fixed = model.checkpoint_hash(), model.fingerprint()
    originals = [p.detach().clone() for p in (model.log_gains, model.tonic)]
    x, y, meta = matched_examples(scenes=64)
    selected = select_microfit_scenes(meta, y)
    indices = (4*np.asarray(selected)[:, None]+np.arange(4)).ravel()
    x = torch.as_tensor(x[:, indices], device=model.tonic.device)
    y = torch.as_tensor(y[:, indices], device=model.tonic.device)
    parent = forward_predictions(model, x, 64)
    initial_loss = float(objective(parent, y, parent, 1000.))
    print('FORWARD_READY '+json.dumps({'hash': before, 'loss': initial_loss}), flush=True)
    gradients, losses = {}, {}
    for start in (64, 0):
        p = horizon_predictions(model, x, 64, start)
        torch.testing.assert_close(p.detach(), parent, rtol=1e-5, atol=1e-5)
        loss = objective(p, y, parent, 1000.)
        gradients[start] = [g.detach() for g in torch.autograd.grad(loss, (model.log_gains, model.tonic))]
        losses[start] = float(loss.detach())
        if any(not torch.isfinite(g).all() for g in gradients[start]):
            raise FloatingPointError('Nonfinite diagnostic gradient')
        print('GRADIENT_READY '+json.dumps({'start': start, 'loss': losses[start],
              'norms': [float(g.norm()) for g in gradients[start]]}), flush=True)
        del p, loss
    trials = []
    for start in (64, 0):
        directions = [first_step_direction(g, rate, eps) for g, rate, eps in
                      zip(gradients[start], (.003, .00001), (1e-14, 1e-10))]
        slopes = {str(s): sum(float((g*d).sum()) for g, d in zip(gradients[s], directions))
                  for s in (64, 0)}
        for scale in (.01, .1, 1.):
            observations = {}
            for sign in (-1, 1):
                with temporary_step(model, originals, [d*(sign*scale) for d in directions]):
                    p = forward_predictions(model, x, 64)
                    observations[str(sign)] = {'loss': float(objective(p, y, parent, 1000.)),
                                               **measurements(p, y, parent)}
                assert model.checkpoint_hash() == before
            trials.append({'gradientStart': start, 'scale': scale, 'predictedSlopes': slopes,
                           'centralSlope': (observations['1']['loss']-observations['-1']['loss'])/(2*scale),
                           'trials': observations})
            print('PERTURBATION_READY '+json.dumps({k: v for k, v in trials[-1].items() if k != 'trials'}), flush=True)
    assert model.checkpoint_hash() == before and model.fingerprint() == fixed
    result = {'checkpointHash': before, 'fixedHash': fixed, 'parametersRestoredExactly': True,
              'optimizerUsed': False, 'candidateSaved': False, 'isServiceEvidence': False,
              'syntheticTrainingContexts': True, 'scenes': [meta[i] for i in selected],
              'sourceHash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fullForwardFrames': 96, 'retainedLossFrames': 32, 'substepsPerFrame': 4,
              'gradientCosinesGainTonic': [cosine(a, b) for a, b in zip(gradients[64], gradients[0])],
              'gradientNorms': {str(s): [float(g.norm()) for g in gradients[s]] for s in (64, 0)},
              'initialLoss': initial_loss, 'initial': measurements(parent, y, parent), 'trials': trials}
    atomic_json(args.out, result)
    print('COMPLETE '+json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
