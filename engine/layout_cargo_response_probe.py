"""Frozen TRAINING-input intervention, not physical behavior or new training.

Test whether fixed cargo sensory channels can affect existing motor outputs.
Interventions deliberately need not match physical cargo; their outputs are
NOT scored against teacher actions as if they were valid counterfactual labels.
No model, decoder, physical world, optimizer or saved candidate is changed.
"""
import argparse
import json
from pathlib import Path
import torch
from .layout_recovery_brain import load_model
from .layout_recovery_train import recurrent_predictions
from .layout_demonstration_train import DemonstrationCache, training_scores
from .layout_demonstration_focus import EventFocusedDemonstrations
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_teacher import KINDS
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def select_batch(focused, device):
    bank = focused.bank
    selections, xs, ys, states = [], [], [], []
    burn = bank.manifest['burn']
    for kind in KINDS:
        for stage in (1, 2, 3):
            found = None
            for index in sorted(bank.groups[kind]['pickup-'+str(stage)]):
                for choice in focused.choices[index]:
                    on = bank.y[index, burn:, choice['fly'], 2] > 1.
                    if on.any() and (~on).any():
                        found = (index, choice)
                        break
                if found:
                    break
            if found is None:
                raise ValueError('No mixed physical pickup actor for family/stage')
            index, choice = found
            fly = choice['fly']
            xs.append(torch.as_tensor(bank.x[index, :, fly:fly+1], device=device))
            ys.append(torch.as_tensor(bank.y[index, :, fly:fly+1], device=device))
            states.append(torch.as_tensor(bank.states[index, :, fly:fly+1], device=device))
            selections.append({'windowIndex': index, **bank.manifest['windows'][index], **choice})
    return (*(torch.cat(v, dim=1) for v in (xs, ys, states)), selections)


def perturb_cargo(x, mode):
    changed = x.detach().clone()
    if mode == 'zero':
        changed[..., 126:129] = 0.
    elif mode == 'cycle':
        changed[..., 126:129] = x[..., [127, 128, 126]]
    elif mode in ('scale-0.25', 'scale-4', 'scale-16'):
        changed[..., 126:129] *= float(mode.split('-')[1])
    else:
        raise ValueError('Unknown bounded cargo intervention')
    torch.testing.assert_close(changed[..., :126], x[..., :126], rtol=0, atol=0)
    torch.testing.assert_close(changed[..., 129:], x[..., 129:], rtol=0, atol=0)
    return changed


def response_summary(prediction, original, positive):
    return {'meanAbsoluteMotorChange': (prediction-original).abs().mean((0, 1)).tolist(),
            'maximumAbsoluteMotorChange': (prediction-original).abs().amax((0, 1)).tolist(),
            'gripThresholdChangedFraction': float(((prediction[..., 2]>1.) != (original[..., 2]>1.)).float().mean()),
            'gripMeanOnOriginalPositiveFrames': float(prediction[..., 2][positive].mean()),
            'gripMeanOnOriginalNegativeFrames': float(prediction[..., 2][~positive].mean()),
            'groupingUsesOriginalLabelsNotCounterfactualTargets': True,
            'isServiceEvidence': False}


def run_probe(model, x, y, state, burn):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Probe requires frozen model parameters and evaluation mode')
    before = model.checkpoint_hash()
    original_x, original_state = x.clone(), state.clone()
    prediction = frozen_predictions(model, x, state, burn)
    positive = y[burn:, :, 2] > 1.
    if not positive.any() or not (~positive).any():
        raise ValueError('Both original label groups required')
    baseline = {**training_scores(prediction, y[burn:]), **response_summary(prediction, prediction, positive)}
    interventions = []
    for mode in ('zero', 'cycle', 'scale-0.25', 'scale-4', 'scale-16'):
        altered = perturb_cargo(x, mode)
        response = frozen_predictions(model, altered, state, burn)
        if not torch.isfinite(response).all():
            raise FloatingPointError('Nonfinite response')
        row = {'mode': mode, **response_summary(response, prediction, positive)}
        interventions.append(row)
        print('CARGO_RESPONSE '+json.dumps(row), flush=True)
    # Exact input Jacobian of original positive-minus-negative grip response.
    # Differentiate only these96 frames, not the constant earlier prefix.
    differentiable = x.detach().clone().requires_grad_(True)
    p = recurrent_predictions(model, differentiable, state, burn, model.weights(), gradient_start=0)
    torch.testing.assert_close(p.detach(), prediction, rtol=1e-5, atol=1e-5)
    contrast = p[..., 2][positive].mean()-p[..., 2][~positive].mean()
    gradient, = torch.autograd.grad(contrast, differentiable)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite input gradient')
    groups = {}
    for name, low, high in [('legacy', 0, 30), ('color', 30, 126), ('cargo', 126, 129), ('status', 129, 297)]:
        g = gradient[..., low:high]
        groups[name] = {'absoluteSum': float(g.abs().sum()), 'maximumAbsolute': float(g.abs().max()),
                        'l2Norm': float(g.norm()), 'channels': high-low}
    assert model.checkpoint_hash() == before and all(p.grad is None for p in model.parameters())
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(state, original_state, rtol=0, atol=0)
    return {'baselineTrainingOnly': baseline, 'inputInterventions': interventions,
            'exactInputGradientOfGripGroupContrast': groups,
            'parametersUnchanged': True, 'inputsAndPrefixUnchanged': True,
            'optimizerUsed': False, 'candidateSaved': False, 'physicalWorldAdvanced': False,
            'isServiceEvidence': False, 'earlierPrefixDifferentiated': False,
            'interventionsMayBePhysicallyInconsistentAndOutOfRange': True}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostics')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    bank = DemonstrationCache(args.cache, before, model.fixed_hash, 64, 32, model.n)
    original_file_hash = file_hash(args.candidate)
    if original_file_hash != bank.manifest['candidateFileHash']:
        raise ValueError('Exact prefix parent file required')
    focused = EventFocusedDemonstrations(bank, args.dataset)
    x, y, state, selections = select_batch(focused, model.tonic.device)
    print('CARGO_RESPONSE_READY '+json.dumps({'parameterHash': before, 'actors': len(selections)}), flush=True)
    result = run_probe(model, x, y, state, 64)
    assert model.checkpoint_hash() == before and model.fingerprint() == model.fixed_hash
    assert file_hash(args.candidate) == original_file_hash
    result.update(parameterHash=before, fixedHash=model.fixed_hash, sourceHash=file_hash(Path(__file__)),
                  candidateFileHash=original_file_hash, cacheManifestHash=file_hash(args.cache/'manifest.json'),
                  physicalTrainingData=focused.record, selectedWindows=selections)
    atomic_json(args.out, result)
    print('CARGO_RESPONSE_COMPLETE '+json.dumps({'parametersUnchanged': True, 'isServiceEvidence': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'dataset', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
