"""Bounded learnability test on twelve REAL physical training histories.

All decision computation and the fixed motor decoder stay in the connectome.
Only this offline diagnostic fits repeated examples: original teacher grip
labels, with parent motor outputs as motion-preservation targets. It is not a
service demonstration, distributed production run, or dopamine learning.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .layout_cargo_response_probe import select_batch
from .layout_demonstration_train import DemonstrationCache, training_scores
from .layout_demonstration_focus import EventFocusedDemonstrations
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_train import recurrent_predictions, motor_head_errors, optimize
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def validate_bounds(updates, save_every, lr, tonic_lr):
    if not 1 <= updates <= 80 or not 1 <= save_every <= updates:
        raise ValueError('Bounded 1..80 updates and checkpoint interval required')
    if not all(np.isfinite(v) and 0 < v <= limit for v, limit in ((lr, .003), (tonic_lr, .00001))):
        raise ValueError('Positive bounded physical-microfit learning rates required')


def fit_passes(scores):
    return (scores['gripRecall'] is not None and scores['gripRecall'] >= .9
            and scores['gripFalsePositiveRate'] is not None and scores['gripFalsePositiveRate'] <= .1
            and all(v <= .05 for v in scores['parentMotionMSE']))


def fitting_scores(prediction, target):
    scores = training_scores(prediction, target)
    scores['parentMotionMSE'] = scores.pop('teacherMotionMSE')
    scores['trainingPrerequisitePassed'] = fit_passes(scores)
    return scores


def run_fit(model, x, labels, state, burn, updates, save_every, lr, tonic_lr, record):
    validate_bounds(updates, save_every, lr, tonic_lr)
    if not 0 < burn < len(x) or x.shape[:2] != labels.shape[:2]:
        raise ValueError('Matching recurrent observations and labels required')
    before = model.checkpoint_hash()
    original_x, original_labels, original_state = x.clone(), labels.clone(), state.clone()
    model.eval().requires_grad_(False)
    reference = frozen_predictions(model, x, state, burn)
    target = labels[burn:].detach().clone()
    target[..., :2] = reference[..., :2]
    if not (target[..., 2] > 1.).any() or not (target[..., 2] <= 1.).any():
        raise ValueError('Actual positive and negative grip examples required')
    # A separate explicitly fresh diagnostic optimizer, not a false claim of
    # exact continuation. The retained parent's optimizer is never read/written.
    model.train().requires_grad_(True)
    optimizer = torch.optim.Adam([
        {'params': [model.log_gains], 'lr': lr, 'eps': 1e-14},
        {'params': [model.tonic], 'lr': tonic_lr, 'eps': 1e-10}])
    began = time.perf_counter()
    record(0, {'update': 0, 'parameterHash': before, **fitting_scores(reference, target)}, optimizer)
    history = []
    for update in range(1, updates+1):
        optimizer.zero_grad(set_to_none=True)
        prediction = recurrent_predictions(model, x, state, burn, model.weights(), gradient_start=0)
        heads = motor_head_errors(prediction, target, balanced_grip=True)
        norms = {}
        loss = optimize(model, optimizer, heads, (8., 4., 2.), True, norms)
        row = {'update': update, 'lossBeforeUpdate': loss, 'seconds': time.perf_counter()-began,
               'gradientNormsBeforeClipping': norms, 'objectiveByHeadBeforeUpdate': heads.detach().cpu().tolist(),
               'isServiceEvidence': False}
        del prediction, heads
        if update % save_every == 0 or update == updates:
            model.eval()
            current = frozen_predictions(model, x, state, burn)
            row.update(parameterHash=model.checkpoint_hash(), **fitting_scores(current, target))
            record(update, row, optimizer)
            model.train()
        history.append(row)
        print('PHYSICAL_MICROFIT '+json.dumps(row), flush=True)
    for original, value in ((original_x, x), (original_labels, labels), (original_state, state)):
        torch.testing.assert_close(original, value, rtol=0, atol=0)
    return {'initialParameterHash': before, 'finalParameterHash': model.checkpoint_hash(),
            'updates': updates, 'history': history, 'inputsLabelsAndPrefixUnchanged': True,
            'seconds': time.perf_counter()-began, 'isServiceEvidence': False,
            'trainingPrerequisitePassed': history[-1]['trainingPrerequisitePassed'],
            'optimizerFreshByDesign': True, 'physicalWorldAdvanced': False}


def main(args):
    validate_bounds(args.updates, args.save_every, args.lr, args.tonic_lr)
    if args.out.exists():
        raise FileExistsError('Preserve diagnostic runs')
    torch.set_num_threads(4)
    torch.manual_seed(9390001)
    source_hash = file_hash(args.candidate)
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    initial_gains = model.log_gains.detach().cpu().numpy().copy()
    initial_tonic = model.tonic.detach().cpu().numpy().copy()
    bank = DemonstrationCache(args.cache, before, model.fixed_hash, 64, 32, model.n)
    if source_hash != bank.manifest['candidateFileHash']:
        raise ValueError('Exact source candidate required for prefix')
    focused = EventFocusedDemonstrations(bank, args.dataset)
    x, y, state, selections = select_batch(focused, model.tonic.device)
    args.out.mkdir(parents=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(parentParameterHash=before, candidateFileHash=source_hash, fixedHash=model.fixed_hash,
                    interface=model.interface, physicalTrainingData=focused.record, selections=selections,
                    cacheManifestHash=file_hash(args.cache/'manifest.json'),
                    canonicalOptimizer='Spark1 only; isolated repeated-example diagnostic',
                    optimizerFreshByDesign=True, parentOptimizerRestored=False,
                    remoteExperienceUsed=False, decoderTrained=False, newNeuronsOrEdges=False,
                    teacherAtInference=False, physicalWorldAdvanced=False, dopamineLearning=False,
                    objective='8*parent-speed-MSE + 4*parent-turn-MSE + 2*original-balanced-grip-MSE-and-contrast',
                    originalTeacherGripLabelsUnchanged=True, labelsAreTrainingOnly=True,
                    gradientFrames=96, lossFrames=32, earlierPrefixDifferentiated=False,
                    maximumPrefixAgeUpdates=80, prefixSemantics='Exact at parent only; no claim of updated full-history state',
                    trainingPrerequisite='recall>=.9, false-positive<=.1, each parent-motion MSE<=.05',
                    serviceEvidence=False, requiresIndependentVerification=True)
    names = ('layout_physical_microfit.py', 'layout_cargo_response_probe.py', 'layout_demonstration_focus.py',
             'layout_demonstration_train.py', 'layout_demonstration_step_probe.py', 'layout_recovery_train.py',
             'layout_recovery_brain.py', 'layout_excitability.py', 'layout_brain.py', 'supervised_steering.py',
             'supervised_joint.py')
    manifest['sourceHashes'] = {n: file_hash(Path(__file__).parent/n) for n in names}
    atomic_json(args.out/'manifest.json', manifest)
    last_saved = 0

    def record(update, row, optimizer):
        nonlocal last_saved
        save_model(args.out/f'candidate-{update}.npz', model)
        torch.save({'optimizer': optimizer.state_dict(), 'updates': update}, args.out/f'optimizer-{update}.pt')
        atomic_json(args.out/f'fit-{update}.json', row)
        atomic_json(args.out/'status.json', {'finished': False, **row})
        last_saved = update

    try:
        result = run_fit(model, x, y, state, 64, args.updates, args.save_every, args.lr, args.tonic_lr, record)
        result.update(audit=model.audit(initial_gains),
                      changedNeurons=int(np.count_nonzero(model.tonic.detach().cpu().numpy() != initial_tonic)))
        assert model.fingerprint() == model.fixed_hash and file_hash(args.candidate) == source_hash
        atomic_json(args.out/'result.json', result)
        atomic_json(args.out/'status.json', {'finished': True, **result['history'][-1]})
        print('PHYSICAL_MICROFIT_COMPLETE '+json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'error': str(error), 'type': type(error).__name__, 'lastSaved': last_saved})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'dataset', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=80)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--tonic-lr', type=float, default=.00001)
    main(parser.parse_args())
