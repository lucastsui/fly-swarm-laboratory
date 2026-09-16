"""Exact per-motor training gradients on original learner mistake histories.

No optimizer is constructed or executed and no parameters are perturbed.
Diagnose loss competition and grip margins before changing a curriculum.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_correction_sampling import CorrectionCache
from .layout_recovery_brain import load_model
from .layout_recovery_train import recurrent_predictions, motor_head_errors
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json
from .layout_recovery_horizon_probe import cosine
from .layout_adam_step_probe import adam_direction


def credit(model, x, y, state, adam, burn=64):
    before = model.checkpoint_hash()
    if model.surrogate_training:
        raise ValueError('Exact derivative comparison required')
    parameters = (model.log_gains, model.tonic)
    p = recurrent_predictions(model, x, state, burn, model.weights(), gradient_start=0)
    heads = motor_head_errors(p, y[burn:], balanced_grip=True)
    gradients = []
    for head in range(3):
        part = [g.detach() for g in torch.autograd.grad(heads[head], parameters, retain_graph=head < 2)]
        if any(not torch.isfinite(g).all() for g in part):
            raise FloatingPointError('Nonfinite motor gradient')
        gradients.append(part)
    result = {'headLosses': heads.detach().cpu().tolist(), 'parameterBlocks': {}}
    groups = adam['optimizer']['param_groups']
    if len(groups) != 2 or any(len(g['params']) != 1 for g in groups):
        raise ValueError('Wrong saved canonical Adam shape')
    for block, name in enumerate(('log_gains', 'tonic')):
        speed, turn, grip = [g[block] for g in gradients]
        motion, handling = 4*speed+2*turn, 2*grip
        saved = adam['optimizer']['state'][groups[block]['params'][0]]
        if (float(saved['step']) != adam['updates'] or saved['exp_avg'].shape != parameters[block].shape
                or saved['exp_avg_sq'].shape != parameters[block].shape
                or not torch.isfinite(saved['exp_avg']).all() or not torch.isfinite(saved['exp_avg_sq']).all()):
            raise ValueError('Saved optimizer does not match brain')
        direction = adam_direction(motion+handling, saved, groups[block])
        result['parameterBlocks'][name] = {
            'headNorms': [float(g.norm()) for g in (speed, turn, grip)],
            'headNonzeroCounts': [int(torch.count_nonzero(g)) for g in (speed, turn, grip)],
            'weightedMotionNorm': float(motion.norm()), 'weightedGripNorm': float(handling.norm()),
            'motionGripCosine': cosine(motion, handling),
            'predictedNextSavedAdamHeadSlopes': [float((g*direction).sum()) for g in (speed, turn, grip)]}
    assert model.checkpoint_hash() == before and all(p.grad is None for p in parameters)
    return p.detach().cpu().numpy(), result


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostic')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).train()
    before, fixed = model.checkpoint_hash(), model.fixed_hash
    if before != args.expected_parameter_hash:
        raise ValueError('Wrong selected brain')
    bank = CorrectionCache(args.cache, args.dataset, before, fixed, model.n)
    if bank.manifest['candidateFileHash'] != file_hash(args.candidate):
        raise ValueError('Exact prefix parent required')
    adam = torch.load(args.optimizer, map_location=model.tonic.device, weights_only=True)
    paths = [args.candidate, args.optimizer, args.cache/'manifest.json', args.cache/'cache.npz',
             args.dataset/'manifest.json']
    paths.extend(args.dataset/e['file'] for e in bank.manifest['datasetFiles'])
    hashes = {str(p.resolve()): file_hash(p) for p in paths}
    names = set(bank.manifest['sourceHashes']) | {'layout_motor_credit_probe.py', 'layout_correction_sampling.py',
            'layout_recovery_train.py', 'layout_adam_step_probe.py', 'layout_recovery_horizon_probe.py'}
    sources = {n: file_hash(Path(__file__).parent/n) for n in sorted(names)}
    ids = bank.diagnostic_indexes()
    x, y, state = bank.batch(ids, model.tonic.device)
    predictions, result = credit(model, x, y, state, adam)
    labels = y[64:].detach().cpu().numpy()
    rows = []
    for actor, index in enumerate(ids):
        on = labels[:, actor, 2] > 1
        grip = predictions[:, actor, 2]
        rows.append({'selection': bank.manifest['windows'][index], 'positiveFrames': int(on.sum()),
                     'gripRawQuantiles': np.quantile(grip, [0, .25, .5, .75, 1]).tolist(),
                     'gripRecall': float((grip[on] > 1).mean()) if on.any() else None,
                     'gripFalsePositiveRate': float((grip[~on] > 1).mean()) if (~on).any() else None,
                     'rawPredictions': predictions[:, actor].tolist(), 'originalLabels': labels[:, actor].tolist()})
    if (model.checkpoint_hash() != before or model.fingerprint() != fixed
            or any(file_hash(Path(p)) != h for p, h in hashes.items())
            or any(file_hash(Path(__file__).parent/n) != h for n, h in sources.items())):
        raise AssertionError('Diagnostic changed original brain, optimizer, sources or data')
    result.update(parameterHash=before, fixedHash=fixed, savedAdamUpdate=adam['updates'],
                  optimizerUsed=False, hypotheticalDirectionOnly=True, parametersUnchanged=True,
                  candidateSaved=False, physicalActionsExecuted=False, isServiceEvidence=False,
                  trainingCasesDeliberatelySelected=True, inputFileHashes=hashes, sourceHashes=sources, trials=rows)
    atomic_json(args.out, result)
    print('MOTOR_CREDIT_COMPLETE '+json.dumps({k: v for k, v in result.items()
          if k not in ('trials', 'inputFileHashes', 'sourceHashes')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'optimizer', 'cache', 'dataset', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-parameter-hash', required=True)
    main(p.parse_args())
