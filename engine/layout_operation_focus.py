"""Exact physical decision-frame TRAINING loss and read-only diagnostics.

The operation metadata selects a loss frame, never a brain input or action.
Keep all 96 recurrent frames and the original observations/labels. Exclude
nearby grip labels from this objective, not from the physical history.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_operation_sampling import OperationCache
from .layout_operation_prefix import CONTEXTS, KINDS
from .layout_grip_objective import threshold_grip_loss
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

VERSION = 'original-physical-decision-frame-grip-v1'


def focus_values(predictions, labels, selections):
    if predictions.shape != labels.shape or predictions.shape != (32, len(selections), 3):
        raise ValueError('Original aligned 32-frame predictions and labels required')
    if any(s['focusTick']-s['lossStart'] != 16 or s['stop']-s['start'] != 96
           or s['stop']-s['lossStart'] != 32 or s['category'] not in CONTEXTS
           or s['kind'] not in KINDS for s in selections):
        raise ValueError('Original physical focus selection required')
    target = labels[16, :, 2]
    desired = torch.as_tensor([s['category'] != CONTEXTS[2] for s in selections], device=target.device)
    if not torch.equal(target > 1., desired):
        raise ValueError('Physical context disagrees with original focus label')
    return predictions[16, :, 2], target


def focused_heads(predictions, labels, parent_motion, selections):
    if parent_motion.shape != predictions.shape[:-1]+(2,):
        raise ValueError('Original aligned parent motion required')
    grip, target = focus_values(predictions, labels, selections)
    motion = (predictions[..., :2]-parent_motion).square().mean((0, 1))
    handling = threshold_grip_loss(grip, target, .1)
    return torch.stack((motion[0], motion[1], handling))


@torch.no_grad()
def focus_fit(model, bank, all_windows=False):
    before, mode = model.checkpoint_hash(), model.training
    ids = list(range(len(bank.manifest['windows']))) if all_windows else bank.diagnostic_indexes()
    try:
        model.eval()
        predictions, labels, anchors = [], [], []
        for first in range(0, len(ids), 16):
            x, y, state, motion = bank.batch(ids[first:first+16], model.tonic.device)
            predictions.append(frozen_predictions(model, x, state, 64))
            labels.append(y[64:]); anchors.append(motion)
        p, y, a = (torch.cat(values, dim=1) for values in (predictions, labels, anchors))
        selections = [bank.manifest['windows'][i] for i in ids]
        grip, target = focus_values(p, y, selections)
        on = target > 1.; active = grip > 1.
        rows = []
        for kind in KINDS:
            for context in CONTEXTS:
                indexes = [i for i, s in enumerate(selections) if s['kind'] == kind and s['category'] == context]
                values = grip[indexes].cpu().numpy()
                rows.append({'kind': kind, 'context': context, 'windows': len(indexes),
                             'correctFocusDecisions': int((active[indexes] == on[indexes]).sum()),
                             'rawGripQuantiles': np.quantile(values, [0, .25, .5, .75, 1]).tolist(),
                             'nearbyLabelDisagreementFraction': float(((y[:, indexes, 2] > 1.) != on[indexes]).float().mean())})
        return {'parameterHash': before, 'prefixParameterHash': bank.manifest['parameterHash'],
                'prefixIsExactForCandidate': before == bank.manifest['parameterHash'],
                'windowIndexes': ids, 'trainingSubsetOnly': True, 'isServiceEvidence': False,
                'objectiveHeads': focused_heads(p, y, a, selections).cpu().tolist(),
                'parentMotionMSE': (p[..., :2]-a).square().mean((0, 1)).cpu().tolist(),
                'positiveFocusLabels': int(on.sum()), 'negativeFocusLabels': int((~on).sum()),
                'focusRecall': float(active[on].float().mean()),
                'focusFalsePositiveRate': float(active[~on].float().mean()), 'contexts': rows}
    finally:
        model.train(mode)
        if model.checkpoint_hash() != before:
            raise AssertionError('Read-only diagnostic changed brain')


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostic')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    if model.checkpoint_hash() != args.expected_parameter_hash:
        raise ValueError('Wrong frozen candidate')
    bank = OperationCache(args.cache, args.dataset, args.prefix_parameter_hash, model.fixed_hash, model.n)
    report = focus_fit(model, bank, True)
    report.update(candidateFileHash=file_hash(args.candidate),
                  cacheFileHash=file_hash(args.cache/'cache.npz'), datasetManifestHash=file_hash(args.dataset/'manifest.json'),
                  sourceHashes={n: file_hash(Path(__file__).parent/n) for n in
                                ('layout_operation_focus.py', 'layout_operation_sampling.py', 'layout_demonstration_step_probe.py')},
                  optimizerUsed=False, parametersUnchanged=True,
                  qualification='96-frame replay from stored parent prefix; not candidate full-history service')
    atomic_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('windowIndexes', 'contexts', 'sourceHashes')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'dataset', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-parameter-hash', required=True)
    p.add_argument('--prefix-parameter-hash', required=True)
    main(p.parse_args())
