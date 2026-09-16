"""Frozen old-prefix versus full-history-prefix audit; never trains a model.

The physical histories remain those of the parent behavior policy. Comparing
two state initializations on those SAME inputs isolates prefix effects, not
closed-loop covariate shift. Original training guards are reported unchanged.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_operation_sampling import OperationCache
from .layout_decision_resume import AnchoredOperationCache
from .layout_contextual_operation_probe import measure_bank
from .layout_margin_projected_train import margin_gate
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def prefix_difference(old, fresh):
    if (old['positiveFocus'] != fresh['positiveFocus']
            or old['protectedLoadedMask'] != fresh['protectedLoadedMask']):
        raise ValueError('Identical original labels and cargo required')
    report = {}
    for name in ('rawGrip', 'protectedLoadedGrip'):
        a, b = np.asarray(old[name]), np.asarray(fresh[name])
        if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Finite aligned outputs required')
        report[name] = {'maxAbs': float(np.max(np.abs(a-b))),
                        'rms': float(np.sqrt(np.mean((a-b)**2))),
                        'thresholdFlips': int(np.count_nonzero((a > 1.) != (b > 1.)))}
    return report


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve previous diagnostics')
    began = time.perf_counter()
    torch.set_num_threads(4)
    parent = load_model(args.root, args.parent).eval().requires_grad_(False)
    fixed = parent.fixed_hash
    if parent.checkpoint_hash() != args.parent_hash:
        raise ValueError('Wrong parent checkpoint')
    old = OperationCache(args.old_cache, args.dataset, args.parent_hash, fixed, parent.n)
    fresh = OperationCache(args.fresh_cache, args.dataset, args.candidate_hash, fixed, parent.n)
    bank = AnchoredOperationCache(fresh, old)
    bank.manifest['motionTargets'] = 'Original parent motion; only recurrent prefix refreshed'
    if (old.manifest['behaviorParameterHash'] != args.parent_hash
            or old.manifest['candidateFileHash'] != file_hash(args.parent)
            or fresh.manifest['candidateFileHash'] != file_hash(args.candidate)):
        raise ValueError('Require original parent behavior and exact checkpoint files')
    paths = [args.parent, args.candidate, args.dataset/'manifest.json']
    paths += [folder/name for folder in (args.old_cache, args.fresh_cache) for name in ('manifest.json', 'cache.npz')]
    paths += [args.dataset/e['file'] for e in old.manifest['datasetFiles']]
    hashes = {str(p.resolve()): file_hash(p) for p in paths}
    names = ('layout_prefix_transfer_audit.py', 'layout_operation_sampling.py', 'layout_decision_resume.py',
             'layout_contextual_operation_probe.py', 'layout_margin_projected_train.py',
             'layout_operation_focus.py', 'layout_demonstration_step_probe.py', 'layout_recovery_brain.py')
    sources = {**old.manifest['sourceHashes'], **fresh.manifest['sourceHashes'],
               **{n: file_hash(Path(__file__).with_name(n)) for n in names}}
    def unchanged(model, expected):
        if (model.checkpoint_hash() != expected or model.fingerprint() != fixed
                or any(p.grad is not None for p in model.parameters())
                or any(file_hash(Path(p)) != h for p,h in hashes.items())
                or any(file_hash(Path(__file__).with_name(n)) != h for n,h in sources.items())):
            raise ValueError('Original parameters, inputs or source changed')
    ids = list(range(len(old.manifest['windows'])))
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json', {'schema': 'frozen-prefix-transfer-audit-v1',
        'parentParameterHash': args.parent_hash, 'candidateParameterHash': args.candidate_hash,
        'behaviorParameterHash': args.parent_hash, 'fixedHash': fixed, 'windows': len(ids),
        'inputFileHashes': hashes, 'sourceHashes': sources, 'optimizerUsed': False,
        'teacherAtInference': False, 'isServiceEvidence': False,
        'limitation': 'Same recorded parent physical histories; not candidate closed-loop service.'})
    try:
        baseline = measure_bank(parent, old, ids)
        unchanged(parent, args.parent_hash)
        atomic_json(args.out/'parent-own-prefix.json', baseline)
        print('PREFIX_AUDIT_PARENT_COMPLETE', flush=True)
        del parent
        gc.collect(); torch.cuda.empty_cache()
        candidate = load_model(args.root, args.candidate).eval().requires_grad_(False)
        unchanged(candidate, args.candidate_hash)
        counterfactual = measure_bank(candidate, old, ids)
        atomic_json(args.out/'candidate-old-prefix.json', counterfactual)
        print('PREFIX_AUDIT_OLD_PREFIX_COMPLETE', flush=True)
        refreshed = measure_bank(candidate, bank, ids)
        unchanged(candidate, args.candidate_hash)
        atomic_json(args.out/'candidate-own-prefix.json', refreshed)
        result = {'originalFullBankGateWithOldPrefix': margin_gate(baseline, counterfactual, 0.),
                  'originalFullBankGateWithOwnPrefix': margin_gate(baseline, refreshed, 0.),
                  'prefixOnlyOutputDifference': prefix_difference(counterfactual, refreshed),
                  'seconds': time.perf_counter()-began, 'parametersUnchanged': True,
                  'optimizerUsed': False, 'candidateSaved': False, 'isServiceEvidence': False,
                  'device': torch.cuda.get_device_name(),
                  'qualification': 'Same-run frozen diagnostic with unchanged original guard thresholds. '
                  'A prefix difference is not proof that it caused closed-loop service failures.'}
        atomic_json(args.out/'result.json', result)
        print('PREFIX_TRANSFER_AUDIT_COMPLETE '+json.dumps(result), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'parent', 'candidate', 'old-cache', 'fresh-cache', 'dataset', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('parent-hash', 'candidate-hash'):
        p.add_argument('--'+name, required=True)
    main(p.parse_args())
