"""Restored read-only step probes on genuine learner error histories.

Use the exact C60 full-history states and saved C60 Adam moments. Compare
existing exact/surrogate derivatives and step scales without an optimizer
step, saved candidate, physical action, or persistent parameter change.
These deliberately selected TRAINING cases are not service evidence.
"""
import argparse
import json
from pathlib import Path
import torch
from .layout_correction_curriculum_train import source_run, restore_state
from .layout_correction_sampling import CorrectionCache
from .layout_demonstration_step_probe import run_probe
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve original diagnostics')
    torch.set_num_threads(4)
    source, manifest, _ = source_run(args.source)
    candidate, adam = source/'candidate-60.npz', source/'optimizer-60.pt'
    model = load_model(args.root, candidate).eval()
    before, fixed = model.checkpoint_hash(), model.fixed_hash
    if before != manifest['parentParameterHash'] or fixed != manifest['fixedHash']:
        raise ValueError('Wrong exact parent for the correction probe')
    bank = CorrectionCache(args.cache, args.dataset, before, fixed, model.n)
    if bank.manifest['candidateFileHash'] != file_hash(candidate):
        raise ValueError('Wrong original parent file')
    optimizer, _ = restore_state(model, torch.load(adam, map_location='cpu', weights_only=True),
                                 manifest['learningRates'])
    paths = [candidate, adam, args.cache/'manifest.json', args.cache/'cache.npz', args.dataset/'manifest.json']
    paths.extend(args.dataset/e['file'] for e in bank.manifest['datasetFiles'])
    hashes = {str(p.resolve()): file_hash(p) for p in paths}
    names = set(bank.manifest['sourceHashes']) | set(manifest['sourceHashes']) | {
        'layout_correction_step_probe.py', 'layout_demonstration_step_probe.py',
        'layout_adam_step_probe.py', 'layout_recovery_step_probe.py', 'layout_recovery_horizon_probe.py'}
    sources = {n: file_hash(Path(__file__).parent/n) for n in sorted(names)}
    indexes = bank.diagnostic_indexes()
    x, y, state = bank.batch(indexes, model.tonic.device)
    print('CORRECTION_PROBE_READY '+json.dumps({'parent': before, 'windows': indexes,
                                               'optimizerStepsWillExecute': False}), flush=True)
    result = run_probe(model, x, y, state, optimizer, 64, scales=(.1, 1., 4.))
    if (model.checkpoint_hash() != before or model.fingerprint() != fixed
            or any(file_hash(Path(p)) != h for p, h in hashes.items())
            or any(file_hash(Path(__file__).parent/n) != h for n, h in sources.items())):
        raise AssertionError('Diagnostic changed brain, original data or source')
    result.update(checkpointHash=before, fixedHash=fixed, savedAdamUpdate=60,
                  prefixParameterHash=bank.manifest['parameterHash'],
                  behaviorParameterHash=bank.manifest['behaviorParameterHash'],
                  windows=[bank.manifest['windows'][i] for i in indexes],
                  inputFileHashes=hashes, sourceHashes=sources,
                  optimizerStepExecuted=False, candidateSaved=False,
                  physicalActionsExecuted=False, teacherActions=False,
                  trainingCasesDeliberatelySelected=True, isServiceEvidence=False)
    atomic_json(args.out, result)
    print('CORRECTION_PROBE_COMPLETE '+json.dumps({'initial': result['initial'],
          'bestTrialLoss': min(r['loss'] for r in result['trials']),
          'parametersRestoredExactly': result['parametersRestoredExactly']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'source', 'cache', 'dataset', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
