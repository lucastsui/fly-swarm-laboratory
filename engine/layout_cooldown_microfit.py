"""Matched V19 microfit, changing ONLY physically-equivalent grip pulse labels.

Independent fresh diagnostic optimizer from the SAME retained parent, SAME
examples/prefix, SAME learning rates/objective/gradient horizon. No runtime
teacher, graph change, learned decoder or physical assistance is introduced.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_physical_microfit import run_fit, validate_bounds
from .layout_grip_timing_probe import attested_batch
from .layout_demonstration_train import DemonstrationCache
from .layout_cooldown_labels import VERSION
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def load_revised_labels(folder, proof, original, device):
    manifest = json.loads((folder/'manifest.json').read_text())
    expected = {'version': VERSION, 'finished': True, 'physicalTrajectoriesUnchanged': True,
                'teacherAtInference': False, 'newRuntimeController': False, 'isServiceEvidence': False,
                'extraFrames': 4, 'extraSeconds': .2, 'proofSelections': proof['selectedWindows'],
                'datasetManifestHash': proof['physicalTrainingData']['datasetManifestHash']}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Label revision attestation mismatch')
    if {a['episode'] for a in manifest['audits']} != {s['episode'] for s in proof['selectedWindows']}:
        raise ValueError('Missing physical episode replay audit')
    if (any(not a['allChangesIgnoredByActualCooldown'] or not a['everyObservationBodyAndEventExactlyMatched']
            or a['isServiceEvidence'] for a in manifest['audits'])
            or any(Path(n).name != n or file_hash(Path(__file__).parent/n) != h
                   for n, h in manifest['originalPhysicsSourceHashes'].items())
            or manifest['sourceHash'] != file_hash(Path(__file__).with_name('layout_cooldown_labels.py'))
            or file_hash(folder/'labels.npz') != manifest['labelsFileHash']):
        raise ValueError('Replay physics or label bytes changed')
    with np.load(folder/'labels.npz', allow_pickle=False) as values:
        if set(values.files) != {'original', 'revised'}:
            raise ValueError('Wrong revised-label arrays')
        prior, revised = values['original'], values['revised']
        if (prior.shape != tuple(original.shape) or revised.shape != prior.shape
                or prior.dtype != np.float32 or revised.dtype != np.float32
                or not np.isfinite(revised).all()
                or not np.array_equal(prior, original.detach().cpu().numpy())
                or not np.array_equal(revised[..., :2], prior[..., :2])):
            raise ValueError('Labels differ from original attested physical actor inputs')
        changed = revised[..., 2] != prior[..., 2]
        if not changed.any() or np.any(changed & ((prior[..., 2] > 1.) | (revised[..., 2] <= 1.))):
            raise ValueError('Only extra protected grip activations allowed')
        target = torch.as_tensor(revised.copy(), device=device)
    return target, manifest


def main(args):
    validate_bounds(args.updates, args.save_every, args.lr, args.tonic_lr)
    if args.out.exists():
        raise FileExistsError('Preserve previous runs')
    torch.set_num_threads(4)
    torch.manual_seed(9390001)
    proof = json.loads(args.proof.read_text())
    parent_file = file_hash(args.candidate)
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    if before != proof['parameterHash'] or parent_file != proof['candidateFileHash']:
        raise ValueError('Matched original V19 parent required')
    initial_gains = model.log_gains.detach().cpu().numpy().copy()
    initial_tonic = model.tonic.detach().cpu().numpy().copy()
    bank = DemonstrationCache(args.cache, before, model.fixed_hash, 64, 32, model.n)
    if file_hash(args.cache/'manifest.json') != proof['cacheManifestHash']:
        raise ValueError('Matched original V19 prefix required')
    x, original_y, state = attested_batch(bank, proof, model.tonic.device)
    targets, revision = load_revised_labels(args.labels, proof, original_y, model.tonic.device)
    args.out.mkdir(parents=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(parentParameterHash=before, candidateFileHash=parent_file, fixedHash=model.fixed_hash,
                    interface=model.interface, canonicalOptimizer='Spark1 only; isolated matched diagnostic',
                    optimizerFreshByDesign=True, parentOptimizerRestored=False, remoteExperienceUsed=False,
                    decoderTrained=False, newNeuronsOrEdges=False, teacherAtInference=False, dopamineLearning=False,
                    physicalWorldAdvanced=False, serviceEvidence=False, requiresIndependentVerification=True,
                    labelRevision=revision, labelRevisionManifestHash=file_hash(args.labels/'manifest.json'),
                    physicalActorProofHash=file_hash(args.proof), cacheManifestHash=proof['cacheManifestHash'],
                    objective='8*parent-speed-MSE + 4*parent-turn-MSE + 2*revised-balanced-grip-MSE-and-contrast',
                    onlyMatchedExperimentChange='Extend grip labels by200ms where exact physical replay proves cooldown ignores them',
                    gradientFrames=96, lossFrames=32, earlierPrefixDifferentiated=False,
                    maximumPrefixAgeUpdates=80, prefixSemantics='Exact at parent only; constant earlier prefix',
                    trainingPrerequisite='revised-label recall>=.9, FPR<=.1, each parent-motion MSE<=.05')
    names = ('layout_cooldown_microfit.py', 'layout_cooldown_labels.py', 'layout_physical_microfit.py',
             'layout_grip_timing_probe.py', 'layout_cargo_response_probe.py', 'layout_demonstration_train.py',
             'layout_demonstration_step_probe.py', 'layout_recovery_train.py', 'layout_recovery_brain.py',
             'layout_excitability.py', 'layout_brain.py', 'supervised_steering.py', 'supervised_joint.py')
    manifest['sourceHashes'] = {n: file_hash(Path(__file__).parent/n) for n in names}
    atomic_json(args.out/'manifest.json', manifest)
    last_saved = 0

    def record(update, row, optimizer):
        nonlocal last_saved
        save_model(args.out/f'candidate-{update}.npz', model)
        torch.save({'optimizer': optimizer.state_dict(), 'updates': update}, args.out/f'optimizer-{update}.pt')
        row = {**row, 'gripMetricsUseRevisedTrainingLabels': True}
        atomic_json(args.out/f'fit-{update}.json', row)
        atomic_json(args.out/'status.json', {'finished': False, **row})
        last_saved = update

    try:
        result = run_fit(model, x, targets, state, 64, args.updates, args.save_every, args.lr, args.tonic_lr, record)
        result.update(audit=model.audit(initial_gains), gripMetricsUseRevisedTrainingLabels=True,
                      changedNeurons=int(np.count_nonzero(model.tonic.detach().cpu().numpy() != initial_tonic)))
        assert model.fingerprint() == model.fixed_hash and file_hash(args.candidate) == parent_file
        atomic_json(args.out/'result.json', result)
        atomic_json(args.out/'status.json', {'finished': True, 'gripMetricsUseRevisedTrainingLabels': True,
                                           **result['history'][-1]})
        print('COOLDOWN_MICROFIT_COMPLETE '+json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'error': str(error), 'type': type(error).__name__, 'lastSaved': last_saved})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'proof', 'labels', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=80)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--tonic-lr', type=float, default=.00001)
    main(parser.parse_args())
