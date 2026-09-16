"""Resume ONLY a completed failed synthetic prerequisite, never live training.

Original data/parent-motion labels and Adam moments are retained. This is not a
general physical-rollout resumer: world/teacher state is not checkpointed there.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .layout_grip_calibration import CalibrationBank, VERSION
from .layout_microfit import select_microfit_scenes


MATCHED_CONTROLS = ('worlds', 'burn', 'gradient_frames', 'lr', 'tonic_lr', 'reset_seconds', 'seed',
                    'balanced_curriculum', 'balanced_grip', 'grip_calibration', 'calibration_scenes',
                    'grip_margin', 'synapse_eps', 'block_clip', 'exact_gradient',
                    'ranking_weight', 'full_calibration_gradient')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_continuation(args, model):
    source = Path(args.resume_microfit_run).resolve()
    if source == args.out.resolve():
        raise ValueError('Continuation requires a separate output directory')
    manifest, result, status, fit, history = [json.loads((source/name).read_text()) for name in
                                           ('manifest.json', 'result.json', 'status.json',
                                            'microfit-result.json', 'history.json')]
    start = result['updates']
    if (not status['finished'] or fit['passed'] or result['microfitGatePassed'] is not False
            or start != manifest['microfit_updates'] or start != fit['update']
            or result['remoteConsumed'] != 0 or not history
            or any(row['trainingPhase'] != 'microfit-prerequisite' for row in history)):
        raise ValueError('Only a finished, failed, exclusively synthetic prerequisite can resume')
    if not start < args.microfit_updates <= args.updates:
        raise ValueError('Continuation must extend the bounded prerequisite')
    for key in MATCHED_CONTROLS:
        if manifest.get(key, False) != getattr(args, key, False):
            raise ValueError('Continuation changes control: '+key)
    expected = source/f'candidate-{start}.npz'
    before = model.checkpoint_hash()
    if (args.candidate.resolve() != expected or result['checkpointHash'] != before
            or status['checkpoint']['parameterHash'] != before
            or manifest['fixedHash'] != model.fixed_hash
            or file_hash(expected) != status['checkpoint']['sha256']):
        raise ValueError('Continuation checkpoint/interface mismatch')
    parent = manifest.get('calibrationParentParameterHash', manifest['parentParameterHash'])
    bank = object.__new__(CalibrationBank)
    with np.load(source/'calibration-bank.npz', allow_pickle=False) as data:
        if str(data['version']) != VERSION or str(data['parent_hash']) != parent:
            raise ValueError('Original calibration identity mismatch')
        bank.observations, bank.targets = data['observations'].copy(), data['targets'].copy()
    bank.burn, bank.parent_hash = args.burn, parent
    bank.metadata = json.loads((source/'calibration-scenes.json').read_text())
    if (bank.observations.shape != (args.burn+args.gradient_frames, 4*args.calibration_scenes, 297)
            or bank.targets.shape != (args.gradient_frames, 4*args.calibration_scenes, 3)
            or len(bank.metadata) != args.calibration_scenes
            or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (bank.observations, bank.targets))):
        raise ValueError('Malformed original calibration bank')
    microfit = bank.subset(select_microfit_scenes(bank.metadata, bank.targets))
    with np.load(source/'microfit-bank.npz', allow_pickle=False) as old:
        if (str(old['parent_hash']) != parent or not np.array_equal(old['observations'], microfit.observations)
                or not np.array_equal(old['targets'], microfit.targets)):
            raise ValueError('Original four-scene subset mismatch')
    optimizer_path = source/f'optimizer-{start}.pt'
    paths = [expected, optimizer_path, source/'calibration-bank.npz', source/'calibration-scenes.json',
             source/'microfit-bank.npz', source/'manifest.json', source/'result.json', source/'status.json']
    record = {'sourceRun': str(source), 'sourceRunId': manifest['runId'], 'startUpdate': start,
              'parameterHash': before, 'originalCalibrationParentHash': parent,
              'optimizerMomentsPreserved': True, 'trainingDataAndMotionTargetsPreserved': True,
              'sourceFileSHA256': {p.name: file_hash(p) for p in paths}}
    return {'start': start, 'bank': bank, 'optimizerPath': optimizer_path, 'record': record}


def restore_optimizer(optimizer, model, continuation):
    # This is the task's own local optimizer file, NOT accepted network input.
    # weights_only keeps loading limited to tensors and ordinary value types.
    path = continuation['optimizerPath']
    if file_hash(path) != continuation['record']['sourceFileSHA256'][path.name]:
        raise ValueError('Optimizer file changed during continuation setup')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['updates'] != continuation['start']:
        raise ValueError('Optimizer step/checkpoint mismatch')
    expected = [(g['lr'], g['eps']) for g in optimizer.param_groups]
    optimizer.load_state_dict(payload['optimizer'])
    if [(g['lr'], g['eps']) for g in optimizer.param_groups] != expected:
        raise ValueError('Optimizer controls differ from preserved experiment')
    for parameter in model.parameters():
        state = optimizer.state[parameter]
        if float(state['step']) != continuation['start']:
            raise ValueError('Adam counter mismatch')
        for key in ('exp_avg', 'exp_avg_sq'):
            if state[key].shape != parameter.shape or not torch.isfinite(state[key]).all():
                raise ValueError('Malformed Adam moment')
    return payload['rng']
