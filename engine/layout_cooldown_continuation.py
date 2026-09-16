"""Continue a FINISHED V20 fit with saved Adam and freshly encoded histories.

Original C100 movement targets and revised grip labels remain fixed. No source
file is overwritten. One bounded80-update segment, not an unbounded loop or
service evidence. Full signed connectome, fixed embodiment/decoder unchanged.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json
from .layout_recovery_brain import load_model, save_model
from .layout_demonstration_train import DemonstrationCache
from .layout_grip_timing_probe import attested_batch
from .layout_cooldown_microfit import load_revised_labels
from .layout_demonstration_step_probe import frozen_predictions
from .layout_demonstration_cache import load_dataset, prefix_states
from .layout_physical_microfit import fitting_scores, validate_bounds
from .layout_recovery_train import recurrent_predictions, motor_head_errors, optimize


def restore_adam(optimizer, payload, update):
    if set(payload) != {'optimizer', 'updates'} or payload['updates'] != update:
        raise ValueError('Wrong deterministic microfit optimizer checkpoint')
    saved = payload['optimizer']
    groups, expected = saved['param_groups'], optimizer.state_dict()['param_groups']
    if len(groups) != len(expected):
        raise ValueError('Optimizer group count changed')
    identifiers = []
    for actual, wanted, live in zip(groups, expected, optimizer.param_groups):
        if ({k: v for k, v in actual.items() if k != 'params'} !=
                {k: v for k, v in wanted.items() if k != 'params'}
                or len(actual['params']) != len(live['params'])):
            raise ValueError('Optimizer settings or parameter order changed')
        for identifier, p in zip(actual['params'], live['params']):
            identifiers.append(identifier)
            value = saved['state'][identifier]
            if (set(value) != {'step', 'exp_avg', 'exp_avg_sq'}
                    or not isinstance(value['step'], torch.Tensor) or value['step'].numel() != 1
                    or float(value['step']) != update):
                raise ValueError('Adam step identity mismatch')
            for key in ('exp_avg', 'exp_avg_sq'):
                v = value[key]
                if v.shape != p.shape or v.dtype != p.dtype or not torch.isfinite(v).all():
                    raise ValueError('Malformed Adam moment')
            if (value['exp_avg_sq'] < 0).any():
                raise ValueError('Negative Adam variance')
    if len(set(identifiers)) != len(identifiers) or set(saved['state']) != set(identifiers):
        raise ValueError('Optimizer state ownership mismatch')
    optimizer.load_state_dict(saved)


def refresh_selected_prefixes(model, episodes, selections):
    indexes = sorted({s['episode'] for s in selections})
    observations = np.concatenate([episodes[i][0]['observations'] for i in indexes], axis=1)
    pending = {}
    states = np.empty((model.n, len(selections)), np.float32)
    written = np.zeros(len(selections), bool)
    for column, s in enumerate(selections):
        meta = episodes[s['episode']][1]
        if meta['seed'] != s['seed'] or meta['kind'] != s['kind']:
            raise ValueError('Fresh prefix episode identity mismatch')
        pending.setdefault(s['start'], []).append((column, 4*indexes.index(s['episode'])+s['fly']))
    for frame, state in prefix_states(model, observations, pending, progress=True):
        for column, actor in pending[frame]:
            states[:, column] = state[:, actor].cpu().numpy()
            written[column] = True
    if not written.all() or not np.isfinite(states).all():
        raise ValueError('Incomplete refreshed physical prefixes')
    return torch.as_tensor(states, device=model.tonic.device)


def train_segment(model, optimizer, x, targets, state, start, updates, save_every, record):
    validate_bounds(updates, save_every, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
    if targets.shape != x.shape[:2]+(3,) or len(x) != 96:
        raise ValueError('Exact96-frame training targets required')
    model.train().requires_grad_(True)
    history, began = [], time.perf_counter()
    for update in range(start+1, start+updates+1):
        optimizer.zero_grad(set_to_none=True)
        p = recurrent_predictions(model, x, state, 64, model.weights(), gradient_start=0)
        heads = motor_head_errors(p, targets[64:], balanced_grip=True)
        norms = {}
        loss = optimize(model, optimizer, heads, (8., 4., 2.), True, norms)
        row = {'update': update, 'lossBeforeUpdate': loss, 'seconds': time.perf_counter()-began,
               'gradientNormsBeforeClipping': norms, 'isServiceEvidence': False,
               'gripMetricsUseRevisedTrainingLabels': True}
        del p, heads
        if update % save_every == 0 or update == start+updates:
            model.eval()
            prediction = frozen_predictions(model, x, state, 64)
            row.update(parameterHash=model.checkpoint_hash(), **fitting_scores(prediction, targets[64:]))
            record(update, row)
            model.train()
        history.append(row)
        print('CONTINUED_MICROFIT '+json.dumps(row), flush=True)
    return history


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve all runs')
    source = args.source.resolve()
    m, result, status = [json.loads((source/n).read_text()) for n in ('manifest.json', 'result.json', 'status.json')]
    start = args.checkpoint
    controls = {'optimizerFreshByDesign': True, 'remoteExperienceUsed': False,
                'teacherAtInference': False, 'decoderTrained': False, 'newNeuronsOrEdges': False,
                'dopamineLearning': False, 'gradientFrames': 96, 'lossFrames': 32}
    if any(m.get(k) != v for k, v in controls.items()):
        raise ValueError('Source is not the matched isolated V20 fit')
    if (m.get('onlyMatchedExperimentChange') != 'Extend grip labels by200ms where exact physical replay proves cooldown ignores them'
            or not status['finished'] or result['updates'] != m['updates']
            or status['update'] != result['updates'] or result['finalParameterHash'] != status['parameterHash']
            or not result['gripMetricsUseRevisedTrainingLabels']
            or [r['update'] for r in result['history']] != list(range(1, result['updates']+1))
            or not all(np.isfinite(r['lossBeforeUpdate']) for r in result['history'])
            or not 0 < start <= result['updates'] or start % m['save_every']):
        raise ValueError('Only a completed attested V20 checkpoint can continue')
    if (not all(result['audit'][k] for k in ('finite', 'signsPreserved', 'fixedGraphSensoryDecoderDynamicsUnchanged'))
            or any(Path(n).name != n or file_hash(Path(__file__).parent/n) != h for n, h in m['sourceHashes'].items())):
        raise ValueError('Source training/graph audit changed')
    validate_bounds(args.updates, 20, m['lr'], m['tonic_lr'])
    fit = json.loads((source/f'fit-{start}.json').read_text())
    if (fit['parameterHash'] != result['history'][start-1]['parameterHash']
            or fit.get('gripMetricsUseRevisedTrainingLabels') is not True):
        raise ValueError('Saved checkpoint and completed history disagree')
    def original_path(key):
        p = Path(m[key])
        p = (source.parent/p).resolve() if not p.is_absolute() else p.resolve()
        if not p.is_relative_to(source.parent):
            raise ValueError('Source input leaves experiment workspace')
        return p
    original_candidate, cache, proof_path, label_folder = [original_path(k) for k in ('candidate', 'cache', 'proof', 'labels')]
    selected = source/f'candidate-{start}.npz'
    optimizer_path = source/f'optimizer-{start}.pt'
    paths = [source/n for n in ('manifest.json', 'result.json', 'status.json', f'fit-{start}.json')]
    paths += [selected, optimizer_path, original_candidate, cache/'manifest.json', cache/'cache.npz',
              proof_path, label_folder/'manifest.json', label_folder/'labels.npz', args.dataset/'manifest.json']
    source_hashes = {str(p): file_hash(p) for p in paths}
    if (file_hash(original_candidate) != m['candidateFileHash'] or file_hash(cache/'manifest.json') != m['cacheManifestHash']
            or file_hash(proof_path) != m['physicalActorProofHash']
            or file_hash(label_folder/'manifest.json') != m['labelRevisionManifestHash']):
        raise ValueError('Original training inputs changed')
    torch.set_num_threads(4)
    proof = json.loads(proof_path.read_text())
    model = load_model(args.root, original_candidate).eval().requires_grad_(False)
    bank = DemonstrationCache(cache, model.checkpoint_hash(), model.fixed_hash, 64, 32, model.n)
    x, original_y, old_state = attested_batch(bank, proof, model.tonic.device)
    targets, label_manifest = load_revised_labels(label_folder, proof, original_y, model.tonic.device)
    # Preserve the ORIGINAL C100 motor targets, not this new segment's parent.
    targets = targets.clone()
    targets[64:, :, :2] = torch.as_tensor(frozen_predictions(model, x, old_state, 64)[..., :2])
    target_hash = hashlib.sha256(targets.cpu().numpy().tobytes()).hexdigest()
    del model, bank
    torch.cuda.empty_cache()
    model = load_model(args.root, selected).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    if before != fit['parameterHash'] or model.fixed_hash != m['fixedHash']:
        raise ValueError('Selected brain identity mismatch')
    original_prefix_fit = fitting_scores(frozen_predictions(model, x, old_state, 64), targets[64:])
    for key in ('gripRecall', 'gripFalsePositiveRate', 'parentMotionMSE'):
        if not np.allclose(original_prefix_fit[key], fit[key], atol=1e-6, rtol=1e-5):
            raise ValueError('Reconstructed original targets fail saved fit identity')
    episodes, dataset = load_dataset(args.dataset)
    if (file_hash(args.dataset/'manifest.json') != label_manifest['datasetManifestHash']
            or dataset['sourceHashes'] != label_manifest['originalPhysicsSourceHashes']):
        raise ValueError('Fresh-prefix data/physics differs from original')
    for column, s in enumerate(proof['selectedWindows']):
        a = episodes[s['episode']][0]
        if not np.array_equal(x[:, column].cpu().numpy(), a['observations'][s['start']:s['stop'], s['fly']]):
            raise ValueError('Training sensory history changed')
    print('REFRESHING_FULL_PHYSICAL_PREFIX '+json.dumps({'checkpoint': start, 'parameterHash': before}), flush=True)
    state = refresh_selected_prefixes(model, episodes, proof['selectedWindows'])
    assert before == model.checkpoint_hash() and model.fingerprint() == model.fixed_hash
    refreshed_fit = fitting_scores(frozen_predictions(model, x, state, 64), targets[64:])
    optimizer = torch.optim.Adam([{'params': [model.log_gains], 'lr': m['lr'], 'eps': 1e-14},
                                  {'params': [model.tonic], 'lr': m['tonic_lr'], 'eps': 1e-10}])
    restore_adam(optimizer, torch.load(optimizer_path, map_location='cpu', weights_only=True), start)
    args.out.mkdir(parents=True)
    with (args.out/'bank.npz').open('xb') as stream:
        np.savez_compressed(stream, observations=x.cpu().numpy(), targets=targets.cpu().numpy(), states=state.cpu().numpy())
    manifest = {'experiment': 'cooldown-microfit-continuation-v1', 'source': str(source), 'startUpdate': start,
                'finalUpdate': start+args.updates, 'parentParameterHash': before, 'fixedHash': model.fixed_hash,
                'interface': model.interface, 'sourceFileSHA256': source_hashes, 'selections': proof['selectedWindows'],
                'prefixCheckpointHash': before, 'fullHistoryReencoded': True, 'maximumPrefixAgeUpdates': 80,
                'originalMotionTargetParent': m['parentParameterHash'], 'targetHash': target_hash,
                'originalPrefixFit': original_prefix_fit, 'refreshedPrefixFit': refreshed_fit,
                'bankHash': file_hash(args.out/'bank.npz'), 'optimizerMomentsPreserved': True,
                'randomSamplerUsed': False, 'sameTrainingInputsAndTargets': True,
                'teacherAtInference': False, 'decoderTrained': False, 'dopamineLearning': False,
                'isServiceEvidence': False, 'canonicalOptimizer': 'Spark1 only', 'remoteExperienceUsed': False,
                'lr': m['lr'], 'tonic_lr': m['tonic_lr'], 'gradientFrames': 96, 'lossFrames': 32,
                'sourceHash': file_hash(Path(__file__)), 'sourceTrainingHashes': m['sourceHashes']}
    atomic_json(args.out/'manifest.json', manifest)
    initial_gains = model.log_gains.detach().cpu().numpy().copy()
    initial_tonic = model.tonic.detach().cpu().numpy().copy()
    last_saved = start
    def record(update, row):
        nonlocal last_saved
        save_model(args.out/f'candidate-{update}.npz', model)
        torch.save({'optimizer': optimizer.state_dict(), 'updates': update}, args.out/f'optimizer-{update}.pt')
        atomic_json(args.out/f'fit-{update}.json', row)
        atomic_json(args.out/'status.json', {'finished': False, **row})
        last_saved = update
    record(start, {'update': start, 'parameterHash': before, **refreshed_fit})
    try:
        history = train_segment(model, optimizer, x, targets, state, start, args.updates, 20, record)
        if any(file_hash(Path(p)) != h for p, h in source_hashes.items()):
            raise ValueError('Source files changed during continuation')
        result = {'startUpdate': start, 'updates': start+args.updates, 'history': history,
                  'finalParameterHash': model.checkpoint_hash(), 'audit': model.audit(initial_gains),
                  'changedNeurons': int(np.count_nonzero(model.tonic.detach().cpu().numpy() != initial_tonic)),
                  'sourceFilesUnchanged': True, 'originalTargetsPreserved': True, 'isServiceEvidence': False}
        atomic_json(args.out/'result.json', result)
        atomic_json(args.out/'status.json', {'finished': True, **history[-1]})
        print('CONTINUATION_FINISHED '+json.dumps({k:v for k,v in result.items() if k != 'history'}), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'type': type(error).__name__, 'error': str(error), 'lastSaved': last_saved})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'source', 'dataset', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoint', type=int, default=80)
    parser.add_argument('--updates', type=int, default=80)
    main(parser.parse_args())
