"""Bounded single-optimizer physical-demo imitation plus learner corrections.

Teacher data enters only this offline TRAINING path. Network packets remain
strictly learner-controlled, and frozen inference never imports this module.
Prefix caches are exact at the parent checkpoint and explicitly stale during
updates, with a hard 80-update limit. This is truncated supervised backprop,
not dopamine learning or evidence of autonomous service.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .layout_demonstration_cache import CACHE_SCHEMA
from .layout_recovery_demonstrations import CONTROL, file_hash, TRAIN_SEED_LOW, TRAIN_SEED_HIGH
from .layout_recovery_world import INTERFACE, PHYSICS, CHANNELS
from .layout_recovery_teacher import KINDS, LABEL_VERSION
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_protocol import Exchange, atomic_json, start_server
from .layout_recovery_train import recurrent_predictions, motor_head_errors, optimize
from .layout_recent_corrections import pop_recent_correction
from .layout_stratified_worker import validate_family_coverage
from .layout_demonstration_resume import prepare_continuation, restore_continuation_optimizer
from .layout_demonstration_update_scale import validate_scale, resolve_update_scale, scaled_learning_rates
from .layout_demonstration_focus import resolve_focus_dataset, EventFocusedDemonstrations

CACHE_MODEL_SOURCES = {'layout_recovery_brain.py', 'layout_excitability.py',
                       'layout_brain.py', 'supervised_steering.py'}


class DemonstrationCache:
    def __init__(self, folder, parameter_hash, fixed_hash, burn, frames, neurons=166700):
        folder = Path(folder)
        self.manifest = json.loads((folder/'manifest.json').read_text())
        m = self.manifest
        expected = {'schema': CACHE_SCHEMA, 'control': CONTROL, 'finished': True,
                    'isBrainEvidence': False, 'optimizerUsed': False, 'parametersUnchanged': True,
                    'brainControlsTeacherWorlds': False, 'parameterHash': parameter_hash,
                    'fixedHash': fixed_hash, 'interface': INTERFACE, 'physics': PHYSICS,
                    'burn': burn, 'gradientFrames': frames}
        if any(m.get(k) != value for k, value in expected.items()):
            raise ValueError('Prefix-cache provenance or parent mismatch')
        if set(m['sourceHashes']) != CACHE_MODEL_SOURCES:
            raise ValueError('Incomplete cache model provenance')
        for name, expected_hash in m['sourceHashes'].items():
            if Path(name).name != name or file_hash(Path(__file__).parent/name) != expected_hash:
                raise ValueError('Cache forward-model source mismatch')
        if file_hash(folder/'cache.npz') != m['cacheFileHash']:
            raise ValueError('Prefix cache byte hash mismatch')
        with np.load(folder/'cache.npz', allow_pickle=False) as values:
            if set(values.files) != {'observations', 'labels', 'states'}:
                raise ValueError('Wrong cache arrays')
            self.x, self.y, self.states = [values[k].copy() for k in ('observations', 'labels', 'states')]
        count = len(m['windows'])
        shapes = ((count, burn+frames, 4, CHANNELS), (count, burn+frames, 4, 3), (count, neurons, 4))
        for value, shape in zip((self.x, self.y, self.states), shapes):
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid numeric cache')
        self.groups = {kind: {} for kind in KINDS}
        for index, spec in enumerate(m['windows']):
            if (not TRAIN_SEED_LOW <= spec['seed'] < TRAIN_SEED_HIGH or spec['kind'] not in KINDS
                    or spec['stop']-spec['start'] != burn+frames or spec['lossStart']-spec['start'] != burn):
                raise ValueError('Invalid training window provenance')
            self.groups[spec['kind']].setdefault(spec['category'], []).append(index)
        if any(not group for group in self.groups.values()):
            raise ValueError('Missing demonstration family')

    def sample(self, rng, device):
        indexes = []
        # Equal family and event-category probability, not frame frequency.
        for kind in KINDS:
            groups = self.groups[kind]
            category = str(rng.choice(sorted(groups)))
            indexes.append(int(rng.choice(groups[category])))
        return (*self.batch(indexes, device),
                {'windowIndexes': indexes, 'windows': [self.manifest['windows'][i] for i in indexes]})

    def batch(self, indexes, device):
        x, y = [torch.as_tensor(np.concatenate([value[i] for i in indexes], axis=1), device=device)
                for value in (self.x, self.y)]
        initial = torch.as_tensor(np.concatenate([self.states[i] for i in indexes], axis=1), device=device)
        return x, y, initial

    def diagnostic_indexes(self):
        indexes = []
        for kind in KINDS:
            group = sorted(i for members in self.groups[kind].values() for i in members)
            indexes.extend(group[i] for i in np.linspace(0, len(group)-1, min(4, len(group))).astype(int))
        return indexes


def training_scores(prediction, target):
    with torch.no_grad():
        positive = target[..., 2] > 1.
        predicted = prediction[..., 2] > 1.
        pc, nc = int(positive.sum()), int((~positive).sum())
        return {'gripPositiveLabels': pc, 'gripNegativeLabels': nc,
                'gripRecall': float(predicted[positive].float().mean()) if pc else None,
                'gripFalsePositiveRate': float(predicted[~positive].float().mean()) if nc else None,
                'teacherMotionMSE': (prediction[..., :2]-target[..., :2]).square().mean((0, 1)).cpu().tolist(),
                'isServiceEvidence': False}


@torch.no_grad()
def demonstration_fit(model, bank, burn):
    # Fixed TRAINING subset only, never physical service or held-out evidence.
    indexes = bank.diagnostic_indexes()
    predictions, targets = [], []
    saved_mode = model.training
    try:
        model.eval()
        weights = model.weights()
        for offset in range(0, len(indexes), 4):
            x, y, state = bank.batch(indexes[offset:offset+4], model.tonic.device)
            part = []
            for frame in range(len(x)):
                _, state = model(x[frame], 4, state, weights)
                if frame >= burn:
                    from .supervised_joint import raw_readout
                    part.append(raw_readout(model, state))
            predictions.append(torch.stack(part))
            targets.append(y[burn:])
        scores = training_scores(torch.cat(predictions, dim=1), torch.cat(targets, dim=1))
        return {'checkpointHash': model.checkpoint_hash(), 'windowIndexes': indexes,
                'prefixCheckpointHash': bank.manifest['parameterHash'], 'trainingSubsetOnly': True, **scores}
    finally:
        model.train(saved_mode)


def validate_bounds(args):
    if getattr(args, 'revise_step_scale', None) is not None:
        validate_scale(args.revise_step_scale)
    if (getattr(args, 'resume_new_demonstration_dataset', None)
            and not getattr(args, 'resume_demonstration_run', None)):
        raise ValueError('A curriculum expansion requires a completed source run')
    if (getattr(args, 'resume_checkpoint_update', None) is not None
            and not getattr(args, 'resume_demonstration_run', None)):
        raise ValueError('Selecting a checkpoint requires a completed source run')
    if not 1 <= args.updates <= 80:
        raise ValueError('Frozen parent-prefix age is bounded at 80 updates; refresh before any extension')
    if min(args.burn, args.gradient_frames, args.save_every) < 1:
        raise ValueError('Positive window/checkpoint bounds required')
    if any(not np.isfinite(value) or value <= 0 for value in (args.lr, args.tonic_lr)):
        raise ValueError('Positive finite learning rates required')


def main(args):
    validate_bounds(args)
    if args.out.exists():
        raise FileExistsError('Preserve previous runs')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_model(args.root, args.candidate, surrogate=False)
    model.train().requires_grad_(True)
    before = model.checkpoint_hash()
    initial_gains = model.log_gains.detach().cpu().numpy().copy()
    initial_tonic = model.tonic.detach().cpu().numpy().copy()
    bank = DemonstrationCache(args.cache, before, model.fixed_hash, args.burn, args.gradient_frames, model.n)
    device = model.tonic.device
    optimizer = torch.optim.Adam([{'params': [model.log_gains], 'lr': args.lr, 'eps': 1e-14},
                                  {'params': [model.tonic], 'lr': args.tonic_lr, 'eps': 1e-10}])
    continuation = None
    if getattr(args, 'resume_demonstration_run', None):
        continuation = prepare_continuation(args, model, bank)
        rng.bit_generator.state = restore_continuation_optimizer(optimizer, continuation)
    step_scale, scale_record = resolve_update_scale(args, continuation)
    focus_dataset, focus_change = resolve_focus_dataset(args, continuation)
    demo_sampler = EventFocusedDemonstrations(bank, focus_dataset) if focus_dataset else bank
    start = continuation['start'] if continuation else 0
    final_update = start+args.updates
    args.out.mkdir(parents=True)
    (args.out/'experience').mkdir()
    exchange = Exchange(args.out, args.token_file.read_text().strip(), model.fixed_hash)
    exchange.update = start
    save_model(args.out/f'candidate-{start}.npz', model)
    exchange.publish(start, f'candidate-{start}.npz', before)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'token_file'}
    manifest.update(runId=exchange.run_id, parentParameterHash=before, fixedHash=model.fixed_hash,
                    interface=INTERFACE, physics=PHYSICS, labelVersion=LABEL_VERSION,
                    canonicalOptimizer='Spark1 only', teacherAtInference=False,
                    externalDecisionNetwork=False, decoderTrained=False, dopamineLearning=False,
                    learning='supervised recurrent backpropagation; exact ReLU derivative',
                    objective='4*teacher-speed-MSE + 2*teacher-turn-MSE + 2*class-balanced-grip-MSE-and-contrast',
                    plannedMix='3/4 event-balanced physical demonstrations; 1/4 fresh learner-only Spark2 correction',
                    correctionSampling=('newest queued packet at selection time' if getattr(args, 'recent_corrections', False)
                                        else 'FIFO oldest queued packet'),
                    fallback='demonstration when no eligible correction packet is queued',
                    startUpdate=start, finalUpdate=final_update,
                    continuation=continuation['record'] if continuation else None,
                    effectiveStepScale=step_scale, updateScaleExperiment=scale_record,
                    effectiveLearningRates=[group['lr']*step_scale for group in optimizer.param_groups],
                    eventFocusedDataset=str(focus_dataset) if focus_dataset else None,
                    demonstrationSamplingRevision=focus_change,
                    eventFocusedSampling=demo_sampler.record if focus_dataset else None,
                    prefixCheckpointHash=before, prefixParentUpdate=start, maximumPrefixAgeUpdates=80,
                    prefixSemantics='Exact full-history state at the parent, then current-weight burn-in; '
                                    'not an exact full-history state under the updated checkpoint',
                    trainingProductsAreVerification=False, physicalOverridesAtInference=False,
                    gradientClipping='one norm per parameter block, limit 1',
                    cacheManifestHash=file_hash(args.cache/'manifest.json'),
                    cacheFileHash=bank.manifest['cacheFileHash'],
                    datasetManifestHash=bank.manifest['datasetManifestHash'],
                    device=torch.cuda.get_device_name())
    names = ('layout_demonstration_train.py', 'layout_demonstration_cache.py', 'layout_recovery_train.py',
             'layout_recovery_worker.py', 'layout_recovery_protocol.py', 'layout_recovery_brain.py',
             'layout_recovery_world.py', 'layout_recovery_teacher.py', 'layout_excitability.py',
             'supervised_steering.py', 'layout_recent_corrections.py',
             'layout_recovery_curriculum.py', 'layout_stratified_worker.py', 'layout_demonstration_resume.py',
             'layout_demonstration_curriculum.py', 'layout_demonstration_update_scale.py',
             'layout_demonstration_focus.py')
    manifest['sourceHashes'] = {name: file_hash(Path(__file__).parent/name) for name in names}
    atomic_json(args.out/'manifest.json', manifest)
    initial_fit = demonstration_fit(model, bank, args.burn)
    atomic_json(args.out/f'demonstration-fit-{start}.json', initial_fit)
    print('DEMONSTRATION_FIT '+json.dumps({'update': start, **initial_fit}), flush=True)
    from .deployment import COORDINATOR_HOST
    server = start_server(exchange, COORDINATOR_HOST, 8843, args.cert, args.key)
    history, last_saved, demo_count, correction_count = [], start, 0, 0
    began = time.perf_counter()
    print('CANONICAL_DEMONSTRATION_LEARNER_STARTED '+json.dumps(exchange.manifest()), flush=True)
    try:
        for update in range(start+1, final_update+1):
            optimizer.zero_grad(set_to_none=True)
            weights = model.weights()
            packet = None
            if update % 4 == 0:
                packet = (pop_recent_correction(exchange) if getattr(args, 'recent_corrections', False)
                          else exchange.pop())
            if packet is None:
                x, y, state, detail = demo_sampler.sample(rng, device)
                source, source_version, rollout_id = 'frozen-prefix-physical-teacher-demonstration', start, None
                demo_count += 1
            else:
                obs, labels, initial, metadata = packet
                if (metadata.get('control') != 'learner-only' or metadata.get('teacherActions') is not False
                        or len(obs) != args.burn+args.gradient_frames):
                    raise ValueError('Correction must be a learner-only matching window')
                if getattr(args, 'require_stratified_corrections', False):
                    validate_family_coverage(metadata, actors=obs.shape[1])
                x, y, state = [torch.as_tensor(v, device=device) for v in (obs, labels, initial)]
                source, source_version, rollout_id = 'spark2-learner-only-correction', metadata['version'], metadata['id']
                detail = {'worlds': metadata['worlds']}
                correction_count += 1
            prediction = recurrent_predictions(model, x, state, args.burn, weights)
            heads = motor_head_errors(prediction, y[args.burn:], balanced_grip=True)
            scores = training_scores(prediction, y[args.burn:])
            norms = {}
            with scaled_learning_rates(optimizer, step_scale):
                loss = optimize(model, optimizer, heads, (4., 2., 2.), True, norms)
            exchange.update = update
            row = {'update': update, 'seconds': time.perf_counter()-began, 'loss': loss,
                   'source': source, 'sourceVersion': source_version, 'rolloutId': rollout_id,
                   'sourceDetail': detail, 'trainingScores': scores,
                   'gradientNormsBeforeClipping': norms, 'objectiveByHead': heads.detach().cpu().tolist(),
                   'effectiveStepScale': step_scale,
                   'demoUpdates': demo_count, 'correctionUpdates': correction_count,
                   'remoteReceived': exchange.received, 'remoteConsumed': exchange.consumed,
                   'gpuMemoryGB': torch.cuda.max_memory_allocated()/2**30}
            history.append(row)
            print(json.dumps(row), flush=True)
            if update % args.save_every == 0 or update == final_update:
                save_model(args.out/f'candidate-{update}.npz', model)
                exchange.publish(update, f'candidate-{update}.npz', model.checkpoint_hash())
                torch.save({'optimizer': optimizer.state_dict(), 'updates': update, 'rng': rng.bit_generator.state},
                           args.out/f'optimizer-{update}.pt')
                atomic_json(args.out/'history.json', history)
                last_saved = update
                print('CHECKPOINT '+json.dumps(exchange.manifest()), flush=True)
                fit = demonstration_fit(model, bank, args.burn)
                atomic_json(args.out/f'demonstration-fit-{update}.json', fit)
                print('DEMONSTRATION_FIT '+json.dumps({'update': update, **fit}), flush=True)
            atomic_json(args.out/'status.json', {**exchange.manifest(), 'lastUpdate': row})
            del prediction, heads, weights, x, y, state
        result = {'checkpointHash': model.checkpoint_hash(), 'audit': model.audit(initial_gains),
                  'changedNeurons': int(np.count_nonzero(model.tonic.detach().cpu().numpy() != initial_tonic)),
                  'updates': final_update, 'startUpdate': start, 'updatesThisRun': args.updates,
                  'demoUpdates': demo_count, 'correctionUpdates': correction_count,
                  'remoteReceived': exchange.received, 'remoteConsumed': exchange.consumed,
                  'requiresIndependentVerification': True, 'seconds': time.perf_counter()-began}
        atomic_json(args.out/'result.json', result)
        print('TRAINING_FINISHED '+json.dumps(result), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'type': type(error).__name__, 'error': str(error),
                                            'lastSavedUpdate': last_saved})
        raise
    finally:
        exchange.finished = True
        atomic_json(args.out/'status.json', exchange.manifest())
        time.sleep(5)
        server.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'cache', 'out', 'token-file', 'cert', 'key'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=80)
    parser.add_argument('--burn', type=int, default=64)
    parser.add_argument('--gradient-frames', type=int, default=32)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--tonic-lr', type=float, default=.000001)
    parser.add_argument('--seed', type=int, default=9380001)
    parser.add_argument('--recent-corrections', action='store_true',
                        help='Prefer newest queued learner-only experience; preserve raw files and staleness checks')
    parser.add_argument('--require-stratified-corrections', action='store_true',
                        help='Reject correction packets missing the four-family training curriculum')
    parser.add_argument('--resume-demonstration-run', type=Path,
                        help='Resume a finished run with preserved Adam/RNG and a newly built parent-prefix cache')
    parser.add_argument('--resume-checkpoint-update', type=int,
                        help='Select an attested saved update within the finished source run; default is its final update')
    parser.add_argument('--resume-new-demonstration-dataset', type=Path,
                        help='Explicit audited curriculum expansion: more physical TRAINING episodes per family; '
                             'fresh cache must match every new dataset window (event-examples=2)')
    parser.add_argument('--revise-step-scale', type=float,
                        help='Explicit bounded training experiment (0.1 to 4): scale both Adam learning rates '
                             'only during each update, preserve saved moments/base controls; default inherits source')
    parser.add_argument('--event-focused-demonstrations', type=Path,
                        help='Explicit sampling revision: select the physical event actor from four windows '
                             'per family; unchanged 16-column batch, observations, labels and cached prefixes; '
                             'dataset must match the prefix cache exactly; default inherits source')
    main(parser.parse_args())
