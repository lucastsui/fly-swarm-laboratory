"""One canonical learner, local trajectories plus Spark2 correction replay."""
import argparse
import hashlib
import json
import secrets
import time
from pathlib import Path
import numpy as np
import torch
from .layout_closed_loop import action_dicts, correction_loss
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_protocol import Exchange, atomic_json, start_server
from .layout_recovery_teacher import LocalTeacher, LABEL_VERSION, training_world
from .layout_recovery_world import INTERFACE, PHYSICS
from .layout_recovery_curriculum import balanced_world, episode_limit, VERSION as CURRICULUM_VERSION
from .supervised_joint import raw_readout
from .layout_grip_calibration import CalibrationBank, cargo_contrast, calibration_progress, VERSION as CALIBRATION_VERSION
from .layout_grip_objective import threshold_grip_loss, matched_grip_ranking, check_margin, VERSION as GRIP_VERSION
from .layout_microfit import select_microfit_scenes, microfit_passes, discard_pending_warmup_replay
from .layout_microfit_resume import load_continuation, restore_optimizer


def motor_head_errors(prediction, labels, balanced_grip=False, grip_margin=None):
    error = prediction-labels
    heads = error.square().reshape(-1, 3).mean(0)
    if grip_margin is not None:
        return torch.stack((heads[0], heads[1], threshold_grip_loss(prediction[..., 2], labels[..., 2], grip_margin)))
    if not balanced_grip:
        return heads
    # Balance across the ENTIRE recurrent window, not each mostly-negative
    # frame. A lower average MSE must not reward never operating the grip.
    pred = prediction[..., 2].reshape(-1)
    target = labels[..., 2].reshape(-1)
    positive = (target > 1.).to(pred.dtype)
    negative = 1.-positive
    pc, nc = positive.sum(), negative.sum()
    pm, nm = (pc > 0).to(pred.dtype), (nc > 0).to(pred.dtype)
    positive_mse = ((pred-target).square()*positive).sum()/pc.clamp_min(1)
    negative_mse = ((pred-target).square()*negative).sum()/nc.clamp_min(1)
    balanced = (positive_mse+negative_mse)/(pm+nm).clamp_min(1)
    pred_gap = (pred*positive).sum()/pc.clamp_min(1)-(pred*negative).sum()/nc.clamp_min(1)
    target_gap = (target*positive).sum()/pc.clamp_min(1)-(target*negative).sum()/nc.clamp_min(1)
    # Contrast requires different responses to grip-on and grip-off contexts;
    # simply raising all interaction activity cannot satisfy it.
    grip = balanced + .5*pm*nm*(pred_gap-target_gap).square()
    return torch.stack((heads[0], heads[1], grip))


def recurrent_predictions(model, observations, initial_state, burn, weights, gradient_start=None):
    # Output selection and differentiation horizon are distinct. Default keeps
    # the old trajectory-replay behavior; full synthetic sequences may retain
    # gradients through their burn-in without adding any burn-in loss labels.
    gradient_start = burn if gradient_start is None else gradient_start
    if not 0 <= gradient_start <= burn < len(observations):
        raise ValueError('Invalid recurrent output/gradient horizon')
    state = initial_state.detach()
    predictions = []
    for frame in range(len(observations)):
        with torch.set_grad_enabled(frame >= gradient_start):
            _, state = model(observations[frame], 4, state, weights)
            if frame >= burn:
                predictions.append(raw_readout(model, state))
    return torch.stack(predictions)


def recurrent_loss(model, observations, labels, initial_state, burn, weights, balanced_grip=False, grip_margin=None):
    prediction = recurrent_predictions(model, observations, initial_state, burn, weights)
    return motor_head_errors(prediction, labels[burn:], balanced_grip, grip_margin)


def calibration_loss(model, observations, targets, burn, weights, grip_margin=None, ranking_weight=1., full_gradient=False):
    if not np.isfinite(ranking_weight) or ranking_weight <= 0:
        raise ValueError('Positive finite cargo-ranking weight required')
    initial = torch.zeros((model.n, observations.shape[1]), device=observations.device)
    prediction = recurrent_predictions(model, observations, initial, burn, weights,
                                       gradient_start=0 if full_gradient else burn)
    errors = motor_head_errors(prediction, targets, True, grip_margin)
    if grip_margin is None:
        contrast = .5*cargo_contrast(prediction, targets)
    else:
        contrast = matched_grip_ranking(prediction[..., 2].reshape(-1, 4),
                                       targets[..., 2].reshape(-1, 4), grip_margin)
    return torch.stack((errors[0], errors[1], errors[2]+ranking_weight*contrast))


def clip_brain_gradients(model, blockwise=False):
    if blockwise:
        # A large neuronal-bias gradient must not rescale every weak synaptic
        # gradient. Adam's epsilon makes that shared scaling consequential.
        return {name: float(torch.nn.utils.clip_grad_norm_([parameter], 1.))
                for name, parameter in model.named_parameters()}
    return {'global': float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))}


def optimize(model, optimizer, heads, head_weights=(4., 2., .25), blockwise=False, gradient_stats=None):
    loss = (heads*heads.new_tensor(head_weights)).sum()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Missing/nonfinite gradient')
    norms = clip_brain_gradients(model, blockwise)
    if gradient_stats is not None:
        gradient_stats.update(norms)
    optimizer.step()
    with torch.no_grad():
        model.log_gains.clamp_(-2., 2.)
        model.tonic.clamp_(-.1, .1)
    return float(loss.detach())


def main(args):
    if args.out.exists():
        raise FileExistsError('Never overwrite a run')
    if min(args.updates, args.worlds, args.gradient_frames, args.save_every) < 1 or args.burn < 1:
        raise ValueError('Positive experiment bounds required')
    if args.grip_margin is not None:
        check_margin(args.grip_margin)
    if not np.isfinite(args.synapse_eps) or args.synapse_eps <= 0:
        raise ValueError('A positive finite synaptic Adam epsilon is required')
    if not np.isfinite(args.ranking_weight) or args.ranking_weight <= 0:
        raise ValueError('Positive finite cargo-ranking weight required')
    if not 0 <= args.microfit_updates <= args.updates or (args.microfit_updates and not args.grip_calibration):
        raise ValueError('Microfit needs a calibration bank and a bounded nonnegative warmup')
    full_calibration_gradient = getattr(args, 'full_calibration_gradient', False)
    if full_calibration_gradient and not args.grip_calibration:
        raise ValueError('Full calibration gradient requires calibration data')
    args.out.mkdir(parents=True)
    (args.out/'experience').mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    make_world = balanced_world if args.balanced_curriculum else training_world
    layout_hash = None
    transition_practice = getattr(args, 'fixed_transition_practice', False)
    if transition_practice and not getattr(args, 'layout_file', None):
        raise ValueError('Transition practice requires an explicit fixed layout')
    if getattr(args, 'layout_file', None):
        from .fixed_layout_curriculum import load_layout, fixed_training_world, transition_training_world, TRANSITION_VERSION
        if args.balanced_curriculum or args.grip_calibration or getattr(args, 'resume_microfit_run', None):
            raise ValueError('Fixed layout cannot silently mix an arbitrary-layout curriculum or cache')
        positions, layout_hash = load_layout(args.layout_file)
        factory = transition_training_world if transition_practice else fixed_training_world
        make_world = lambda rng, index: factory(rng, index, positions, layout_hash)
        atomic_json(args.out/'layout.json', json.loads(args.layout_file.read_text()))
    model = load_model(args.root, args.candidate, surrogate=not args.exact_gradient, migrate=args.migrate)
    before = model.checkpoint_hash()
    initial_gains = model.log_gains.detach().cpu().numpy().copy()
    initial_tonic = model.tonic.detach().cpu().numpy().copy()
    continuation = load_continuation(args, model) if getattr(args, 'resume_microfit_run', None) else None
    start_update = continuation['start'] if continuation else 0
    calibration = microfit = None
    if args.grip_calibration:
        print(('RESTORING_ORIGINAL_CALIBRATION' if continuation else 'CACHING_PARENT_MOTION')
              + ': synthetic training contexts, not service evidence', flush=True)
        calibration = (continuation['bank'] if continuation else
                       CalibrationBank(model, 9320001, args.calibration_scenes, args.burn, args.gradient_frames))
        if model.checkpoint_hash() != before:
            raise AssertionError('Parent cache must not change parameters')
        calibration.save(args.out/'calibration-bank.npz')
        atomic_json(args.out/'calibration-scenes.json', calibration.metadata)
        print('PARENT_MOTION_CACHED '+json.dumps({'scenes': len(calibration.metadata),
              'parameterHash': calibration.parent_hash,
              'gripPositiveFraction': float((calibration.targets[..., 2] > 1).mean())}), flush=True)
        if args.microfit_updates:
            microfit = calibration.subset(select_microfit_scenes(calibration.metadata, calibration.targets))
            microfit.save(args.out/'microfit-bank.npz')
            atomic_json(args.out/'microfit-scenes.json', microfit.metadata)
    optimizer = torch.optim.Adam([{'params': [model.log_gains], 'lr': args.lr, 'eps': args.synapse_eps},
                                  {'params': [model.tonic], 'lr': args.tonic_lr}], eps=1e-10)
    resumed_rng = restore_optimizer(optimizer, model, continuation) if continuation else None
    exchange = Exchange(args.out, args.token_file.read_text().strip(), model.fixed_hash, layout_hash=layout_hash)
    save_model(args.out/f'candidate-{start_update}.npz', model)
    exchange.update = start_update
    exchange.publish(start_update, f'candidate-{start_update}.npz', before)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('token_file',)}
    manifest.update(runId=exchange.run_id, parentParameterHash=before, fixedHash=model.fixed_hash,
                    startUpdate=start_update, continuation=continuation['record'] if continuation else None,
                    calibrationParentParameterHash=calibration.parent_hash if calibration else None,
                    interface=INTERFACE, physics=PHYSICS, labelVersion=LABEL_VERSION,
                    teacherAtInference=False, trainingProductsAreVerification=False,
                    externalDecisionNetwork=False, decoderTrained=False, dopamineLearning=False,
                    learning=('supervised recurrent backpropagation; exact ReLU derivative' if args.exact_gradient else
                              'supervised recurrent backpropagation; surrogate derivative only in training'),
                    canonicalOptimizer='Spark1 only', remoteRole='versioned learner-only trajectory collection',
                    curriculum=CURRICULUM_VERSION if args.balanced_curriculum else 'index-coupled-v1',
                    gripLoss=GRIP_VERSION if args.grip_margin is not None else
                    ('class-balanced MSE plus half group contrast' if args.balanced_grip else 'ordinary MSE'),
                    calibration=CALIBRATION_VERSION if calibration else None,
                    calibrationTargets='parent speed/turn cached before any update; local-teacher grip' if calibration else None,
                    calibrationHeadWeights=([8., 4., 8.] if args.grip_margin is not None else [8., 4., 2.]) if calibration else None,
                    calibrationDataSeed=9320001 if calibration else None,
                    calibrationDifferentiatedFrames=(args.burn+args.gradient_frames if full_calibration_gradient
                                                      else args.gradient_frames) if calibration else None,
                    calibrationLossFrames=args.gradient_frames if calibration else None,
                    physicalReplayDifferentiatedFrames=args.gradient_frames,
                    trajectoryHeadWeights=[4., 2., 2.] if args.grip_margin is not None else [4., 2., .25],
                    gradientClipping='one norm per parameter block' if args.block_clip else 'shared global norm',
                    tonicAdamEpsilon=1e-10,
                    microfitGate=({'recallAtLeast': .9, 'falsePositiveAtMost': .1,
                                   'eachParentMotionMSEAtMost': .05, 'isServiceEvidence': False}
                                  if microfit else None),
                    device=torch.cuda.get_device_name(),
                    sourceHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest['sourceHashes'] = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                              for name in ('layout_recovery_train.py', 'layout_recovery_worker.py',
                                           'layout_recovery_curriculum.py', 'layout_recovery_teacher.py',
                                           'layout_recovery_world.py', 'layout_recovery_brain.py',
                                           'layout_recovery_protocol.py', 'layout_grip_calibration.py',
                                           'layout_grip_objective.py', 'layout_microfit.py', 'layout_microfit_resume.py')}
    if layout_hash:
        manifest.update(layoutHash=layout_hash, curriculum=TRANSITION_VERSION if transition_practice else 'fixed-layout-normal-and-loaded-v1',
                        optimizerContinuation=False, parentWeightsPreserved=True)
        manifest['sourceHashes']['fixed_layout_curriculum.py'] = hashlib.sha256(
            Path(__file__).with_name('fixed_layout_curriculum.py').read_bytes()).hexdigest()
    atomic_json(args.out/'manifest.json', manifest)
    if calibration is not None and args.calibration_metrics:
        fit = calibration_progress(model, calibration)
        atomic_json(args.out/f'calibration-fit-{start_update}.json', fit)
        print('CALIBRATION_FIT '+json.dumps({'update': start_update, **fit}), flush=True)
    if microfit is not None:
        fit = calibration_progress(model, microfit)
        atomic_json(args.out/f'microfit-{start_update}.json', fit)
        print('MICROFIT_FIT '+json.dumps({'update': start_update, **fit}), flush=True)
    from .deployment import COORDINATOR_HOST
    server = start_server(exchange, COORDINATOR_HOST, 8843, args.cert, args.key)
    worlds_meta = [make_world(rng, i) for i in range(args.worlds)]
    if resumed_rng is not None:
        # Pure warmup never advanced these worlds. Recreate their original
        # seeded state first, then restore the later training sampler state.
        rng.bit_generator.state = resumed_rng
    worlds = [w for w, _ in worlds_meta]
    teachers = [LocalTeacher() for _ in range(4*args.worlds)]
    state = torch.zeros((model.n, 4*args.worlds), device=model.tonic.device)
    history, replay = [], []
    began = time.perf_counter()
    episode = args.worlds
    last_saved = start_update
    retired_products = 0
    early_stop = None
    microfit_gate_passed = None
    print('CANONICAL_LEARNER_STARTED '+json.dumps(exchange.manifest()), flush=True)
    try:
        for update in range(start_update+1, args.updates+1):
            optimizer.zero_grad(set_to_none=True)
            weights = model.weights()
            warmup_turn = update <= args.microfit_updates
            calibration_turn = warmup_turn or (calibration is not None and update % 4 == 1)
            calibration_scenes = None
            head_weights = (4., 2., 2.) if args.grip_margin is not None else (4., 2., .25)
            packet = exchange.pop() if update % 2 == 0 and not warmup_turn else None
            # Half remote turns prefer fresh episodes; replay balances high-error
            # cases with uniform samples. Loss is sampling priority, not success.
            if packet is not None:
                replay.append({'packet': packet, 'priority': 1.})
                replay = replay[-24:]
            elif update % 2 == 0 and not warmup_turn and replay:
                valid = [r for r in replay if update-r['packet'][3]['version'] <= 80]
                if valid:
                    scores = np.asarray([r['priority'] for r in valid], float)
                    probability = .5/len(valid)+.5*scores/scores.sum()
                    packet = valid[int(rng.choice(len(valid), p=probability))]['packet']
            if calibration_turn:
                x, y, calibration_scenes = (microfit.all_examples(model.tonic.device) if warmup_turn
                                             else calibration.sample(rng, model.tonic.device))
                heads = calibration_loss(model, x, y, args.burn, weights, args.grip_margin, args.ranking_weight,
                                         full_gradient=full_calibration_gradient)
                head_weights = (8., 4., 8.) if args.grip_margin is not None else (8., 4., 2.)
                source = 'spark1-microfit-prerequisite' if warmup_turn else 'spark1-synthetic-grip-parent-motion'
                source_version, rollout_id = 0, None
            elif packet is not None:
                x, y, previous_state, metadata = packet
                if x.shape[0] != args.burn+args.gradient_frames:
                    raise ValueError('Worker and learner recurrent windows differ')
                heads = recurrent_loss(model, torch.as_tensor(x, device='cuda'),
                                       torch.as_tensor(y, device='cuda'),
                                       torch.as_tensor(previous_state, device='cuda'), args.burn, weights,
                                       args.balanced_grip, args.grip_margin)
                source = 'spark2-replay'
                source_version = metadata['version']
                rollout_id = metadata['id']
            else:
                state = state.detach()
                predictions, expected = [], []
                for frame in range(args.burn+args.gradient_frames):
                    obs = np.concatenate([w.sensory() for w in worlds])
                    labels = np.asarray([teacher.label(agent) for teacher, agent in
                                         zip(teachers, [a for w in worlds for a in w.agents])])
                    with torch.set_grad_enabled(frame >= args.burn):
                        actions, state = model(torch.as_tensor(obs, device='cuda'), 4, state, weights)
                        if frame >= args.burn:
                            predictions.append(raw_readout(model, state))
                            expected.append(torch.as_tensor(labels, device='cuda'))
                    motors = action_dicts(actions.detach().cpu().numpy())
                    for i, world in enumerate(worlds):
                        world.advance(motors[4*i:4*i+4])
                heads = motor_head_errors(torch.stack(predictions), torch.stack(expected), args.balanced_grip, args.grip_margin)
                state = state.detach()
                source, source_version, rollout_id = 'spark1-own-trajectory', update-1, None
                for i, world in enumerate(worlds):
                    limit = episode_limit(worlds_meta[i][1], args.reset_seconds) if args.balanced_curriculum else args.reset_seconds
                    if world.steps*.05 >= limit:
                        retired_products += world.deliveries
                        worlds[i], meta = make_world(rng, episode)
                        worlds_meta[i] = (worlds[i], meta)
                        episode += 1
                        teachers[4*i:4*i+4] = [LocalTeacher() for _ in range(4)]
                        state[:, 4*i:4*i+4] = 0
            gradient_norms = {}
            loss = optimize(model, optimizer, heads, head_weights, args.block_clip, gradient_norms)
            for item in replay:
                if item['packet'][3]['id'] == rollout_id:
                    item['priority'] = min(100., max(.05, loss))
            exchange.update = update
            row = {'update': update, 'seconds': time.perf_counter()-began, 'loss': loss,
                   'trainingPhase': 'microfit-prerequisite' if warmup_turn else 'mixed-service-training',
                   'objectiveByHead': heads.detach().cpu().tolist(), 'headWeights': head_weights, 'source': source,
                   'calibrationScenes': calibration_scenes,
                   'gradientNormsBeforeClipping': gradient_norms,
                   'sourceVersion': source_version, 'rolloutId': rollout_id,
                   'trainingProductsNotVerification': sum(w.deliveries for w in worlds),
                   'retiredTrainingProductsNotVerification': retired_products,
                   'localTrainingWorlds': [meta for _, meta in worlds_meta],
                   'trainingReturns': sum(w.returns for w in worlds),
                   'remoteReceived': exchange.received, 'remoteConsumed': exchange.consumed,
                   'gpuMemoryGB': torch.cuda.max_memory_allocated()/2**30}
            history.append(row)
            print(json.dumps(row), flush=True)
            atomic_json(args.out/'status.json', {**exchange.manifest(), 'lastUpdate': row})
            if update % args.save_every == 0 or update == args.updates or update == args.microfit_updates:
                filename = f'candidate-{update}.npz'
                save_model(args.out/filename, model)
                exchange.publish(update, filename, model.checkpoint_hash())
                torch.save({'optimizer': optimizer.state_dict(), 'updates': update,
                            'rng': rng.bit_generator.state}, args.out/f'optimizer-{update}.pt')
                atomic_json(args.out/'history.json', history)
                last_saved = update
                print('CHECKPOINT '+json.dumps(exchange.manifest()), flush=True)
                if calibration is not None and args.calibration_metrics:
                    fit = calibration_progress(model, calibration)
                    atomic_json(args.out/f'calibration-fit-{update}.json', fit)
                    print('CALIBRATION_FIT '+json.dumps({'update': update, **fit}), flush=True)
                if microfit is not None:
                    fitted = calibration_progress(model, microfit)
                    atomic_json(args.out/f'microfit-{update}.json', fitted)
                    print('MICROFIT_FIT '+json.dumps({'update': update, **fitted}), flush=True)
                    if update == args.microfit_updates:
                        microfit_gate_passed = microfit_passes(fitted)
                        atomic_json(args.out/'microfit-result.json', {'passed': microfit_gate_passed,
                                    'update': update, 'trainingFitOnly': True, 'fit': fitted})
                        if not microfit_gate_passed:
                            early_stop = 'microfit prerequisite failed; mixed training not started'
                            break
                        retired = discard_pending_warmup_replay(exchange)
                        atomic_json(args.out/'warmup-replay-retirement.json', retired)
                        replay.clear()
            # Drop autograd graphs before constructing the next window.
            del heads, weights
        result = {'checkpointHash': model.checkpoint_hash(), 'audit': model.audit(initial_gains),
                  'changedNeurons': int(np.count_nonzero(model.tonic.detach().cpu().numpy() != initial_tonic)),
                  'updates': history[-1]['update'], 'plannedUpdates': args.updates,
                  'startUpdate': start_update, 'updatesThisRun': history[-1]['update']-start_update,
                  'earlyStop': early_stop, 'microfitGatePassed': microfit_gate_passed,
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
        # Give the worker a bounded opportunity to observe completion.
        time.sleep(5)
        server.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'out', 'token-file', 'cert', 'key'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=240)
    parser.add_argument('--worlds', type=int, default=2)
    parser.add_argument('--burn', type=int, default=64)
    parser.add_argument('--gradient-frames', type=int, default=32)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--tonic-lr', type=float, default=.00001)
    parser.add_argument('--reset-seconds', type=float, default=90.)
    parser.add_argument('--seed', type=int, default=8600001)
    parser.add_argument('--migrate', action='store_true')
    parser.add_argument('--balanced-curriculum', action='store_true')
    parser.add_argument('--layout-file', type=Path, help='Lock all training worlds to this saved dashboard layout')
    parser.add_argument('--fixed-transition-practice', action='store_true',
                        help='Training-only pickup/departure/delivery starts; same senses and body')
    parser.add_argument('--balanced-grip', action='store_true')
    parser.add_argument('--grip-calibration', action='store_true')
    parser.add_argument('--calibration-scenes', type=int, default=64)
    parser.add_argument('--grip-margin', type=float, help='Opt-in threshold-aware training objective; motor threshold stays fixed')
    parser.add_argument('--synapse-eps', type=float, default=1e-10,
                        help='Adam denominator epsilon for existing synaptic gains; tonic epsilon stays 1e-10')
    parser.add_argument('--block-clip', action='store_true', help='Clip synaptic and neuronal gradients separately')
    parser.add_argument('--exact-gradient', action='store_true', help='Use the exact existing ReLU derivative; forward dynamics unchanged')
    parser.add_argument('--calibration-metrics', action='store_true', help='Log fixed training-subset fit separately from service evaluation')
    parser.add_argument('--ranking-weight', type=float, default=1.,
                        help='Training-only matched-cargo ranking weight; no inference or label change')
    parser.add_argument('--microfit-updates', type=int, default=0,
                        help='Optional four-scene prerequisite; stop if training-fit gate fails before mixed training')
    parser.add_argument('--full-calibration-gradient', action='store_true',
                        help='Differentiate synthetic burn-in as well as retained loss frames; physical replay unchanged')
    parser.add_argument('--resume-microfit-run', type=Path,
                        help='Extend a completed failed synthetic-only prerequisite; preserve original targets and Adam state')
    main(parser.parse_args())
