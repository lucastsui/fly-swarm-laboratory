"""Bounded, recurrent correction training on the fly's own physical states.

The teacher supplies training loss labels, optionally demonstration actions
in explicitly mixed training worlds. It is absent from frozen evaluation.
No new network, target input, routing controller or learned decoder is added.
Only the existing connectome's synaptic gains and neuron excitability change.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .layout_excitability import ExcitableConnectome
from .layout_training import KINDS, LABEL_VERSION, evaluate, local_label
from .layout_world import LayoutSwarmWorld
from .supervised_joint import raw_readout


def action_dicts(values):
    """The original fixed motor decoder's action convention."""
    return [{'speed': float(v[0]), 'turn': float(v[1]),
             'interact': bool(v[2] > .025)} for v in values]


def choose_training_actions(predicted, labels, teacher_mask):
    actions = action_dicts(predicted)
    for i in np.flatnonzero(teacher_mask):
        actions[i] = {'speed': float(labels[i, 0]), 'turn': float(labels[i, 1]),
                      'interact': bool(labels[i, 2] > 1.)}
    return actions


def correction_loss(predicted, labels):
    # Every head has an independent raw, pre-clamp corrective derivative.
    # Locomotion is not drowned out by the numerically larger interaction loss.
    by_head = (predicted - labels).square().mean(0)
    return (by_head * predicted.new_tensor([4., 2., .25])).sum(), by_head


def teacher_fraction(update, total, initial):
    # The final quarter is entirely learner-controlled, even during training.
    return initial * max(0., 1. - (update - 1) / max(1., .75 * total - 1))


def new_world(rng, index, kinds, curriculum):
    world = LayoutSwarmWorld(int(rng.integers(8000000, 8090000)),
                            kind=kinds[index % len(kinds)], stock=False)
    if curriculum:
        # Expose all loaded stages. These are training-only initial conditions;
        # product counts in such worlds are NOT full assembly-line evidence.
        for i, agent in enumerate(world.agents):
            agent.cargo = i
    return world


def save_candidate(path, model):
    if path.exists():
        raise FileExistsError('Never overwrite a saved experiment')
    np.savez_compressed(path, gains=model.log_gains.detach().cpu().numpy(),
                        tonic=model.tonic.detach().cpu().numpy(),
                        interface=np.asarray(model.interface))


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve previous experiments')
    if min(args.updates, args.worlds, args.gradient_frames, args.save_every) < 1:
        raise ValueError('Positive training bounds required')
    if args.rollout_frames < 0 or not 0 <= args.teacher_start <= 1:
        raise ValueError('Invalid rollout or demonstration fraction')
    if min(args.lr, args.tonic_lr, args.reset_seconds) <= 0:
        raise ValueError('Positive learning rates and reset interval required')
    kinds = tuple(args.kinds.split(','))
    if any(kind not in KINDS for kind in kinds):
        raise ValueError('Unknown layout family')
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.candidate, allow_pickle=False) as archive:
        gains, tonic = archive['gains'].copy(), archive['tonic'].copy()
        if str(archive['interface']) != 'annotated-color-cargo-v1':
            raise ValueError('This experiment requires the fixed annotated interface')
    model = ExcitableConnectome(args.root, gains, tonic,
                               annotated=True, surrogate=args.surrogate)
    initial_hash = model.checkpoint_hash()
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(parentHash=initial_hash, fixedInterfaceHash=model.fixed_hash,
                    sensoryInterface=model.interface, labelVersion=LABEL_VERSION,
                    device=torch.cuda.get_device_name(),
                    sourceHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    training='truncated recurrent supervised correction on physical trajectories',
                    dopamineLearning=False, externalDecisionNetwork=False,
                    decoderTrained=False, teacherAtEvaluation=False,
                    trainingProductsAreVerification=False)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    optimizer = torch.optim.Adam([
        {'params': [model.log_gains], 'lr': args.lr},
        {'params': [model.tonic], 'lr': args.tonic_lr}], eps=1e-10)
    worlds = [new_world(rng, i, kinds, args.curriculum and i % 2 == 1)
              for i in range(args.worlds)]
    state = None
    history = []
    began = time.perf_counter()
    retired_products = 0
    episode = 0
    actor_counts = np.zeros(4, np.int64)
    print('TRAINING_STARTED ' + json.dumps(manifest), flush=True)
    for update in range(1, args.updates + 1):
        if state is not None:
            state = state.detach()
        optimizer.zero_grad(set_to_none=True)
        weights = model.weights()
        beta = teacher_fraction(update, args.updates, args.teacher_start)
        # Keep a coherent control source for a whole short trajectory window.
        teacher_mask = rng.random(4 * args.worlds) < beta
        errors = []
        for frame in range(args.rollout_frames + args.gradient_frames):
            obs = np.concatenate([w.sensory() for w in worlds])
            targets = np.asarray([local_label(a) for w in worlds for a in w.agents])
            actor_counts += np.bincount([a.cargo for w in worlds for a in w.agents], minlength=4)
            with torch.set_grad_enabled(frame >= args.rollout_frames):
                action, state = model(torch.as_tensor(obs, device='cuda'), 4, state, weights)
                if frame >= args.rollout_frames:
                    _, heads = correction_loss(raw_readout(model, state),
                                                torch.as_tensor(targets, device='cuda'))
                    errors.append(heads)
            actions = choose_training_actions(action.detach().cpu().numpy(), targets, teacher_mask)
            for i, world in enumerate(worlds):
                world.advance(actions[4*i:4*i+4])
        heads = torch.stack(errors).mean(0)
        loss = (heads * heads.new_tensor([4., 2., .25])).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Missing or nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            model.log_gains.clamp_(-2., 2.)
            model.tonic.clamp_(-.1, .1)
        state = state.detach()
        # Episodic resets diversify TRAINING layouts only. Frozen service tests
        # retain every body, cargo, station state and recurrent state throughout.
        for i, world in enumerate(worlds):
            if world.steps * .05 >= args.reset_seconds:
                retired_products += world.deliveries
                episode += 1
                worlds[i] = new_world(rng, i + episode, kinds,
                                      args.curriculum and episode % 2 == 1)
                state[:, 4*i:4*i+4] = 0
        if update == 1 or update % 10 == 0:
            row = {'update': update, 'seconds': time.perf_counter() - began,
                   'loss': float(loss.detach()), 'mseByHead': heads.detach().cpu().tolist(),
                   'teacherFraction': beta, 'trainingActorFramesByCargo': actor_counts.tolist(),
                   'trainingProductsNotVerification': retired_products + sum(w.deliveries for w in worlds),
                   'trainingTransfers': sum(w.transfers for w in worlds)}
            history.append(row)
            print(json.dumps(row), flush=True)
        if update % args.save_every == 0 or update == args.updates:
            save_candidate(args.out / f'candidate-{update}.npz', model)
            (args.out / 'history.json').write_text(json.dumps(history, indent=2))
            print('CHECKPOINT ' + str(update) + ' ' + model.checkpoint_hash(), flush=True)
    audit = model.audit(gains)
    audit['fixedGraphSensoryAndMotorDecoderUnchanged'] = audit.pop('fixedGraphSensoryDecoderDynamicsUnchanged')
    result = {'updates': args.updates, 'checkpointHash': model.checkpoint_hash(),
              'audit': audit, 'changedNeurons': int(np.count_nonzero(model.tonic.detach().cpu().numpy() != tonic)),
              'seconds': time.perf_counter() - began, 'requiresFrozenVerification': True}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2))
    torch.save({'optimizer': optimizer.state_dict(), 'updates': args.updates,
                'rng': rng.bit_generator.state}, args.out / 'optimizer.pt')
    print('TRAINING_FINISHED ' + json.dumps(result), flush=True)
    if args.eval_seconds:
        model.eval().requires_grad_(False)
        report = evaluate(model, args.eval_seed, args.eval_cases, args.eval_seconds,
                          kinds=('compact', 'wide'))
        report['synapticGainsHash'] = report['checkpointHash']
        report['checkpointHash'] = model.checkpoint_hash()
        report['excitabilityHash'] = hashlib.sha256(model.tonic.detach().cpu().numpy().tobytes()).hexdigest()
        assert report['checkpointHash'] == result['checkpointHash']
        (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2))
        print('FROZEN_RESULT ' + json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=240)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--rollout-frames', type=int, default=24)
    parser.add_argument('--gradient-frames', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=80)
    parser.add_argument('--teacher-start', type=float, default=0.)
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--surrogate', action='store_true')
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--tonic-lr', type=float, default=.00002)
    parser.add_argument('--reset-seconds', type=float, default=90.)
    parser.add_argument('--seed', type=int, default=8200000)
    parser.add_argument('--kinds', default='compact,wide,rotated,permuted')
    parser.add_argument('--eval-seconds', type=int, default=240)
    parser.add_argument('--eval-seed', type=int, default=8100000)
    parser.add_argument('--eval-cases', type=int, default=4)
    main(parser.parse_args())
