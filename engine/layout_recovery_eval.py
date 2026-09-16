"""Independent frozen evaluation: no teacher import or teacher action path."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from .evaluate_supervised import FrozenPolicy
from .layout_excitability import ExcitableConnectome
from .layout_recovery_brain import load_model
from .layout_recovery_world import RecoveryWorld, PHYSICS
from .layout_world import random_layout
from .layout_interaction_trace import before_interaction, interaction_events


def wilson(successes, count):
    z = 1.96
    p = successes/count
    center = (p+z*z/(2*count))/(1+z*z/count)
    radius = z*((p*(1-p)/count+z*z/(4*count*count))**.5)/(1+z*z/count)
    return [max(0., center-radius), min(1., center+radius)]


def service_metrics(delivery_times, seconds, move_at=0):
    """Report post-move service separately; earlier output cannot pass that gate."""
    if seconds <= 0 or not 0 <= move_at < seconds:
        raise ValueError('Invalid service horizon')
    result = {'sustainedSuccess': len(delivery_times) >= 3 and
              any(t > seconds/2 for t in delivery_times)}
    if move_at:
        count = sum(t > move_at for t in delivery_times)
        result.update(postMoveProducts=count, postMoveSuccess=count >= 3)
    return result


def evaluate(model, seed, cases, seconds, kinds, random_starts=False, move_at=0, original_senses=False,
             record_interactions=False, positions=None):
    if positions is not None and move_at:
        raise ValueError('Fixed-layout tests cannot move boxes')
    model.eval().requires_grad_(False)
    before = model.checkpoint_hash()
    worlds = [(kind, seed+1000*j+i, RecoveryWorld(seed+1000*j+i, kind=kind, random_starts=random_starts, positions=positions))
              for j, kind in enumerate(kinds) for i in range(cases)]
    initial_positions = [[[s['x'], s['y']] for s in w.stations] for _, _, w in worlds]
    initial_avatars = [[[a.x, a.y, a.heading] for a in w.agents] for _, _, w in worlds]
    policy = FrozenPolicy(model, len(worlds)*4)
    times = [[] for _ in worlds]
    traces = [[] for _ in worlds]
    events = [[] for _ in worlds]
    event_counts = [dict.fromkeys(('pickup', 'return', 'transfer', 'delivery', 'rejected'), 0) for _ in worlds]
    began = time.perf_counter()
    for tick in range(round(seconds/.05)):
        if move_at and tick == round(move_at/.05):
            for kind, world_seed, world in worlds:
                world.set_layout(random_layout(world_seed+200000, kind))
        observations = np.concatenate([w.sensory() for _, _, w in worlds])
        if original_senses:
            observations = observations[:, :129]
        actions = policy.act(observations)
        for i, (_, _, world) in enumerate(worlds):
            old = world.deliveries
            previous = before_interaction(world) if record_interactions else None
            world.advance(actions[4*i:4*i+4])
            if record_interactions:
                for event in interaction_events(world, previous, tick+1):
                    event_counts[i][event['event']] += 1
                    if len(events[i]) < 4000:
                        events[i].append(event)
            times[i].extend([(tick+1)*.05]*(world.deliveries-old))
            if (tick+1) % 100 == 0:
                traces[i].append({'time': (tick+1)*.05, 'products': world.deliveries,
                                  'returns': world.returns,
                                  'stocks': [s['stock'] for s in world.stations],
                                  'returned': [s['returned'] for s in world.stations],
                                  'timers': [s['timer'] for s in world.stations],
                                  'flies': [[a.x, a.y, a.heading, a.cargo, a.speed, a.turn, a.contact]
                                            for a in world.agents]})
        if (tick+1) % 1200 == 0:
            print(json.dumps({'evalSeconds': (tick+1)*.05, 'wallSeconds': time.perf_counter()-began,
                              'products': sum(w.deliveries for _, _, w in worlds)}), flush=True)
    trials = [{'kind': kind, 'seed': s, 'initialPositions': p, 'initialAvatars': av,
               'finalPositions': [[b['x'], b['y']] for b in w.stations],
               'products': w.deliveries, 'pickups': w.pickups, 'transfers': w.transfers,
               'returns': w.returns, 'bumps': w.bump_events, 'deliveryTimes': t,
               'agentContributions': [{'pickups': a.pickups, 'transfers': a.transfers, 'products': a.deliveries,
                                       'returns': a.returns} for a in w.agents],
               **service_metrics(t, seconds, move_at), 'trace': trace}
              for (kind, s, w), p, av, t, trace in zip(worlds, initial_positions, initial_avatars, times, traces)]
    if record_interactions:
        for trial, records, counts in zip(trials, events, event_counts):
            trial.update(interactionEvents=records, interactionCounts=counts,
                         interactionEventsTruncated=sum(counts.values()) > len(records))
    assert model.checkpoint_hash() == before
    summary = {}
    for kind in kinds:
        subset = [t for t in trials if t['kind'] == kind]
        successes = sum(t['sustainedSuccess'] for t in subset)
        summary[kind] = {'cases': cases, 'products': sum(t['products'] for t in subset),
                         'worldsWithProduct': sum(t['products'] > 0 for t in subset),
                         'worldsWithThree': sum(t['products'] >= 3 for t in subset),
                         'sustainedSuccesses': successes, 'wilson95': wilson(successes, cases)}
        if move_at:
            moved_successes = sum(t['postMoveSuccess'] for t in subset)
            summary[kind].update(postMoveProducts=sum(t['postMoveProducts'] for t in subset),
                                 postMoveSuccesses=moved_successes,
                                 postMoveWilson95=wilson(moved_successes, cases))
    return {'checkpointHash': before, 'fixedHash': model.fixed_hash, 'physics': PHYSICS,
            'interface': model.interface, 'teacher': False, 'learning': False, 'noise': False,
            'injectedCargo': False, 'deliveryResets': False, 'randomStarts': random_starts,
            'moveAtSeconds': move_at, 'seconds': seconds, 'batchActors': 4*len(worlds),
            'interactionEventRecording': record_interactions,
            'device': torch.cuda.get_device_name(), 'summary': summary, 'trials': trials}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve previous evaluation')
    if args.cases < 1 or args.seconds <= 0 or not 0 <= args.move_at < args.seconds:
        raise ValueError('Invalid evaluation bounds')
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    if args.original_senses:
        with np.load(args.candidate, allow_pickle=False) as data:
            if str(data['interface']) != 'annotated-color-cargo-v1':
                raise ValueError('Wrong baseline interface')
            model = ExcitableConnectome(args.root, data['gains'].copy(), data['tonic'].copy(), annotated=True)
    else:
        model = load_model(args.root, args.candidate, migrate=args.migrate)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    positions = None
    if getattr(args, 'layout_file', None):
        from .fixed_layout_curriculum import load_layout
        positions, layout_hash = load_layout(args.layout_file)
        if args.move_at:
            raise ValueError('Fixed-layout test cannot move boxes')
        args.kinds = 'fixed-current'
        manifest.update(kinds=args.kinds, layoutHash=layout_hash, fixedPositions=positions.tolist())
    manifest.update(sourceHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    checkpointHash=model.checkpoint_hash(), fixedHash=model.fixed_hash)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    report = evaluate(model, args.seed, args.cases, args.seconds, tuple(args.kinds.split(',')),
                      args.random_starts, args.move_at, args.original_senses, args.interaction_events, positions)
    if positions is not None:
        report.update(layoutHash=layout_hash, fixedPositions=positions.tolist(), fixedLayoutOnly=True)
    (args.out/'evaluation.json').write_text(json.dumps(report, indent=2))
    print('FROZEN_RESULT '+json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=8700000)
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--kinds', default='compact,wide,rotated,permuted')
    parser.add_argument('--random-starts', action='store_true')
    parser.add_argument('--move-at', type=int, default=0)
    parser.add_argument('--original-senses', action='store_true')
    parser.add_argument('--migrate', action='store_true')
    parser.add_argument('--interaction-events', action='store_true')
    parser.add_argument('--layout-file', type=Path)
    main(parser.parse_args())
