"""Teacher feasibility diagnostic. These products are NEVER brain evidence."""
import argparse
import json
from pathlib import Path
from .layout_recovery_protocol import atomic_json
from .layout_recovery_teacher import LocalTeacher, KINDS
from .layout_recovery_world import RecoveryWorld


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostics')
    worlds = [(kind, args.seed+j*1000+i, RecoveryWorld(args.seed+j*1000+i, kind=kind,
                                                     random_starts=args.random_starts))
              for j, kind in enumerate(KINDS) for i in range(args.cases)]
    teachers = [[LocalTeacher() for _ in range(4)] for _ in worlds]
    times = [[] for _ in worlds]
    for tick in range(args.seconds*20):
        for i, ((_, _, world), labels) in enumerate(zip(worlds, teachers)):
            motors = []
            for agent, teacher in zip(world.agents, labels):
                v = teacher.label(agent)
                motors.append({'speed': float(v[0]), 'turn': float(v[1]), 'interact': bool(v[2] > 1.)})
            before = world.deliveries
            world.advance(motors)
            times[i].extend([(tick+1)*.05]*(world.deliveries-before))
    trials = [{'kind': kind, 'seed': seed, 'products': world.deliveries,
               'returns': world.returns, 'deliveryTimes': t,
               'cargo': [a.cargo for a in world.agents], 'stock': [s['stock'] for s in world.stations]}
              for (kind, seed, world), t in zip(worlds, times)]
    report = {'teacherActions': True, 'brainWasUsed': False, 'isBrainEvidence': False,
              'seconds': args.seconds, 'trials': trials}
    atomic_json(args.out, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=8680000)
    parser.add_argument('--cases', type=int, default=2)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--random-starts', action='store_true')
    main(parser.parse_args())
