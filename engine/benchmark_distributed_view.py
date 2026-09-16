"""Bounded three-GPU inference comparison over authenticated SSH stdio.

No network listener, training, or persistent remote service is installed.
The laptop owns the unchanged four-fly collision world; each Spark owns one
independent neural state. All actions are joined before each physics step.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import getpass
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .evaluate_supervised import FrozenPolicy
from .fast_frozen_policy import FastFrozenPolicy, SplitFrozenPolicy
from .layout_recovery_brain import load_model
from .layout_recovery_world import RecoveryWorld


def worker(args):
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).eval().requires_grad_(False)
    policy = FastFrozenPolicy(model, 1, cuda_graph=True, index32=True)
    print(json.dumps({'ready': True, 'hash': model.checkpoint_hash(),
                      'device': torch.cuda.get_device_name()}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request['op'] == 'quit':
            break
        if request['op'] == 'reset':
            policy.reset()
            response = {'reset': True}
        elif request['op'] == 'state':
            array = policy.state.cpu().numpy().astype('<f4')
            response = {'state': base64.b64encode(array.tobytes()).decode()}
        else:
            response = {'actions': policy.act(request['observations'])}
        print(json.dumps(response), flush=True)


class Remote:
    def __init__(self, client, command, expected):
        self.stdin, self.stdout, self.stderr = client.exec_command(command, timeout=120)
        hello = json.loads(self.stdout.readline())
        if hello['hash'] != expected:
            raise ValueError('Remote checkpoint mismatch')
        self.device = hello['device']

    def request(self, request):
        self.stdin.write(json.dumps(request) + '\n')
        self.stdin.flush()
        return json.loads(self.stdout.readline())

    def act(self, observation):
        return self.request({'op': 'act', 'observations': [observation.tolist()]})['actions'][0]


def main(args):
    import paramiko
    import shlex
    if args.deployment is None:
        raise ValueError('--deployment JSON with known_hosts and two worker configurations is required')
    deployment = json.loads(args.deployment.read_text())
    workers = deployment['workers']
    if len(workers) != 2:
        raise ValueError('Exactly two explicit workers required')
    if args.out.exists():
        raise FileExistsError('Preserve previous benchmark')
    clients = []
    remotes = []
    try:
        known = Path(deployment['known_hosts'])
        def connect(host, user, password, sock=None):
            client = paramiko.SSHClient()
            client.load_host_keys(str(known))
            client.connect(host, username=user, password=password, sock=sock,
                           allow_agent=False, look_for_keys=False, timeout=20)
            clients.append(client)
            return client
        first = connect(workers[0]['host'], workers[0]['user'], getpass.getpass('Spark password: '))
        tunnel = first.get_transport().open_channel('direct-tcpip', (workers[1]['host'], 22), ('127.0.0.1', 0))
        second = connect(workers[1]['host'], workers[1]['user'], getpass.getpass('Spark2 password: '), tunnel)
        torch.set_num_threads(4)
        model = load_model(args.root, args.candidate).eval().requires_grad_(False)
        identity = model.checkpoint_hash()
        roots = [shlex.quote(w['root']) for w in workers]
        runtimes = [shlex.quote(w['python']) for w in workers]
        candidates = [shlex.quote(w['candidate']) for w in workers]
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(Remote, client,
                f'cd {root} && {runtime} -u -m engine.benchmark_distributed_view --worker --root data --candidate {candidate}', identity)
                for client, root, runtime, candidate in zip(clients, roots, runtimes, candidates)]
            remotes = [job.result() for job in jobs]
            local = SplitFrozenPolicy(model, 2)
            original = FrozenPolicy(model, 4)
            all_local = SplitFrozenPolicy(model, 4)
            def distributed(obs):
                jobs = [pool.submit(remote.act, obs[i+2]) for i, remote in enumerate(remotes)]
                return local.act(obs[:2]) + [job.result() for job in jobs]
            def reset():
                local.reset()
                for remote in remotes:
                    remote.request({'op': 'reset'})
            world = RecoveryWorld(5500000, flies=4, continuous=True, kind='wide')
            maximum = 0.
            for frame in range(80):
                obs = world.sensory()
                expected = original.act(obs)
                actual = distributed(obs)
                np.testing.assert_allclose([[a['speed'], a['turn']] for a in actual],
                                           [[a['speed'], a['turn']] for a in expected], atol=1e-5, rtol=1e-4)
                assert [a['interact'] for a in actual] == [a['interact'] for a in expected]
                if frame % 20 == 19:
                    states = [local.state.cpu().numpy()]
                    for remote in remotes:
                        raw = remote.request({'op': 'state'})['state']
                        states.append(np.frombuffer(base64.b64decode(raw), dtype='<f4').reshape(model.n, 1))
                    state = np.concatenate(states, axis=1)
                    reference = original.state.cpu().numpy()
                    maximum = max(maximum, float(np.max(np.abs(state-reference))))
                    np.testing.assert_allclose(state, reference, atol=1e-6, rtol=1e-4)
                world.advance(expected)
            print('THREE_GPU_PARITY_PASSED', flush=True)
            results = {'laptop-per-fly': [], 'three-gpu': []}
            for repeat in range(3):
                for name in (list(results) if repeat % 2 == 0 else list(reversed(results))):
                    reset()
                    all_local.reset()
                    world = RecoveryWorld(5500000, flies=4, continuous=True, kind='wide')
                    act = distributed if name == 'three-gpu' else all_local.act
                    for _ in range(10):
                        world.advance(act(world.sensory()))
                    start = time.perf_counter()
                    for _ in range(100):
                        world.advance(act(world.sensory()))
                    speed = 5/(time.perf_counter()-start)
                    results[name].append(speed)
                    print(json.dumps({'repeat': repeat, 'configuration': name, 'simulationSpeed': speed}), flush=True)
            assert model.checkpoint_hash() == identity
            report = {'checkpointHash': identity, 'weightsUnchanged': True, 'training': False,
                      'serviceEvidence': False, 'factoryFlies': 4, 'partition': [2, 1, 1],
                      'devices': [torch.cuda.get_device_name()] + [r.device for r in remotes],
                      'parityFrames': 80, 'fullStateChecks': 4, 'maxStateError': maximum,
                      'interactionDecisionsEqual': True, 'neuralSubsteps': 4, 'physicsStepSeconds': .05,
                      'results': {name: {'simulationSpeeds': speeds, 'medianSimulationSpeed': float(np.median(speeds))}
                                  for name, speeds in results.items()}}
            args.out.write_text(json.dumps(report, indent=2))
            print('DISTRIBUTED_BENCHMARK_DONE ' + json.dumps(report), flush=True)
    finally:
        for remote in remotes:
            try:
                remote.stdin.write('{"op":"quit"}\n')
                remote.stdin.flush()
            except Exception:
                pass
        for client in reversed(clients):
            client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    for name in ('root', 'candidate'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--deployment', type=Path, help='Private JSON: known_hosts and two workers (host, user, root, python, candidate)')
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        main(args)
