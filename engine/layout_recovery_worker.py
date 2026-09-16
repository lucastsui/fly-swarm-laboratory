"""Spark2: experience only, no optimizer, gradients or independent training."""
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from .layout_closed_loop import action_dicts
from .layout_recovery_brain import load_model
from .layout_recovery_protocol import Client, atomic_json, pack_rollout
from .layout_recovery_teacher import LocalTeacher, LABEL_VERSION
from .layout_recovery_world import INTERFACE, PHYSICS
from .layout_recovery_curriculum import episode_limit
from .layout_stratified_worker import worker_world


def main(args):
    stratified = getattr(args, 'stratified_families', False)
    if stratified and (not args.balanced_curriculum or args.worlds != 4):
        raise ValueError('Stratified corrections require exactly four balanced worlds')
    if args.out.exists():
        raise FileExistsError('Preserve prior worker evidence')
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)
    layout_hash = positions = None
    transition_practice = getattr(args, 'fixed_transition_practice', False)
    if transition_practice and not getattr(args, 'layout_file', None):
        raise ValueError('Transition practice requires an explicit fixed layout')
    if getattr(args, 'layout_file', None):
        from .fixed_layout_curriculum import load_layout, fixed_training_world, transition_training_world
        if args.balanced_curriculum or stratified:
            raise ValueError('Fixed layout cannot silently mix layout families')
        positions, layout_hash = load_layout(args.layout_file)
        atomic_json(args.out/'layout.json', json.loads(args.layout_file.read_text()))
    def make_world(rng, index, slot):
        if layout_hash:
            factory = transition_training_world if transition_practice else fixed_training_world
            return factory(rng, index, positions, layout_hash)
        return worker_world(rng, index, slot, balanced=args.balanced_curriculum, stratified=stratified)
    client = Client(args.coordinator, args.token_file, args.cert)
    model = weights = None
    version = -1
    run_id = None
    worlds_meta = [make_world(rng, i+100, i) for i in range(args.worlds)]
    worlds = [w for w, _ in worlds_meta]
    teachers = [LocalTeacher() for _ in range(4*args.worlds)]
    state = None
    episode = 100+args.worlds
    began = time.monotonic()
    last_contact = began
    accepted = 0
    print('EXPERIENCE_WORKER_STARTED: no optimizer; waiting for canonical learner', flush=True)
    for window in range(args.windows):
        if time.monotonic()-began > args.max_seconds:
            break
        while True:
            try:
                manifest = client.manifest()
                last_contact = time.monotonic()
                break
            except Exception as error:
                if time.monotonic()-last_contact > args.contact_timeout:
                    atomic_json(args.out/'finished.json', {'reason': 'coordinator unavailable',
                                                          'accepted': accepted, 'error': str(error)})
                    return
                time.sleep(5)
        if manifest['finished']:
            break
        if manifest.get('layoutHash') != layout_hash:
            raise ValueError('Worker and canonical learner have different layout tasks')
        if manifest['queued'] >= 22:
            time.sleep(2)
            continue
        if run_id is not None and manifest['runId'] != run_id:
            raise ValueError('Coordinator experiment changed; do not mix experience')
        run_id = manifest['runId']
        if manifest['version'] != version:
            blob = client.request('/checkpoint/'+str(manifest['version']))
            if hashlib.sha256(blob).hexdigest() != manifest['checkpoint']['sha256']:
                raise ValueError('Checkpoint transfer hash mismatch')
            path = args.out/f"canonical-{manifest['version']}.npz"
            if path.exists():
                raise FileExistsError('Unexpected reused checkpoint version')
            path.write_bytes(blob)
            if model is None:
                model = load_model(args.root, path)
                model.eval().requires_grad_(False)
            else:
                with np.load(path, allow_pickle=False) as data:
                    if str(data['interface']) != model.interface or str(data['fixed_hash']) != model.fixed_hash:
                        raise ValueError('Changed fixed interface')
                    with torch.no_grad():
                        model.log_gains.copy_(torch.as_tensor(data['gains'], device='cuda'))
                        model.tonic.copy_(torch.as_tensor(data['tonic'], device='cuda'))
            if model.checkpoint_hash() != manifest['checkpoint']['parameterHash'] or model.fixed_hash != manifest['fixedHash']:
                raise ValueError('Canonical checkpoint identity mismatch')
            version = manifest['version']
            with torch.no_grad():
                weights = model.weights()
            # Explicit TRAINING-only neural resynchronization. Physical worlds
            # survive. Burn-in uses current sensory trajectories before labels
            # contribute to a recurrent gradient on the learner.
            state = torch.zeros((model.n, 4*args.worlds), device='cuda')
            print('CANONICAL_SYNC '+json.dumps({'version': version, 'hash': model.checkpoint_hash()}), flush=True)
        initial_state = state.cpu().numpy().copy()
        observations, labels = [], []
        for frame in range(args.burn+args.gradient_frames):
            obs = np.concatenate([w.sensory() for w in worlds])
            target = np.asarray([teacher.label(agent) for teacher, agent in
                                 zip(teachers, [a for w in worlds for a in w.agents])])
            observations.append(obs)
            labels.append(target)
            with torch.no_grad():
                action, state = model(torch.as_tensor(obs, device='cuda'), 4, state, weights)
            motors = action_dicts(action.cpu().numpy())
            for i, world in enumerate(worlds):
                world.advance(motors[4*i:4*i+4])
        metadata = {'id': uuid.uuid4().hex, 'runId': run_id, 'worker': 'spark2',
                    'version': version, 'parameterHash': manifest['checkpoint']['parameterHash'],
                    'fixedHash': model.fixed_hash, 'interface': INTERFACE, 'physics': PHYSICS,
                    'control': 'learner-only', 'labelVersion': LABEL_VERSION,
                    'worlds': [m for _, m in worlds_meta], 'burn': args.burn,
                    'teacherActions': False, 'productsAreVerification': False}
        metadata['layoutHash'] = layout_hash
        blob = pack_rollout(np.asarray(observations), np.asarray(labels), initial_state, metadata)
        # Retries use the identical id/payload, so acknowledgements cannot
        # double-count experience after an uncertain connection interruption.
        while True:
            try:
                result = client.upload(blob)
                last_contact = time.monotonic()
                if result == 'full':
                    time.sleep(3)
                    continue
                break
            except Exception as error:
                if time.monotonic()-last_contact > args.contact_timeout:
                    atomic_json(args.out/'finished.json', {'reason': 'upload connection lost',
                                                          'accepted': accepted, 'error': str(error)})
                    return
                time.sleep(5)
        accepted += int(result == 'accepted')
        status = {'window': window, 'version': version, 'result': result, 'accepted': accepted,
                  'productsNotVerification': sum(w.deliveries for w in worlds),
                  'returns': sum(w.returns for w in worlds), 'seconds': time.monotonic()-began}
        atomic_json(args.out/'status.json', status)
        print(json.dumps(status), flush=True)
        if result == 'finished':
            break
        for i, world in enumerate(worlds):
            limit = episode_limit(worlds_meta[i][1], args.reset_seconds) if args.balanced_curriculum else args.reset_seconds
            if world.steps*.05 >= limit:
                worlds[i], meta = make_world(rng, episode, i)
                worlds_meta[i] = (worlds[i], meta)
                episode += 1
                teachers[4*i:4*i+4] = [LocalTeacher() for _ in range(4)]
                state[:, 4*i:4*i+4] = 0
    atomic_json(args.out/'finished.json', {'reason': 'bounded worker completed', 'accepted': accepted,
                                          'seconds': time.monotonic()-began})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'out', 'token-file', 'cert'):
        parser.add_argument('--'+name, type=Path, required=True)
    from .deployment import COORDINATOR_URL
    parser.add_argument('--coordinator', default=COORDINATOR_URL)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--burn', type=int, default=64)
    parser.add_argument('--gradient-frames', type=int, default=32)
    parser.add_argument('--reset-seconds', type=float, default=90.)
    parser.add_argument('--seed', type=int, default=8600101)
    parser.add_argument('--windows', type=int, default=4000)
    parser.add_argument('--max-seconds', type=int, default=21600)
    parser.add_argument('--contact-timeout', type=int, default=180)
    parser.add_argument('--balanced-curriculum', action='store_true')
    parser.add_argument('--layout-file', type=Path, help='Collect only on the saved dashboard layout')
    parser.add_argument('--fixed-transition-practice', action='store_true',
                        help='Training-only pickup/departure/delivery starts')
    parser.add_argument('--stratified-families', action='store_true',
                        help='One persistent worker slot per layout family; scenarios remain independently sampled')
    main(parser.parse_args())
