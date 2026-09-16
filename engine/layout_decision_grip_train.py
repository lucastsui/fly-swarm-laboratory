"""Bounded single-optimizer experiment on original physical decision frames.

Branch from retained C60 Adam/RNG; use a freshly replayed 16-world learner
dataset. The only objective change from V24 is supervising grip at its actual
event/request frame instead of averaging all nearby labels. Parent motion
remains a soft anchor. No inference, body, graph or decoder changes.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_correction_curriculum_train import source_run, restore_state
from .layout_correction_prefix import load_histories
from .layout_operation_sampling import OperationCache
from .layout_operation_focus import VERSION, focused_heads, focus_fit
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_train import recurrent_predictions, optimize
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def train(model, bank, optimizer, rng, updates, record):
    if type(updates) is not int or not 1 <= updates <= 40:
        raise ValueError('Hard40-update prefix age bound required')
    began = time.perf_counter(); history = []
    model.train().requires_grad_(True)
    for update in range(61, 61+updates):
        optimizer.zero_grad(set_to_none=True)
        x, y, state, motion, selection = bank.sample(rng, model.tonic.device)
        if x.shape != (96,16,297) or y.shape != (96,16,3) or motion.shape != (32,16,2):
            raise ValueError('Exactly16 independent full96-frame actor histories required')
        p = recurrent_predictions(model, x, state, 64, model.weights(), gradient_start=0)
        heads = focused_heads(p, y[64:], motion, selection['selections']); norms = {}
        loss = optimize(model, optimizer, heads, (8.,4.,8.), True, norms)
        row = {'update': update, 'lossBeforeUpdate': loss,
               'objectiveHeadsBeforeUpdate': heads.detach().cpu().tolist(),
               'gradientNormsBeforeClipping': norms, 'seconds': time.perf_counter()-began,
               'sampling': selection, 'gripObjective': VERSION, 'isServiceEvidence': False}
        del x, y, state, motion, p, heads
        if update % 20 == 0 or update == 60+updates:
            row['trainingFit'] = focus_fit(model, bank)
            row['parameterHash'] = model.checkpoint_hash()
            record(update, row)
        history.append(row)
        print('DECISION_GRIP_UPDATE '+json.dumps(row), flush=True)
    return history


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve canonical runs')
    if type(args.updates) is not int or not 1 <= args.updates <= 40:
        raise ValueError('Bounded correction experiment required')
    source, parent, _ = source_run(args.source)
    publication = args.publication.resolve()
    pm = json.loads((publication/'manifest.json').read_text())
    pf = json.loads((publication/'fit-60.json').read_text())
    candidate = source/'candidate-60.npz'
    if (pm['experiment'] != 'physical-operation-grip-parent-motion-v1'
            or pm['parentParameterHash'] != parent['parentParameterHash']
            or pf['parameterHash'] != parent['parentParameterHash']
            or file_hash(publication/'candidate-60.npz') != file_hash(candidate)):
        raise ValueError('Original canonical C60 publication required')
    # Repeat exact serialized physical replay on the canonical training host.
    histories, data = load_histories(args.dataset, replay=True)
    if (len(histories) != 16 or data['canonicalRun'] != publication.name or data['canonicalVersion'] != 60
            or data['parameterHash'] != parent['parentParameterHash']
            or data['candidateFileHash'] != file_hash(candidate)
            or any(meta['frames'] != 12000 for _, meta in histories)):
        raise ValueError('Complete16 original canonical600-second training histories required')
    del histories
    torch.set_num_threads(4)
    model = load_model(args.root, candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    if before != parent['parentParameterHash'] or model.fixed_hash != parent['fixedHash']:
        raise ValueError('Wrong retained C60')
    bank = OperationCache(args.cache, args.dataset, before, model.fixed_hash, model.n)
    if (bank.manifest['candidateFileHash'] != file_hash(candidate)
            or bank.manifest['prefixCanonicalRun'] != publication.name
            or bank.manifest['prefixCanonicalVersion'] != 60):
        raise ValueError('Wrong original versioned operation prefix')
    optimizer, rng = restore_state(model, torch.load(source/'optimizer-60.pt', map_location='cpu', weights_only=True),
                                   parent['learningRates'])
    previous_rates = [g['lr'] for g in optimizer.param_groups]
    for group in optimizer.param_groups:
        group['lr'] *= 10
    paths = [source/n for n in ('manifest.json','result.json','status.json','fit-60.json','candidate-60.npz','optimizer-60.pt')]
    paths += [publication/n for n in ('manifest.json','fit-60.json','candidate-60.npz')]
    paths += [args.cache/'manifest.json', args.cache/'cache.npz', args.dataset/'manifest.json']
    paths += [args.dataset/e['file'] for e in bank.manifest['datasetFiles']]
    inputs = {str(p.resolve()): file_hash(p) for p in paths}
    names = set(parent['sourceHashes']) | set(bank.manifest['sourceHashes']) | {
        'layout_decision_grip_train.py','layout_operation_focus.py','layout_operation_sampling.py',
        'layout_correction_curriculum_train.py','layout_demonstration_step_probe.py'}
    sources = {n: file_hash(Path(__file__).parent/n) for n in sorted(names)}
    initial = focus_fit(model, bank)
    if max(initial['parentMotionMSE']) > 1e-8:
        raise ValueError('Parent anchors disagree with frozen parent')
    args.out.mkdir(parents=True)
    manifest = {'experiment': VERSION, 'startUpdate': 60, 'finalUpdate': 60+args.updates,
        'sourceAdamRun': str(source), 'canonicalParentPublication': str(publication),
        'parentParameterHash': before, 'fixedHash': model.fixed_hash, 'interface': model.interface,
        'sourceHashes': sources, 'inputFileHashes': inputs,
        'optimizerMomentsPreserved': True, 'samplerRNGPreserved': True, 'explicitRateFactor': 10,
        'previousLearningRates': previous_rates, 'learningRates': [g['lr'] for g in optimizer.param_groups],
        'canonicalOptimizer': 'Spark1 only', 'gradientFrames': 96, 'motionLossFrames': 32,
        'gripLossFrames': 1, 'gripFocusFrameRelativeToLossStart': 16, 'maximumPrefixAgeUpdates': 40,
        'objective': '8*parent-speed-MSE+4*parent-turn-MSE+8*class-balanced focus-frame grip hinge(margin0.1)',
        'datasetEpisodes': 16, 'serializedPhysicalReplayOnTrainingHostPassed': True,
        'labelsUnchanged': True, 'allContextsEveryUpdate': True,
        'teacherAtInference': False, 'externalDecisionNetwork': False, 'decoderTrained': False,
        'dopamineLearning': False, 'learning': 'supervised recurrent backpropagation; exact ReLU derivative',
        'isServiceEvidence': False}
    atomic_json(args.out/'manifest.json', manifest)
    initial_gains = model.log_gains.detach().cpu().numpy().copy()
    initial_tonic = model.tonic.detach().cpu().numpy().copy()
    last_saved = 60
    def record(update, row):
        nonlocal last_saved
        save_model(args.out/f'candidate-{update}.npz', model)
        torch.save({'optimizer': optimizer.state_dict(), 'updates': update, 'rng': rng.bit_generator.state},
                   args.out/f'optimizer-{update}.pt')
        atomic_json(args.out/f'fit-{update}.json', row)
        atomic_json(args.out/'status.json', {'finished': False, **row})
        last_saved = update
    try:
        record(60, {'update': 60, 'parameterHash': before, 'trainingFit': initial, 'isServiceEvidence': False})
        history = train(model, bank, optimizer, rng, args.updates, record)
        if (any(file_hash(Path(p)) != h for p, h in inputs.items())
                or any(file_hash(Path(__file__).parent/n) != h for n, h in sources.items())):
            raise ValueError('Source/data changed during training')
        result = {'startUpdate': 60, 'updates': 60+args.updates, 'history': history,
                  'finalParameterHash': model.checkpoint_hash(), 'audit': model.audit(initial_gains),
                  'changedNeurons': int(np.count_nonzero(model.tonic.detach().cpu().numpy() != initial_tonic)),
                  'sourceFilesUnchanged': True, 'isServiceEvidence': False}
        atomic_json(args.out/'result.json', result)
        atomic_json(args.out/'status.json', {'finished': True, **history[-1]})
        print('DECISION_GRIP_FINISHED '+json.dumps({k:v for k,v in result.items() if k != 'history'}), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'type': type(error).__name__, 'error': str(error), 'lastSaved': last_saved})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','source','publication','cache','dataset','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--updates', type=int, default=40)
    main(p.parse_args())
