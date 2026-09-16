"""Bounded whole-service curriculum; one canonical Spark1 optimizer.

Explicitly resume a retained V20 checkpoint and its Adam moments, then change
the TRAINING objective from pickup-only/parent-motion preservation to balanced
physical teacher motion plus timing-aware grip across every service context.
Spark2 supplies a checkpoint-versioned full-history prefix cache. This offline
phase has NO online packet consumer, teacher at inference, or external decoder.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_cooldown_continuation import restore_adam
from .layout_service_balanced_sampling import ServiceBalancedDemonstrations
from .layout_demonstration_train import DemonstrationCache, demonstration_fit
from .layout_recovery_train import recurrent_predictions, motor_head_errors, optimize
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def train_curriculum(model, sampler, optimizer, rng, start, updates, record):
    if type(updates) is not int or not 1 <= updates <= 80:
        raise ValueError('Hard1..80 update prefix-age bound required')
    history, began = [], time.perf_counter()
    model.train().requires_grad_(True)
    for update in range(start+1, start+updates+1):
        optimizer.zero_grad(set_to_none=True)
        x, y, state, selection = sampler.sample(rng, model.tonic.device)
        if x.shape[:2] != (96, 16) or y.shape != (96, 16, 3):
            raise ValueError('Exactly16 independently recurrent96-frame actor histories required')
        p = recurrent_predictions(model, x, state, 64, model.weights(), gradient_start=0)
        heads = motor_head_errors(p, y[64:], balanced_grip=True)
        norms = {}
        loss = optimize(model, optimizer, heads, (4., 2., 2.), True, norms)
        row = {'update': update, 'lossBeforeUpdate': loss, 'seconds': time.perf_counter()-began,
               'gradientNormsBeforeClipping': norms, 'objectiveByHeadBeforeUpdate': heads.detach().cpu().tolist(),
               'sampling': selection, 'isServiceEvidence': False}
        del x, y, state, p, heads
        if update % 20 == 0 or update == start+updates:
            model.eval()
            row.update(parameterHash=model.checkpoint_hash(), trainingFit=demonstration_fit(model, sampler, 64))
            record(update, row)
            model.train()
        history.append(row)
        print('SERVICE_CURRICULUM_UPDATE '+json.dumps(row), flush=True)
    return history


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve all training runs')
    if not 1 <= args.updates <= 80 or args.checkpoint not in (20, 40, 60, 80):
        raise ValueError('Bounded segment and saved original V20 checkpoint required')
    source = args.source.resolve()
    source_manifest = json.loads((source/'manifest.json').read_text())
    source_result = json.loads((source/'result.json').read_text())
    source_status = json.loads((source/'status.json').read_text())
    start = args.checkpoint
    if (source_manifest.get('onlyMatchedExperimentChange') !=
            'Extend grip labels by200ms where exact physical replay proves cooldown ignores them'
            or source_manifest.get('optimizerFreshByDesign') is not True
            or source_status.get('finished') is not True or source_result['updates'] != 80
            or source_status['update'] != 80 or source_result['finalParameterHash'] != source_status['parameterHash']
            or [r['update'] for r in source_result['history']] != list(range(1,81))
            or not all(np.isfinite(r['lossBeforeUpdate']) for r in source_result['history'])
            or not all(source_result['audit'][k] for k in
                       ('finite','signsPreserved','fixedGraphSensoryDecoderDynamicsUnchanged'))):
        raise ValueError('Require completed original V20 source; no silent optimizer restart')
    candidate = source/f'candidate-{start}.npz'
    fit = json.loads((source/f'fit-{start}.json').read_text())
    if fit['parameterHash'] != source_result['history'][start-1]['parameterHash']:
        raise ValueError('Selected saved checkpoint differs from completed source history')
    engine = Path(__file__).parent
    if any(file_hash(engine/n) != h for n,h in source_manifest['sourceHashes'].items()):
        raise ValueError('Source V20 training implementation changed')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)  # Newly introduced sampler; V20 had none.
    model = load_model(args.root, candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    if before != fit['parameterHash'] or model.fixed_hash != source_manifest['fixedHash']:
        raise ValueError('Selected brain or fixed interface changed')
    bank = DemonstrationCache(args.cache, before, model.fixed_hash, 64, 32, model.n)
    cm = bank.manifest
    if (cm.get('generatorVersion') != 'independent-batched-full-prefix-v1'
            or cm['candidateFileHash'] != file_hash(candidate)
            or cm['sourceHash'] != file_hash(engine/'layout_batched_prefix_cache.py')
            or any(file_hash(engine/n) != h for n,h in cm['helperSourceHashes'].items())):
        raise ValueError('Full-history Spark2 cache provenance changed')
    sampler = ServiceBalancedDemonstrations(bank, args.dataset, args.labels)
    optimizer = torch.optim.Adam([{'params':[model.log_gains], 'lr':source_manifest['lr'], 'eps':1e-14},
                                  {'params':[model.tonic], 'lr':source_manifest['tonic_lr'], 'eps':1e-10}])
    restore_adam(optimizer, torch.load(source/f'optimizer-{start}.pt', map_location='cpu', weights_only=True), start)
    previous_rates = [g['lr'] for g in optimizer.param_groups]
    # Explicit10x reduction for broader targets to limit abrupt forgetting.
    for group in optimizer.param_groups:
        group['lr'] *= .1
    paths = [source/n for n in ('manifest.json','result.json','status.json',f'fit-{start}.json',
                                f'candidate-{start}.npz',f'optimizer-{start}.pt')]
    paths += [args.cache/n for n in ('manifest.json','cache.npz')]
    paths += [args.dataset/'manifest.json', args.labels/'manifest.json']
    paths += [args.dataset/e['file'] for e in sampler.revision['originalDatasetFiles']]
    paths += [args.labels/e['file'] for e in sampler.revision['episodes']]
    inputs = {str(p.resolve()):file_hash(p) for p in paths}
    names = set(source_manifest['sourceHashes']) | set(cm['sourceHashes']) | set(cm['helperSourceHashes'])
    names |= set(sampler.revision['originalPhysicsSourceHashes']) | set(sampler.revision['sourceHashes'])
    names |= {'layout_service_curriculum_train.py','layout_service_balanced_sampling.py',
              'layout_cooldown_sampling.py','layout_cooldown_continuation.py','layout_batched_prefix_cache.py'}
    sources = {n:file_hash(engine/n) for n in sorted(names)}
    args.out.mkdir(parents=True)
    manifest = {'experiment':'service-balanced-cooldown-curriculum-v1', 'sourceRun':str(source),
                'startUpdate':start, 'finalUpdate':start+args.updates, 'parentParameterHash':before,
                'fixedHash':model.fixed_hash, 'interface':model.interface, 'inputFileHashes':inputs,
                'sourceHashes':sources, 'sampler':sampler.record, 'optimizerMomentsPreserved':True,
                'previousLearningRates':previous_rates, 'learningRates':[g['lr'] for g in optimizer.param_groups],
                'explicitLearningRateFactor':.1, 'seed':args.seed, 'newSamplerRNGByDesign':True,
                'objective':'4*teacher-speed-MSE+2*teacher-turn-MSE+2*balanced-timing-grip-MSE/contrast',
                'previousObjective':'pickup microfit with parent-motion preservation',
                'gradientFrames':96, 'lossFrames':32, 'maximumPrefixAgeUpdates':80,
                'canonicalOptimizer':'Spark1 only', 'spark2Role':'frozen versioned physical-training prefix generation',
                'onlineCorrectionsUsed':False, 'teacherAtInference':False, 'externalDecisionNetwork':False,
                'decoderTrained':False, 'dopamineLearning':False, 'isServiceEvidence':False,
                'learning':'supervised recurrent backpropagation; exact ReLU derivative'}
    atomic_json(args.out/'manifest.json', manifest)
    initial_gains = model.log_gains.detach().cpu().numpy().copy()
    initial_tonic = model.tonic.detach().cpu().numpy().copy()
    last_saved = start
    def record(update, row):
        nonlocal last_saved
        save_model(args.out/f'candidate-{update}.npz', model)
        torch.save({'optimizer':optimizer.state_dict(), 'updates':update, 'rng':rng.bit_generator.state},
                   args.out/f'optimizer-{update}.pt')
        atomic_json(args.out/f'fit-{update}.json', row)
        atomic_json(args.out/'status.json', {'finished':False, **row})
        last_saved = update
    try:
        record(start, {'update':start, 'parameterHash':before,
                       'trainingFit':demonstration_fit(model, sampler, 64), 'isServiceEvidence':False})
        history = train_curriculum(model, sampler, optimizer, rng, start, args.updates, record)
        if (any(file_hash(Path(p)) != h for p,h in inputs.items())
                or any(file_hash(engine/n) != h for n,h in sources.items())):
            raise ValueError('Original inputs or source changed during curriculum')
        result = {'startUpdate':start, 'updates':start+args.updates, 'history':history,
                  'finalParameterHash':model.checkpoint_hash(), 'audit':model.audit(initial_gains),
                  'changedNeurons':int(np.count_nonzero(model.tonic.detach().cpu().numpy()!=initial_tonic)),
                  'sourceFilesUnchanged':True, 'isServiceEvidence':False}
        atomic_json(args.out/'result.json', result)
        atomic_json(args.out/'status.json', {'finished':True, **history[-1]})
        print('SERVICE_CURRICULUM_FINISHED '+json.dumps({k:v for k,v in result.items() if k!='history'}), flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json', {'type':type(error).__name__, 'error':str(error), 'lastSaved':last_saved})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root','source','dataset','cache','labels','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoint', type=int, default=60)
    parser.add_argument('--updates', type=int, default=80)
    parser.add_argument('--seed', type=int, default=9390022)
    main(parser.parse_args())
