"""Audited optimizer continuation with a freshly rebuilt full-prefix cache.

Only task-owned finished runs may resume. Worker worlds restart explicitly;
this preserves Adam and the demonstration sampler, not physical episode state.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .layout_recovery_demonstrations import file_hash
from .layout_demonstration_curriculum import validate_curriculum_change


CONTROLS = ('burn', 'gradient_frames', 'lr', 'tonic_lr', 'seed',
            'recent_corrections', 'require_stratified_corrections')
SOURCES = ('layout_demonstration_cache.py', 'layout_recovery_train.py',
           'layout_recovery_worker.py', 'layout_recovery_protocol.py',
           'layout_recovery_brain.py', 'layout_recovery_world.py',
           'layout_recovery_teacher.py', 'layout_excitability.py',
           'supervised_steering.py', 'layout_recent_corrections.py',
           'layout_recovery_curriculum.py', 'layout_stratified_worker.py')


def prepare_continuation(args, model, bank):
    source = Path(args.resume_demonstration_run).resolve()
    if source == args.out.resolve():
        raise ValueError('Continuation requires a separate output directory')
    manifest, result, status, history = [json.loads((source/name).read_text()) for name in
                                       ('manifest.json', 'result.json', 'status.json', 'history.json')]
    completed = result['updates']
    previous_start = manifest.get('startUpdate', 0)
    if (type(completed) is not int or completed <= previous_start or status.get('finished') is not True
            or status.get('version') != completed or status.get('update') != completed
            or [r['update'] for r in history] != list(range(previous_start+1, completed+1))
            or any(r['source'] not in ('frozen-prefix-physical-teacher-demonstration',
                                      'spark2-learner-only-correction') for r in history)
            or manifest.get('learning') != 'supervised recurrent backpropagation; exact ReLU derivative'
            or manifest.get('canonicalOptimizer') != 'Spark1 only'
            or any(result['audit'].get(key) is not True for key in
                   ('finite', 'signsPreserved', 'fixedGraphSensoryDecoderDynamicsUnchanged'))):
        raise ValueError('Only a completed, audited demonstration run can resume')
    start = getattr(args, 'resume_checkpoint_update', None)
    start = completed if start is None else start
    if type(start) is not int or not previous_start < start <= completed:
        raise ValueError('Selected checkpoint must be a saved update within the finished run')
    for key in CONTROLS:
        if manifest.get(key, False) != getattr(args, key, False):
            raise ValueError('Continuation changes control: '+key)
    for name in SOURCES:
        if manifest['sourceHashes'].get(name) != file_hash(Path(__file__).parent/name):
            raise ValueError('Continuation changes training/model source: '+name)
    final_path = source/f'candidate-{completed}.npz'
    if (status.get('runId') != manifest['runId'] or status['checkpoint']['file'] != final_path.name
            or result['checkpointHash'] != status['checkpoint']['parameterHash']
            or file_hash(final_path) != status['checkpoint']['sha256']
            or manifest['fixedHash'] != model.fixed_hash or status['fixedHash'] != model.fixed_hash):
        raise ValueError('Completed source run identity mismatch')
    checkpoint = status['checkpoint']
    extra_paths = []
    selected_audit = None
    if start != completed:
        # Historical publication records attest the original checkpoint bytes.
        # Never rename an intermediate candidate or rewrite a finished run.
        log_path = source.parent/(source.name+'.log')
        publications = [json.loads(line[len('CHECKPOINT '):]) for line in log_path.read_text().splitlines()
                        if line.startswith('CHECKPOINT ')]
        matches = [item for item in publications if item.get('version') == start]
        if len(matches) != 1:
            raise ValueError('Selected checkpoint requires one historical publication')
        publication = matches[0]
        if (publication.get('runId') != manifest['runId'] or publication.get('update') != start
                or publication.get('fixedHash') != model.fixed_hash
                or publication.get('interface') != model.interface
                or publication.get('physics') != manifest['physics']):
            raise ValueError('Historical checkpoint publication mismatch')
        checkpoint = publication['checkpoint']
        fit_path = source/f'demonstration-fit-{start}.json'
        fit = json.loads(fit_path.read_text())
        if (fit.get('checkpointHash') != checkpoint['parameterHash']
                or fit.get('prefixCheckpointHash') != manifest['prefixCheckpointHash']
                or fit.get('trainingSubsetOnly') is not True or fit.get('isServiceEvidence') is not False):
            raise ValueError('Historical checkpoint diagnostic identity mismatch')
        with torch.no_grad():
            if any(not torch.isfinite(p).all() for p in (model.log_gains, model.tonic)):
                raise ValueError('Nonfinite selected checkpoint')
            if torch.any(model.log_gains.abs() > 2.) or torch.any(model.tonic.abs() > .1):
                raise ValueError('Selected checkpoint outside parameter bounds')
            selected_audit = model.audit(model.log_gains.detach().cpu().numpy())
        if any(selected_audit.get(key) is not True for key in
               ('finite', 'signsPreserved', 'fixedGraphSensoryDecoderDynamicsUnchanged')):
            raise ValueError('Selected checkpoint failed independent model audit')
        extra_paths = [log_path, fit_path, final_path]
    expected = source/f'candidate-{start}.npz'
    before = model.checkpoint_hash()
    if (args.candidate.resolve() != expected or checkpoint['file'] != expected.name
            or checkpoint['parameterHash'] != before or file_hash(expected) != checkpoint['sha256']):
        raise ValueError('Continuation checkpoint/interface mismatch')
    old_cache = Path(manifest['cache'])
    if not old_cache.is_absolute():
        old_cache = source.parent/old_cache
    old_cache = old_cache.resolve()
    if not old_cache.is_relative_to(source.parent):
        raise ValueError('Prior cache must remain inside this experiment workspace')
    if old_cache == args.cache.resolve():
        raise ValueError('A fresh prefix cache is required for continuation')
    old_manifest = old_cache/'manifest.json'
    if file_hash(old_manifest) != manifest['cacheManifestHash']:
        raise ValueError('Original prefix manifest changed')
    old = json.loads(old_manifest.read_text())
    if old['datasetManifestHash'] != manifest['datasetManifestHash']:
        raise ValueError('Original training dataset identity mismatch')
    fresh = bank.manifest
    if (fresh['parameterHash'] != before or fresh['fixedHash'] != model.fixed_hash
            or fresh['candidateFileHash'] != file_hash(expected)
            or fresh['sourceHashes'] != old['sourceHashes']
            or fresh.get('finished') is not True or fresh.get('parametersUnchanged') is not True):
        raise ValueError('Fresh prefix cache does not preserve the model/parent')
    dataset_change = None
    new_dataset = getattr(args, 'resume_new_demonstration_dataset', None)
    if new_dataset:
        if not Path(new_dataset).resolve().is_relative_to(source.parent):
            raise ValueError('New training dataset must remain inside this experiment workspace')
        dataset_change = validate_curriculum_change(new_dataset, bank, old)
    elif (fresh['datasetManifestHash'] != manifest['datasetManifestHash']
            or fresh['datasetFiles'] != old['datasetFiles'] or fresh['windows'] != old['windows']):
        raise ValueError('Fresh prefix cache does not preserve the training data')
    optimizer_path = source/f'optimizer-{start}.pt'
    paths = [expected, optimizer_path, source/'manifest.json', source/'result.json',
             source/'status.json', source/'history.json', *extra_paths]
    record = {'sourceRun': str(source), 'sourceRunId': manifest['runId'], 'startUpdate': start,
              'sourceFinalUpdate': completed, 'selectedIntermediateCheckpoint': start != completed,
              'selectedCheckpointAudit': selected_audit,
              'parameterHash': before, 'optimizerMomentsPreserved': True,
              'demonstrationSamplerStatePreserved': True, 'sameDemonstrationDataset': dataset_change is None,
              'demonstrationDatasetChange': dataset_change,
              'physicalWorldsResumed': False, 'newWorkerEpisodes': True,
              'refreshedPrefixManifestHash': file_hash(args.cache/'manifest.json'),
              'sourceFileSHA256': {p.name: file_hash(p) for p in paths}}
    return {'start': start, 'optimizerPath': optimizer_path, 'record': record}


def restore_continuation_optimizer(optimizer, continuation):
    path = continuation['optimizerPath']
    if file_hash(path) != continuation['record']['sourceFileSHA256'][path.name]:
        raise ValueError('Optimizer file changed during continuation setup')
    # Task-owned optimizer files only; network transport still accepts numeric NPZ.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['updates'] != continuation['start']:
        raise ValueError('Optimizer checkpoint counter mismatch')
    saved = payload['optimizer']
    expected = optimizer.state_dict()['param_groups']
    groups = saved['param_groups']
    if len(groups) != len(expected):
        raise ValueError('Optimizer parameter groups mismatch')
    ids = []
    for actual, wanted, live in zip(groups, expected, optimizer.param_groups):
        if ({k: v for k, v in actual.items() if k != 'params'} !=
                {k: v for k, v in wanted.items() if k != 'params'}
                or len(actual['params']) != len(live['params'])):
            raise ValueError('Optimizer controls or parameter layout mismatch')
        for identifier, parameter in zip(actual['params'], live['params']):
            ids.append(identifier)
            state = saved['state'][identifier]
            step = state['step']
            if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != continuation['start']:
                raise ValueError('Adam counter mismatch')
            for key in ('exp_avg', 'exp_avg_sq'):
                value = state[key]
                if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                    raise ValueError('Malformed Adam moment')
            if torch.any(state['exp_avg_sq'] < 0):
                raise ValueError('Negative Adam second moment')
    if len(set(ids)) != len(ids) or set(saved['state']) != set(ids):
        raise ValueError('Optimizer state ownership mismatch')
    rng = np.random.default_rng()
    rng.bit_generator.state = payload['rng']  # Validate before touching optimizer.
    optimizer.load_state_dict(saved)
    return rng.bit_generator.state
