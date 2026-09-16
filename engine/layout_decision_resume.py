"""Strict continuation state and immutable original motion-anchor selection."""
import json
from pathlib import Path
import numpy as np
import torch
from .layout_cooldown_continuation import restore_adam
from .layout_operation_sampling import OperationCache
from .layout_operation_focus import VERSION
from .layout_recovery_demonstrations import file_hash


def validate_records(manifest, result, status, fit):
    expected = {'experiment': VERSION, 'startUpdate': 60, 'finalUpdate': 100,
                'gradientFrames': 96, 'gripLossFrames': 1, 'motionLossFrames': 32,
                'gripFocusFrameRelativeToLossStart': 16, 'maximumPrefixAgeUpdates': 40,
                'datasetEpisodes': 16, 'serializedPhysicalReplayOnTrainingHostPassed': True,
                'labelsUnchanged': True, 'optimizerMomentsPreserved': True,
                'samplerRNGPreserved': True, 'teacherAtInference': False,
                'externalDecisionNetwork': False, 'decoderTrained': False, 'dopamineLearning': False}
    if (any(manifest.get(k) != v for k,v in expected.items())
            or not status.get('finished') or status['update'] != 100
            or result['startUpdate'] != 60 or result['updates'] != 100
            or result.get('sourceFilesUnchanged') is not True
            or [r['update'] for r in result['history']] != list(range(61,101))
            or fit != result['history'][-1]
            or fit['parameterHash'] != status['parameterHash']
            or fit['parameterHash'] != result['finalParameterHash']):
        raise ValueError('Require original completed V25 C100 publication')
    if (not all(result['audit'][k] for k in ('finite','signsPreserved','fixedGraphSensoryDecoderDynamicsUnchanged'))
            or any(not np.isfinite(r['lossBeforeUpdate']) or r['lossBeforeUpdate'] < 0
                   or set(r['gradientNormsBeforeClipping']) != {'log_gains','tonic'}
                   or any(not np.isfinite(g) or g <= 0 for g in r['gradientNormsBeforeClipping'].values())
                   for r in result['history'])):
        raise ValueError('Source training/fixed-graph audit failed')


def source_run(folder):
    p = Path(folder).resolve()
    m,r,s,f = [json.loads((p/n).read_text()) for n in ('manifest.json','result.json','status.json','fit-100.json')]
    validate_records(m,r,s,f)
    if ((p/'failure.json').exists()
            or any(file_hash(Path(n)) != h for n,h in m['inputFileHashes'].items())
            or any(Path(n).name != n or file_hash(Path(__file__).parent/n) != h for n,h in m['sourceHashes'].items())):
        raise ValueError('Original V25 source/input changed')
    return p,m,r,f


def restore_state(model, payload, rates):
    if set(payload) != {'optimizer','updates','rng'} or payload['updates'] != 100:
        raise ValueError('Require saved C100 Adam and sampler RNG')
    optimizer = torch.optim.Adam([{'params':[model.log_gains],'lr':rates[0],'eps':1e-14},
                                  {'params':[model.tonic],'lr':rates[1],'eps':1e-10}])
    restore_adam(optimizer, {'optimizer':payload['optimizer'],'updates':100}, 100)
    rng = np.random.default_rng()
    rng.bit_generator.state = payload['rng']
    return optimizer,rng


class AnchoredOperationCache(OperationCache):
    """New exact C100 prefix, but keep C60 motion targets without any mutation."""
    def __init__(self, fresh, original):
        keys = ('windows','datasetManifestHash','datasetFiles','fixedHash','behaviorParameterHash',
                'behaviorCanonicalRun','behaviorCanonicalVersion')
        if (any(fresh.manifest[k] != original.manifest[k] for k in keys)
                or not np.array_equal(fresh.x,original.x) or not np.array_equal(fresh.y,original.y)
                or fresh.states.shape != original.states.shape or fresh.motion.shape != original.motion.shape
                or fresh.manifest['parameterHash'] == original.manifest['parameterHash']):
            raise ValueError('Only prefix refresh on identical original physical histories is allowed')
        self.__dict__ = fresh.__dict__.copy()
        self.motion = original.motion
        self.manifest = dict(fresh.manifest)
        self.manifest['motionAnchorParameterHash'] = original.manifest['parameterHash']
        self.manifest['motionTargets'] = 'Retained original C60 motion targets; only the state prefix is refreshed'
